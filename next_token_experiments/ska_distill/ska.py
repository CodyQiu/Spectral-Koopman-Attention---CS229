"""Structured Kernel Attention module used in the retrofit experiment."""

import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from contextlib import nullcontext

# ============================================================================
# Backend detection
# ============================================================================

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Shared utilities
# ============================================================================

def _spectral_normalize_power_iter(A, n_iters=6):
    """Spectral normalization via power iteration."""
    v = torch.ones(*A.shape[:-1], 1, device=A.device, dtype=A.dtype) / math.sqrt(A.shape[-1])
    with torch.no_grad():
        for _ in range(n_iters):
            Av = A @ v
            u = Av / Av.norm(dim=-2, keepdim=True).clamp(min=1e-8)
            Atu = A.transpose(-1, -2) @ u
            v = Atu / Atu.norm(dim=-2, keepdim=True).clamp(min=1e-8)
        sigma_max = (A @ v).norm(dim=-2, keepdim=False).squeeze(-1)
    scale = torch.clamp(sigma_max, min=1.0).unsqueeze(-1).unsqueeze(-1)
    return A / scale, sigma_max


def _power_spectral_filter(A_w, w_q, power_K=4):
    """Apply A_w^power_K @ w_q by iterating on w_q."""
    result = w_q
    for _ in range(power_K):
        result = A_w @ result
    return result


# ============================================================================
# Shared: batched statistics + cumsum + cholesky
# ============================================================================

def _compute_chunk_stats_and_cholesky(z_f, zq_f, v_f, r, H, P, CS, ridge_eps):
    """Build chunked sufficient statistics for SKA."""
    B, T = z_f.shape[:2]
    device = z_f.device
    dtype = z_f.dtype

    n_chunks = (T + CS - 1) // CS
    T_padded = n_chunks * CS
    pad_len = T_padded - T

    if pad_len > 0:
        z_f = torch.nn.functional.pad(z_f, (0, 0, 0, 0, 0, pad_len))
        zq_f = torch.nn.functional.pad(zq_f, (0, 0, 0, 0, 0, pad_len))
        v_f = torch.nn.functional.pad(v_f, (0, 0, 0, 0, 0, pad_len))

    C = n_chunks
    z_c = z_f.reshape(B, C, CS, H, r)
    zq_c = zq_f.reshape(B, C, CS, H, r)
    v_c = v_f.reshape(B, C, CS, H, P)

    # Per-chunk statistics
    G_chunks = torch.einsum('bcthr,bcths->bchrs', z_c, z_c)
    M_chunks = torch.einsum('bcthr,bcths->bchrs', z_c[:, :, 1:], z_c[:, :, :-1])
    C_chunks = torch.einsum('bcthp,bcthr->bchpr', v_c, z_c)

    # Cross-chunk boundary transitions
    if C > 1:
        M_boundary = torch.einsum('bchr,bchs->bchrs',
                                  z_c[:, 1:, 0], z_c[:, :-1, -1])

    # Exclusive prefix sum
    eye_r = torch.eye(r, device=device, dtype=dtype)

    G_cumsum = torch.cumsum(G_chunks, dim=1)
    G_excl = torch.zeros_like(G_cumsum)
    G_excl[:, 1:] = G_cumsum[:, :-1]
    G_excl = G_excl + ridge_eps * eye_r

    M_cumsum = torch.cumsum(M_chunks, dim=1)
    M_excl = torch.zeros_like(M_cumsum)
    M_excl[:, 1:] = M_cumsum[:, :-1]

    if C > 1:
        M_bnd_full = torch.zeros(B, C, H, r, r, device=device, dtype=dtype)
        M_bnd_full[:, 1:] = M_boundary
        M_bnd_inclusive = torch.cumsum(M_bnd_full, dim=1)
        M_excl = M_excl + M_bnd_inclusive

    C_cumsum = torch.cumsum(C_chunks, dim=1)
    C_excl = torch.zeros_like(C_cumsum)
    C_excl[:, 1:] = C_cumsum[:, :-1]

    # Return raw G (Cholesky moved to implicit backward function)
    BCH = B * C * H
    G_flat = G_excl.reshape(BCH, r, r)
    G_flat = 0.5 * (G_flat + G_flat.transpose(-1, -2))

    M_flat = M_excl.reshape(BCH, r, r)
    C_flat = C_excl.reshape(BCH, P, r)
    zq_flat = zq_c.permute(0, 1, 3, 2, 4).reshape(BCH, CS, r)  # [BCH, S, r]

    return G_flat, M_flat, C_flat, zq_flat, (B, C, H, P, CS, T, T_padded, pad_len), (z_c, zq_c, v_c)


def _intra_chunk_causal_attention(z_c, zq_c, v_c, shapes):
    """Causal dot-product attention within each chunk."""
    B, C, H, P, CS, T, T_padded, pad_len = shapes

    # Reshape for attention: [B*C, H, CS, dim]
    BC = B * C
    q = zq_c.reshape(BC, CS, H, -1).permute(0, 2, 1, 3)  # [BC, H, CS, r]
    k = z_c.reshape(BC, CS, H, -1).permute(0, 2, 1, 3)    # [BC, H, CS, r]
    v = v_c.reshape(BC, CS, H, -1).permute(0, 2, 1, 3)    # [BC, H, CS, P]

    r = q.shape[-1]
    scale = 1.0 / math.sqrt(r)

    # Compute attention scores: [BC, H, CS, CS]
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale

    # Causal mask: lower triangular (each token sees itself and earlier tokens)
    causal_mask = torch.triu(
        torch.full((CS, CS), float('-inf'), device=q.device, dtype=q.dtype),
        diagonal=1,
    )
    attn = attn + causal_mask

    attn = torch.softmax(attn, dim=-1)

    # Apply attention to values: [BC, H, CS, P]
    y = torch.matmul(attn, v)

    # Reshape back: [B, T, H, P]
    y = y.permute(0, 2, 1, 3).reshape(B, C, CS, H, P)
    y = y.reshape(B, T_padded, H, P)
    if pad_len > 0:
        y = y[:, :T]
    return y


def _post_cholesky_pytorch(L_flat, M_flat, C_flat, zq_flat, ssn_gamma,
                           power_K, shapes, z_c=None, zq_c=None, v_c=None,
                           intra_weight=0.5):
    """Post-Cholesky SKA retrieval using batched PyTorch calls."""
    B, C, H, P, CS, T, T_padded, pad_len = shapes

    # Cross-chunk Koopman retrieval (existing code)
    # Whitened Koopman operator: A_w = L^{-1} M L^{-T}
    Z = torch.linalg.solve_triangular(L_flat, M_flat, upper=False)
    A_w = torch.linalg.solve_triangular(L_flat, Z.mT, upper=False).mT
    A_w, _ = _spectral_normalize_power_iter(A_w)
    gamma_safe = 0.3 + 0.7 * torch.sigmoid(ssn_gamma)  # smooth (0.3, 1.0), no dead gradients
    A_w = A_w * gamma_safe

    Bv_T = torch.cholesky_solve(C_flat.transpose(-1, -2), L_flat)
    B_v = Bv_T.transpose(-1, -2)

    w_q = torch.linalg.solve_triangular(L_flat, zq_flat, upper=False)

    w_f = _power_spectral_filter(A_w, w_q, power_K)
    z_out = L_flat @ w_f
    y_flat = B_v @ z_out

    y_koopman = y_flat.reshape(B, C, H, P, CS).permute(0, 1, 4, 2, 3)
    y_koopman = y_koopman.reshape(B, T_padded, H, P)
    if pad_len > 0:
        y_koopman = y_koopman[:, :T]

    # Intra-chunk causal attention
    if z_c is not None and zq_c is not None and v_c is not None:
        y_intra = _intra_chunk_causal_attention(z_c, zq_c, v_c, shapes)
        y_hat = (1.0 - intra_weight) * y_koopman + intra_weight * y_intra
    else:
        y_hat = y_koopman

    return y_hat


# ============================================================================
# SKA Module
# ============================================================================


class KoopmanMLP(nn.Module):
    """Spectral Koopman MLP block."""
    def __init__(self, d_in, expansion=1.0):
        super().__init__()
        # d_k rounded to nearest multiple of 64
        d_k = int(math.ceil(d_in * expansion / 64)) * 64
        if d_k % 2 != 0:
            d_k += 64  # ensure even for pair-wise rotation
        self.d_k = d_k
        
        self.lift = nn.Linear(d_in, d_k, bias=False)
        self.readout = nn.Linear(d_k, d_in, bias=False)
        
        # Complex eigenvalues: gamma + i*omega per pair
        n_pairs = d_k // 2
        self.gamma = nn.Parameter(torch.ones(n_pairs) * 0.9)
        self.omega = nn.Parameter(torch.zeros(n_pairs))
        
        nn.init.zeros_(self.readout.weight)
        nn.init.xavier_uniform_(self.lift.weight, gain=0.1)
    
    def forward(self, h):
        g = F.silu(self.lift(h))  # [B, T, d_k]
        
        # Clamp eigenvalue modulus to unit disk
        rho = torch.sqrt(self.gamma ** 2 + self.omega ** 2).clamp(min=1e-6)
        scale = torch.clamp(rho, max=1.0) / rho
        gamma_c = self.gamma * scale
        omega_c = self.omega * scale
        
        # Block-diagonal rotation: reshape into pairs
        B, T, _ = g.shape
        g_pairs = g.reshape(B, T, -1, 2)  # [B, T, n_pairs, 2]
        
        # Apply 2x2 rotation per pair:
        # [z1, z2] = [[gamma, omega], [-omega, gamma]] @ [g1, g2]
        g1 = g_pairs[..., 0]  # [B, T, n_pairs]
        g2 = g_pairs[..., 1]
        
        z1 = gamma_c * g1 + omega_c * g2
        z2 = -omega_c * g1 + gamma_c * g2
        
        z = torch.stack([z1, z2], dim=-1).reshape(B, T, -1)  # [B, T, d_k]
        
        return h + self.readout(z)

class SKAModule(nn.Module):
    """Drop-in SKA replacement for transformer attention."""
    def __init__(self, d_model, n_heads, rank=48, head_dim=None,
                 ridge_eps=1e-3, scale=1.5, power_K=4, chunk_size=256,
                 num_koopman_mlps=2, backend='auto'):
        super().__init__()
        self.rank = rank
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        self.H = n_heads
        self.P = head_dim or (d_model // n_heads)
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.num_koopman_mlps = num_koopman_mlps

        if backend == 'auto':
            self.backend = 'triton' if _TRITON_AVAILABLE else 'pytorch'
        else:
            self.backend = backend

        self.key_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.query_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.value_proj = nn.Linear(d_model, n_heads * self.P, bias=False)
        self.koopman_mlps = nn.ModuleList([
            KoopmanMLP(n_heads * self.P, expansion=1.0)
            for _ in range(num_koopman_mlps)
        ])

        self.out_proj = nn.Linear(n_heads * self.P, d_model, bias=False)

        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)

        self.eta_raw = nn.Parameter(torch.tensor(-1.2))
        self.ssn_gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, state=None):
        """Forward pass with optional streaming state."""
        B, T, _ = hidden_states.shape
        r = self.rank
        H, P = self.H, self.P
        CS = self.chunk_size

        z = self.key_proj(hidden_states).reshape(B, T, H, r)
        zq = self.query_proj(hidden_states).reshape(B, T, H, r)
        v = self.value_proj(hidden_states).reshape(B, T, H, P)

        # If state is provided, use streaming mode
        if state is not None:
            return self._forward_streaming(z, zq, v, hidden_states, state)

        ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda \
              else nullcontext()

        with ctx:
            z_f = z.float()
            zq_f = zq.float()
            v_f = v.float()

            max_norm = z_f.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            z_f = z_f / max_norm
            zq_f = zq_f / max_norm

            G_flat, M_flat, C_flat, zq_flat, shapes, chunks = \
                _compute_chunk_stats_and_cholesky(
                    z_f, zq_f, v_f, r, H, P, CS, self.ridge_eps)
            z_c, zq_c, v_c = chunks

            from ska_implicit import BlockCausalSKAFunction, _spectral_norm_scale

            # Spectral normalization (detached). ssn_gamma upcast to fp32 so
            # bf16 doesn't quantize the sigmoid derivative to zero.
            if self.power_K > 0:
                with torch.no_grad():
                    L_tmp = torch.linalg.cholesky(G_flat.detach())
                gamma_safe = 0.25 + 0.75 * torch.sigmoid(self.ssn_gamma.float())
                with torch.no_grad():
                    alpha = _spectral_norm_scale(M_flat.detach(), L_tmp, gamma_safe.detach())
            else:
                alpha = torch.ones(G_flat.shape[0], device=G_flat.device, dtype=G_flat.dtype)

            # Implicit SKA: y = Cv (G^{-1} M)^K G^{-1} q
            # No autograd through Cholesky
            y_flat = BlockCausalSKAFunction.apply(
                G_flat, M_flat, C_flat, zq_flat, alpha, self.power_K)

            # y_flat: [BCH, CS, P] -> reshape
            B_s, C_s, H_s, P_s, CS_s, T_s, T_pad, pad_len = shapes
            y_hat = y_flat.reshape(B_s, C_s, H_s, CS_s, P_s)
            y_hat = y_hat.permute(0, 1, 3, 2, 4).contiguous()  # [B, C, CS, H, P]
            y_hat = y_hat.reshape(B_s, T_pad, H_s, P_s)
            if pad_len > 0:
                y_hat = y_hat[:, :T_s]

        eta = 1.2 + 1.3 * torch.sigmoid(self.eta_raw.float())
        y_hat = eta * y_hat.to(hidden_states.dtype)
        y_flat = y_hat.reshape(B, T, H * P)
        for mlp in self.koopman_mlps:
            y_flat = mlp(y_flat)
        output = self.out_proj(y_flat)
        return output


    def _forward_streaming(self, z, zq, v, hidden_states, state):
        """Per-token streaming with state accumulation."""
        import torch
        B, T, H, r = z.shape
        P = v.shape[-1]

        # Initialize or unpack state
        if state.get("G") is None:
            device = z.device
            dtype = torch.float32
            state = {
                "G": torch.zeros(B, H, r, r, device=device, dtype=dtype),
                "M": torch.zeros(B, H, r, r, device=device, dtype=dtype),
                "C_v": torch.zeros(B, H, P, r, device=device, dtype=dtype),
                "z_prev": torch.zeros(B, H, r, device=device, dtype=dtype),
                "max_norm": torch.ones(B, 1, 1, 1, device=device, dtype=dtype),
                "t": 0,
            }

        G = state["G"]
        M = state["M"]
        C_v = state["C_v"]
        z_prev = state["z_prev"]
        t = state["t"]

        outputs = []

        for i in range(T):
            z_t = z[:, i].float()     # [B, H, r]
            zq_t = zq[:, i].float()   # [B, H, r]
            v_t = v[:, i].float()     # [B, H, P]

            # Normalize
            norm_t = z_t.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            z_t = z_t / norm_t
            zq_t = zq_t / norm_t

            # Update sufficient statistics
            # G += z_t z_t^T
            G = G + torch.einsum('bhr,bhs->bhrs', z_t, z_t)

            # M += z_t z_prev^T (lag-one cross-covariance)
            if t > 0:
                M = M + torch.einsum('bhr,bhs->bhrs', z_t, z_prev)

            # C_v += v_t z_t^T
            C_v = C_v + torch.einsum('bhp,bhr->bhpr', v_t, z_t)

            # Cholesky factorization
            G_reg = G + self.ridge_eps * torch.eye(r, device=G.device, dtype=G.dtype)
            G_sym = 0.5 * (G_reg + G_reg.transpose(-1, -2))
            L = torch.linalg.cholesky(G_sym)  # [B, H, r, r]

            # Whitened Koopman operator: A_w = L^{-1} M L^{-T}
            Z_aw = torch.linalg.solve_triangular(L, M, upper=False)
            A_w = torch.linalg.solve_triangular(
                L, Z_aw.transpose(-1, -2), upper=False
            ).transpose(-1, -2)

            # Spectral normalization with learned gamma (fp32 sigmoid)
            gamma_safe = 0.25 + 0.75 * torch.sigmoid(self.ssn_gamma.float())
            frob = torch.sqrt((A_w * A_w).sum(dim=(-1, -2), keepdim=True).clamp(min=1e-12))
            scale = torch.maximum(frob, torch.ones_like(frob))
            A_w = A_w / scale * gamma_safe

            # Ridge retrieval: B_v = C_v G^{-1}
            C_v_t = C_v.transpose(-1, -2)  # [B, H, r, P]
            Bv_t = torch.linalg.solve_triangular(L, C_v_t, upper=False)
            Bv_t = torch.linalg.solve_triangular(
                L.transpose(-1, -2), Bv_t, upper=True
            )
            B_v = Bv_t.transpose(-1, -2)  # [B, H, P, r]

            # Query whitening
            w_q = torch.linalg.solve_triangular(
                L, zq_t.unsqueeze(-1), upper=False
            ).squeeze(-1)

            # Power filter
            w_f = w_q
            for _ in range(self.power_K):
                w_f = torch.einsum('bhrs,bhs->bhr', A_w, w_f)

            # Readout
            z_out = torch.einsum('bhrs,bhs->bhr', L, w_f)
            eta_val = 1.2 + 1.3 * torch.sigmoid(self.eta_raw.float())
            y_t = eta_val * torch.einsum('bhpr,bhr->bhp', B_v, z_out)

            outputs.append(y_t)

            z_prev = z_t
            t += 1

        # Stack outputs
        y_hat = torch.stack(outputs, dim=1)  # [B, T, H, P]
        y_hat = y_hat.to(hidden_states.dtype)
        y_flat = y_hat.reshape(B, T, H * P)
        for mlp in self.koopman_mlps:
            y_flat = mlp(y_flat)
        output = self.out_proj(y_flat)

        new_state = {
            "G": G,
            "M": M,
            "C_v": C_v,
            "z_prev": z_prev,
            "max_norm": norm_t,
            "t": t,
        }

        return output, new_state

    def extra_repr(self):
        return (f'd_model={self.d_model}, n_heads={self.H}, rank={self.rank}, '
                f'chunk_size={self.chunk_size}, backend={self.backend}')
