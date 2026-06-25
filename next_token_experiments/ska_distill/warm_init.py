"""Initialization utilities for SKA projection weights."""

import torch
import torch.nn as nn
from typing import Optional


class LogSingularValueResidualMLP(nn.Module):
    """MLP residual scaler for singular values, initialized to SVD-sqrt."""

    def __init__(self, hidden_dim: int = 32, residual_scale: float = 0.3):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def make_features(self, singular_values: torch.Tensor):
        rank = singular_values.shape[0]
        log_sigma = torch.log(singular_values + 1e-8)
        log_sigma_feature = (
            log_sigma - log_sigma.mean()
        ) / (log_sigma.std(unbiased=False) + 1e-8)

        if rank == 1:
            index_feature = torch.zeros(
                1, device=singular_values.device, dtype=singular_values.dtype
            )
        else:
            index_feature = torch.linspace(
                0.0, 1.0, rank,
                device=singular_values.device,
                dtype=singular_values.dtype,
            )

        features = torch.stack([log_sigma_feature, index_feature], dim=1)
        return features, log_sigma

    def forward(self, singular_values: torch.Tensor):
        features, log_sigma = self.make_features(singular_values)
        residual = self.net(features).squeeze(-1)
        log_g = 0.5 * log_sigma + self.residual_scale * residual
        log_g = torch.clamp(log_g, min=-10.0, max=10.0)
        return torch.exp(log_g)


def _state_dict_from_checkpoint(checkpoint, rank: int):
    """Support raw state_dict, {'state_dict': ...}, or rank-keyed checkpoints."""
    if isinstance(checkpoint, dict):
        for key in (rank, str(rank)):
            if key in checkpoint:
                return _state_dict_from_checkpoint(checkpoint[key], rank)
        for key in ("state_dict", "model_state_dict", "mlp_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _load_mlp_scaler(mlp_checkpoint, rank: int, hidden_dim: int,
                     residual_scale: float, device: torch.device) -> nn.Module:
    if mlp_checkpoint is None:
        raise ValueError(
            "strategy='mlp_log_singular_values' requires a trained MLP checkpoint. "
            "Pass mlp_state_path to warm_init_ska_from_attention or --mlp_state_path "
            "to the training/cache script."
        )

    state_dict = _state_dict_from_checkpoint(mlp_checkpoint, rank)
    model = LogSingularValueResidualMLP(
        hidden_dim=hidden_dim,
        residual_scale=residual_scale,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _get_model_dims(model):
    cfg = model.config
    d_model = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, 'head_dim', None) or (d_model // n_heads)
    return d_model, n_heads, head_dim

def _get_layers(model):
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            return lm.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in {type(model).__name__}")

def _get_attn_module(layer):
    for name in ['self_attn', 'attention', 'attn']:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(f"Cannot find attention in {type(layer).__name__}")

def _init_compressed_projection(W_full: torch.Tensor, 
                                target_dim: int, 
                                strategy: str,
                                scale_a: float = 1.0,
                                power_b: float = 0.5,
                                mlp_scaler: Optional[nn.Module] = None,
                                ) -> torch.Tensor:
    if target_dim > W_full.shape[0]:
        pad = torch.zeros(
            target_dim - W_full.shape[0],
            W_full.shape[1],
            device=W_full.device,
            dtype=W_full.dtype,
        )
        return torch.cat([W_full, pad], dim=0)
    
    W = W_full.float()
    strategy = strategy.lower()

    if strategy == "pca":
        centered = W - W.mean(dim=0, keepdim=True)
        _,S,Vt = torch.linalg.svd(centered, full_matrices=False)
        result = torch.sqrt(S[:target_dim]).unsqueeze(1) * Vt[:target_dim, :]
        return result.to(W_full.dtype)
    
    elif strategy == "svd_sqrt":
        _,S,Vt = torch.linalg.svd(W, full_matrices=False)
        result = torch.sqrt(S[:target_dim]).unsqueeze(1) * Vt[:target_dim, :]
        return result.to(W_full.dtype)

    elif strategy == "svd_full":
        _,S,Vt = torch.linalg.svd(W, full_matrices=False)
        result = S[:target_dim].unsqueeze(1) * Vt[:target_dim, :]
        return result.to(W_full.dtype)

    elif strategy == "svd_noscale":
        _,S,Vt = torch.linalg.svd(W, full_matrices=False)
        result = Vt[:target_dim, :]
        return result.to(W_full.dtype)

    elif strategy == "change_b":
        _,S,Vt = torch.linalg.svd(W, full_matrices=False)
        result = S[:target_dim].pow(power_b).unsqueeze(1) * Vt[:target_dim, :]
        return result.to(W_full.dtype)

    elif strategy == "change_a_b":
        _,S,Vt = torch.linalg.svd(W, full_matrices=False)
        result = scale_a * S[:target_dim].pow(power_b).unsqueeze(1) * Vt[:target_dim, :]
        return result.to(W_full.dtype)

    elif strategy == "mlp_log_singular_values":
        if mlp_scaler is None:
            raise ValueError(
                "strategy='mlp_log_singular_values' requires a trained MLP scaler."
            )
        _, S, Vt = torch.linalg.svd(W, full_matrices=False)
        with torch.no_grad():
            g = mlp_scaler(S[:target_dim].to(next(mlp_scaler.parameters()).device))
            g = g.to(device=W.device, dtype=W.dtype)
        result = g.unsqueeze(1) * Vt[:target_dim, :]
        return result.to(W_full.dtype)

    elif strategy == "random_orth":
        random_matrix = torch.randn(
            W_full.shape[1],
            target_dim,
            device=W_full.device,
            dtype=torch.float32,
        )
        Q, _ = torch.linalg.qr(random_matrix, mode="reduced")
        return Q.T.to(W_full.dtype)

    raise ValueError(
        f"Unknown initialization strategy {strategy!r}. "
        "Expected one of: 'pca', 'svd_sqrt', 'svd_noscale', 'svd_full', "
        "'change_b', 'change_a_b', 'mlp_log_singular_values', 'random_orth'."
    )

def _resize_projection_svd(W_full: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """Resize a 2D projection weight to target_shape with an SVD-based fallback."""
    target_rows, target_cols = target_shape
    W = W_full.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    copy_rows = min(target_rows, W.shape[0])
    copy_cols = min(target_cols, W.shape[1])
    k = min(copy_rows, copy_cols, S.numel())

    result = torch.zeros(
        target_rows,
        target_cols,
        device=W_full.device,
        dtype=torch.float32,
    )
    result[:copy_rows, :copy_cols] = (
        U[:copy_rows, :k] * S[:k].unsqueeze(0)
    ) @ Vt[:k, :copy_cols]
    return result.to(W_full.dtype)



def _expand_gqa_weight(W_kv: torch.Tensor, n_kv_heads: int, n_heads: int,
                        head_dim: int) -> torch.Tensor:
    """
    Expand GQA K/V weight [n_kv_heads * head_dim, d_model] to full
    [n_heads * head_dim, d_model] by repeating KV head groups.
    """
    if n_kv_heads == n_heads:
        return W_kv
    
    repeat_factor = n_heads // n_kv_heads
    W_grouped = W_kv.reshape(n_kv_heads, head_dim, -1)
    W_expanded = W_grouped.repeat_interleave(repeat_factor, dim=0)
    return W_expanded.reshape(n_heads * head_dim, -1)

def _rank_for_layer(rank, layer_idx):
    if isinstance(rank, dict):
        return rank[layer_idx]
    return rank

def warm_init_ska_from_attention(
    teacher_model: nn.Module,
    ska_modules: nn.ModuleDict,
    distill_indices: list,
    rank: int,
    strategy="pca",
    scale_a: float = 1.0,
    power_b: float = 0.5,
    mlp_state_path: Optional[str] = None,
    mlp_hidden_dim: int = 32,
    mlp_residual_scale: float = 0.3,
    verbose: bool = True,
    diagnostics: bool = True,
):
    """Initialize SKA module weights from teacher attention weights."""
    d_model, n_heads, head_dim = _get_model_dims(teacher_model)
    layers = _get_layers(teacher_model)
    
    cfg = teacher_model.config
    n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
    is_gqa = (n_kv_heads != n_heads)
    mlp_checkpoint = None
    if strategy == "mlp_log_singular_values":
        if mlp_state_path is None:
            raise ValueError(
                "strategy='mlp_log_singular_values' requires mlp_state_path."
            )
        mlp_checkpoint = torch.load(mlp_state_path, map_location="cpu")
    
    if verbose:
        print(f"Warm-starting SKA from attention weights:")
        print(f"  d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")
        print(f"  n_kv_heads={n_kv_heads}, GQA={'yes' if is_gqa else 'no'}")
        print(f"  SKA rank={rank}, strategy={strategy}")
        if strategy in {"change_b", "change_a_b"}:
            print(f"  learned scaling: a={scale_a}, b={power_b}")
        if strategy == "mlp_log_singular_values":
            print(
                f"  learned MLP scaling: state={mlp_state_path}, "
                f"hidden_dim={mlp_hidden_dim}, residual_scale={mlp_residual_scale}"
            )
    
    for idx in distill_indices:
        layer = layers[idx]
        attn = _get_attn_module(layer)
        ska = ska_modules[str(idx)]
        
        W_q, W_k, W_v, W_o = _extract_qkvo(attn, d_model, n_heads, n_kv_heads, head_dim)
        
        if W_q is None:
            if verbose:
                print(f"  Layer {idx}: Could not extract attention weights, using random init")
            continue
        
        layer_rank = _rank_for_layer(rank, idx)
        n_ska_heads = ska.H
        ska_key_dim = n_ska_heads * layer_rank    # [n_ska_heads * rank, d_model]
        ska_query_dim = n_ska_heads * layer_rank
        ska_value_dim = n_ska_heads * head_dim  # [n_ska_heads * head_dim, d_model]

        if ska.key_proj.weight.shape != (ska_key_dim, d_model):
            raise ValueError(
                f"Layer {idx}: expected key_proj shape {(ska_key_dim, d_model)}, "
                f"got {tuple(ska.key_proj.weight.shape)}"
            )
        if ska.query_proj.weight.shape != (ska_query_dim, d_model):
            raise ValueError(
                f"Layer {idx}: expected query_proj shape {(ska_query_dim, d_model)}, "
                f"got {tuple(ska.query_proj.weight.shape)}"
            )

        if is_gqa:
            W_k = _expand_gqa_weight(W_k, n_kv_heads, n_ska_heads, head_dim)
            W_v = _expand_gqa_weight(W_v, n_kv_heads, n_ska_heads, head_dim)

        # Match teacher query heads to SKA heads.
        n_q_heads = W_q.shape[0] // head_dim
        W_q_heads = W_q.reshape(n_q_heads, head_dim, d_model)
        if n_q_heads >= n_ska_heads:
            W_q_heads = W_q_heads[:n_ska_heads]
        else:
            if n_ska_heads % n_q_heads != 0:
                raise ValueError(
                    f"Layer {idx}: cannot expand {n_q_heads} Q heads to {n_ska_heads} SKA heads"
                )
            W_q_heads = W_q_heads.repeat_interleave(n_ska_heads // n_q_heads, dim=0)

        W_k_heads = W_k.reshape(n_ska_heads, head_dim, d_model)
        mlp_scaler = None
        if strategy == "mlp_log_singular_values":
            mlp_scaler = _load_mlp_scaler(
                mlp_checkpoint=mlp_checkpoint,
                rank=layer_rank,
                hidden_dim=mlp_hidden_dim,
                residual_scale=mlp_residual_scale,
                device=W_k_heads.device,
            )
        
        with torch.no_grad():
            W_k_init = torch.cat([
                _init_compressed_projection(
                    W_k_heads[h], layer_rank, strategy,
                    scale_a=scale_a, power_b=power_b, mlp_scaler=mlp_scaler
                )
                for h in range(n_ska_heads)
            ], dim=0)
            ska.key_proj.weight.copy_(W_k_init)
            
            W_q_init = torch.cat([
                _init_compressed_projection(
                    W_q_heads[h], layer_rank, strategy,
                    scale_a=scale_a, power_b=power_b, mlp_scaler=mlp_scaler
                )
                for h in range(n_ska_heads)
            ], dim=0)
            ska.query_proj.weight.copy_(W_q_init)
            
            if W_v.shape == ska.value_proj.weight.shape:
                ska.value_proj.weight.copy_(W_v)
            else:
                W_v_init = _init_compressed_projection(W_v, ska_value_dim, "svd_full")
                ska.value_proj.weight.copy_(W_v_init)
            
            if W_o.shape == ska.out_proj.weight.shape:
                ska.out_proj.weight.copy_(W_o)
            else:
                W_o_init = _resize_projection_svd(W_o, tuple(ska.out_proj.weight.shape))
                ska.out_proj.weight.copy_(W_o_init)

        if diagnostics:
            W_k_init = W_k_init.reshape(n_ska_heads, layer_rank, d_model)
            W_k_orig_heads = W_k_heads
            W_q_init = W_q_init.reshape(n_ska_heads, layer_rank, d_model)
            W_q_orig_heads = W_q_heads
            for h in range(n_ska_heads):
                s = torch.linalg.svdvals(W_k_init[h].float())
                eps = 1e-12
                condition = s[0] / s[-1].clamp_min(eps)
                if s.numel() > 1:
                    spectral_gap = s[0] / s[1].clamp_min(eps)
                else:
                    spectral_gap = torch.tensor(float("inf"), device=s.device)
                print(
                    f"  Layer {idx} Head {h}: "
                    f"Conditioning of Key Matrix: {condition.item():.4g}, "
                    f"Spectral Gap: {spectral_gap.item():.4g}"
                )
                k_energy = _captured_energy(W_k_orig_heads[h], W_k_init[h])
                q_energy = _captured_energy(W_q_orig_heads[h], W_q_init[h])
                print(f"  Layer {idx} Head {h}: K energy={k_energy:.1%}, Q energy={q_energy:.1%}")

    
    if verbose:
        print("  Warm initialization complete.")


def _extract_qkvo(attn, d_model, n_heads, n_kv_heads, head_dim):
    """
    Extract Q, K, V, O weight matrices from various attention implementations.
    Returns (W_q, W_k, W_v, W_o) or (None, None, None, None) if not found.
    """
    # Separate q_proj, k_proj, v_proj, o_proj.
    if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj'):
        return (
            attn.q_proj.weight.data.clone(),
            attn.k_proj.weight.data.clone(),
            attn.v_proj.weight.data.clone(),
            attn.o_proj.weight.data.clone(),
        )
    
    # Fused qkv_proj.
    if hasattr(attn, 'qkv_proj'):
        W_qkv = attn.qkv_proj.weight.data.clone()
        q_dim = n_heads * head_dim
        k_dim = n_kv_heads * head_dim
        v_dim = n_kv_heads * head_dim
        W_q = W_qkv[:q_dim]
        W_k = W_qkv[q_dim:q_dim + k_dim]
        W_v = W_qkv[q_dim + k_dim:q_dim + k_dim + v_dim]
        W_o = attn.o_proj.weight.data.clone() if hasattr(attn, 'o_proj') else \
              attn.out_proj.weight.data.clone() if hasattr(attn, 'out_proj') else None
        if W_o is None:
            return None, None, None, None
        return W_q, W_k, W_v, W_o

    # GPT-NeoX/Pythia fused QKV.
    if hasattr(attn, 'query_key_value'):
        W_qkv = attn.query_key_value.weight.data.clone()
        q_dim = n_heads * head_dim
        k_dim = n_kv_heads * head_dim
        v_dim = n_kv_heads * head_dim
        W_q = W_qkv[:q_dim]
        W_k = W_qkv[q_dim:q_dim + k_dim]
        W_v = W_qkv[q_dim + k_dim:q_dim + k_dim + v_dim]
        W_o = attn.dense.weight.data.clone() if hasattr(attn, 'dense') else None
        if W_o is None:
            return None, None, None, None
        return W_q, W_k, W_v, W_o
    
    # GPT-style combined QKV.
    if hasattr(attn, 'c_attn'):
        W_qkv = attn.c_attn.weight.data.clone()
        dim = n_heads * head_dim
        W_q = W_qkv[:, :dim].T  # GPT-2 style: [in, 3*out]
        W_k = W_qkv[:, dim:2*dim].T
        W_v = W_qkv[:, 2*dim:3*dim].T
        W_o = attn.c_proj.weight.data.clone().T if hasattr(attn, 'c_proj') else None
        if W_o is None:
            return None, None, None, None
        return W_q, W_k, W_v, W_o
    
    # Packed QKV.
    if hasattr(attn, 'W_pack'):
        W_qkv = attn.W_pack.weight.data.clone()
        q_dim = n_heads * head_dim
        k_dim = n_kv_heads * head_dim
        v_dim = n_kv_heads * head_dim
        W_q = W_qkv[:q_dim]
        W_k = W_qkv[q_dim:q_dim + k_dim]
        W_v = W_qkv[q_dim + k_dim:]
        W_o = attn.o_proj.weight.data.clone() if hasattr(attn, 'o_proj') else None
        if W_o is None:
            return None, None, None, None
        return W_q, W_k, W_v, W_o

    return None, None, None, None


def _captured_energy(W_full: torch.Tensor, W_trunc: torch.Tensor) -> float:
    """Fraction of Frobenius norm energy captured by truncation."""
    full_norm = W_full.float().norm().item()
    if full_norm < 1e-8:
        return 1.0
    trunc_norm = W_trunc.float().norm().item()
    return min(trunc_norm / full_norm, 1.0)
