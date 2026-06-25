#!/usr/bin/env python3
"""Implicit autograd for the block-causal SKA readout.

The operator computes y = C_v (G^{-1} M)^K G^{-1} q using Cholesky solves
in the forward pass and a custom backward pass that avoids differentiating
through the Cholesky factorization.
"""

import torch
import torch.nn.functional as F


def _sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


def _spectral_norm_scale(M, L, gamma):
    """Estimate the detached spectral scaling for A_w = L^{-1} M L^{-T}."""
    r = L.shape[-1]
    # Power iteration without materializing A_w.
    with torch.no_grad():
        u = torch.randn(*L.shape[:-1], device=L.device, dtype=L.dtype)
        u = F.normalize(u, dim=-1, eps=1e-6)

        for _ in range(6):
            tmp = torch.linalg.solve_triangular(L.transpose(-1, -2),
                                                 u.unsqueeze(-1), upper=True).squeeze(-1)
            tmp = (M.transpose(-1, -2) @ tmp.unsqueeze(-1)).squeeze(-1)
            v = torch.linalg.solve_triangular(L, tmp.unsqueeze(-1), upper=False).squeeze(-1)
            v = F.normalize(v, dim=-1, eps=1e-6)

            tmp = torch.linalg.solve_triangular(L.transpose(-1, -2),
                                                 v.unsqueeze(-1), upper=True).squeeze(-1)
            tmp = (M @ tmp.unsqueeze(-1)).squeeze(-1)
            u = torch.linalg.solve_triangular(L, tmp.unsqueeze(-1), upper=False).squeeze(-1)
            u = F.normalize(u, dim=-1, eps=1e-6)

        tmp = torch.linalg.solve_triangular(L.transpose(-1, -2),
                                             v.unsqueeze(-1), upper=True).squeeze(-1)
        tmp = (M @ tmp.unsqueeze(-1)).squeeze(-1)
        Av = torch.linalg.solve_triangular(L, tmp.unsqueeze(-1), upper=False).squeeze(-1)
        sigma = (u * Av).sum(dim=-1).abs().clamp_min(1.0)

    alpha = gamma / sigma
    return alpha


class BlockCausalSKAFunction(torch.autograd.Function):
    """Custom autograd for y = C_v (G^{-1} M)^K G^{-1} q."""

    @staticmethod
    def forward(ctx, G, M, Cv, q, alpha, power_K):
        """Run the SKA readout for one batch of chunk statistics."""
        # Symmetrize G (cholesky only reads lower tri)
        G = 0.5 * (G + G.transpose(-1, -2))
        L = torch.linalg.cholesky(G)

        # Solve x_0 = G^{-1}q in [N, r, S] layout.
        qt = q.transpose(-1, -2).contiguous()
        x = torch.cholesky_solve(qt, L)  # [N, r, S]

        # Apply the Koopman power filter.
        if alpha.dim() == 1:
            alpha_bc = alpha[:, None, None]  # [N, 1, 1]
        else:
            alpha_bc = alpha

        for _ in range(power_K):
            Mx = M @ x  # [N, r, S]
            x = alpha_bc * torch.cholesky_solve(Mx, L)

        y = (Cv @ x).transpose(-1, -2).contiguous()  # [N, S, P]

        ctx.save_for_backward(L, M, Cv, q, alpha)
        ctx.power_K = power_K
        return y

    @staticmethod
    def backward(ctx, grad_y):
        L, M, Cv, q, alpha = ctx.saved_tensors
        K = ctx.power_K

        if alpha.dim() == 1:
            alpha_bc = alpha[:, None, None]
        else:
            alpha_bc = alpha

        # Recompute Krylov states x_0 ... x_K
        qt = q.transpose(-1, -2).contiguous()  # [N, r, S]
        x = torch.cholesky_solve(qt, L)
        xs = [x]
        for _ in range(K):
            x = alpha_bc * torch.cholesky_solve(M @ x, L)
            xs.append(x)

        xK = xs[-1]  # [N, r, S]

        grad_yt = grad_y.transpose(-1, -2).contiguous()

        grad_Cv = grad_yt @ xK.transpose(-1, -2)  # [N, P, r]
        bar_x = Cv.transpose(-1, -2) @ grad_yt     # [N, r, S]

        grad_M = torch.zeros_like(M)
        grad_G = torch.zeros_like(M)

        # Reverse the Koopman recurrence.
        for i in range(K, 0, -1):
            xi = xs[i]
            xprev = xs[i - 1]

            # lambda_i = G^{-1} bar_x
            lam = torch.cholesky_solve(bar_x, L)  # [N, r, S]

            r = alpha_bc * lam

            grad_M = grad_M + r @ xprev.transpose(-1, -2)

            grad_G = grad_G - lam @ xi.transpose(-1, -2)

            bar_x = M.transpose(-1, -2) @ r

        lam0 = torch.cholesky_solve(bar_x, L)
        grad_q_t = lam0  # [N, r, S]
        grad_G = grad_G - lam0 @ xs[0].transpose(-1, -2)

        grad_G = _sym(grad_G)

        grad_q = grad_q_t.transpose(-1, -2).contiguous()

        return grad_G, grad_M, grad_Cv, grad_q, None, None


def block_causal_ska(G, M, Cv, q, gamma, power_K, ridge_eps=1e-3):
    """Convenience wrapper that adds ridge regularization and spectral scaling."""
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    G_reg = G + ridge_eps * eye

    G_reg = 0.5 * (G_reg + G_reg.transpose(-1, -2))

    try:
        L_test = torch.linalg.cholesky(G_reg)
    except RuntimeError:
        G_reg = G_reg + 1e-4 * eye

    if power_K > 0:
        L_tmp = torch.linalg.cholesky(G_reg)
        gamma_safe = 0.3 + 0.7 * torch.sigmoid(gamma)
        alpha = _spectral_norm_scale(M, L_tmp, gamma_safe)
    else:
        alpha = torch.ones(G.shape[0], device=G.device, dtype=G.dtype)

    return BlockCausalSKAFunction.apply(G_reg, M, Cv, q, alpha, power_K)


def _test_gradcheck():
    """Verify backward correctness with finite differences."""
    torch.manual_seed(42)
    N, r, P, S = 2, 4, 6, 3
    K = 2

    # Build SPD G
    A = torch.randn(N, r, r, dtype=torch.float64)
    G = A @ A.transpose(-1, -2) + 0.1 * torch.eye(r, dtype=torch.float64)
    G = G.requires_grad_(True)

    M = (torch.randn(N, r, r, dtype=torch.float64) * 0.1).requires_grad_(True)
    Cv = (torch.randn(N, P, r, dtype=torch.float64) * 0.1).requires_grad_(True)
    q = (torch.randn(N, S, r, dtype=torch.float64) * 0.1).requires_grad_(True)
    alpha = torch.ones(N, dtype=torch.float64)  # no spectral scaling for test

    def fn(G_, M_, Cv_, q_):
        return BlockCausalSKAFunction.apply(G_, M_, Cv_, q_, alpha, K).sum()

    ok = torch.autograd.gradcheck(fn, (G, M, Cv, q), atol=1e-4, eps=1e-6)
    print(f"gradcheck (K={K}): {ok}")

    def fn0(G_, M_, Cv_, q_):
        return BlockCausalSKAFunction.apply(G_, M_, Cv_, q_, alpha, 0).sum()

    ok0 = torch.autograd.gradcheck(fn0, (G, M, Cv, q), atol=1e-4, eps=1e-6)
    print(f"gradcheck (K=0): {ok0}")

    return ok and ok0


def _test_forward_matches_naive():
    """Verify forward matches naive computation."""
    torch.manual_seed(42)
    N, r, P, S = 2, 4, 6, 3
    K = 2

    A = torch.randn(N, r, r, dtype=torch.float64)
    G = A @ A.transpose(-1, -2) + 0.1 * torch.eye(r, dtype=torch.float64)
    M = torch.randn(N, r, r, dtype=torch.float64) * 0.1
    Cv = torch.randn(N, P, r, dtype=torch.float64) * 0.1
    q = torch.randn(N, S, r, dtype=torch.float64) * 0.1
    alpha = torch.ones(N, dtype=torch.float64)

    # Our function
    y = BlockCausalSKAFunction.apply(G, M, Cv, q, alpha, K)

    # Naive: y = Cv @ (G^{-1} M)^K @ G^{-1} @ q^T
    Ginv = torch.linalg.inv(G)
    GinvM = Ginv @ M
    x = Ginv @ q.transpose(-1, -2)
    for _ in range(K):
        x = GinvM @ x
    y_naive = (Cv @ x).transpose(-1, -2)

    err = (y - y_naive).norm() / (y_naive.norm() + 1e-30)
    print(f"Forward relative error: {err.item():.2e}")
    assert err < 1e-10, f"Forward mismatch: {err}"
    return True


if __name__ == "__main__":
    print("Testing BlockCausalSKAFunction...")
    print()
    print("== Forward correctness ==")
    _test_forward_matches_naive()
    print()
    print("== Gradient correctness ==")
    _test_gradcheck()
    print()
    print("All tests passed!")
