from itertools import combinations
from pathlib import Path
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from starter import strategies

MODEL_ID = "EleutherAI/pythia-160m"
LAYER_IDX = 2
RANKS = [8, 16, 32, 64]
SEEDS = [7, 11, 19, 29, 37]
BATCH_TEXTS = 512
MAX_LENGTH = 256
RIDGE = 1e-3
MAX_TRAIN_PAIRS = 16384
CURVE_TRAIN_PAIRS = [64, 128, 256, 512, 1024, 2048, 4096]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 1000
DEFAULT_LAMBDA = 0
CHANGE_A_B_LAMBDA = 0
CHANGE_B_LAMBDA = 0
MLP_LAMBDA = 0
MLP_RESIDUAL_SCALE = 0.3
MLP_HIDDEN_DIM = 32
BASELINE_STRATEGIES = [
    "random_orthogonal",
    "svd_sqrt",
    "svd_full",
    "svd_noscale",
    "pca"
]
CHANGE_A_B_STRATEGY = "change_a_b"
CHANGE_B_STRATEGY = "change_b"
MLP_STRATEGY = "mlp_log_singular_values"
TOP_STRATEGIES = BASELINE_STRATEGIES + [
    CHANGE_A_B_STRATEGY,
    CHANGE_B_STRATEGY,
    MLP_STRATEGY,
]

PLOT_COLORS = {
    "svd_sqrt": "tab:blue",
    "svd_full": "tab:purple",
    "svd_noscale": "tab:green",
    "pca": "tab:brown",
    "random_orthogonal": "tab:orange",
    "random_gaussian": "tab:red",
    "nmf": "tab:pink",
    CHANGE_A_B_STRATEGY: "tab:cyan",
    CHANGE_B_STRATEGY: "tab:olive",
    MLP_STRATEGY: "tab:gray",
}

STRATEGY_FNS = {
    "random_gaussian": strategies.random_gaussian,
    "random_orthogonal": strategies.init_random_orthogonal,
    "svd_sqrt": strategies.init_svd_sqrt,
    "svd_full": strategies.init_svd_full,
    "svd_noscale": strategies.init_svd_noscale,
    "pca": strategies.init_pca,
    "nmf": strategies.init_nmf,
}


def load_texts(split, n_texts):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [ex["text"] for ex in ds if ex["text"].strip()]
    return texts[:n_texts]


def get_pythia_hidden_states():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    train_texts = load_texts("train", BATCH_TEXTS)
    val_texts = load_texts("validation", BATCH_TEXTS // 2)
    test_texts = load_texts("test", BATCH_TEXTS // 2)

    def encode(texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(DEVICE)

    with torch.no_grad():
        train_out = model(**encode(train_texts), output_hidden_states=True)
        val_out = model(**encode(val_texts), output_hidden_states=True)
        test_out = model(**encode(test_texts), output_hidden_states=True)

    x_train = train_out.hidden_states[LAYER_IDX + 1].detach().cpu().float()
    x_val = val_out.hidden_states[LAYER_IDX + 1].detach().cpu().float()
    x_test = test_out.hidden_states[LAYER_IDX + 1].detach().cpu().float()

    layer = model.gpt_neox.layers[LAYER_IDX]
    w_qkv = layer.attention.query_key_value.weight.detach().cpu()
    d_model = model.config.hidden_size
    w_k = w_qkv[d_model:2 * d_model, :].float()

    return x_train, x_val, x_test, w_k


def build_init_matrix(w_k, rank, strategy_name, strategy_fn):
    d_model = w_k.shape[1]

    # Random initializations do not use W_k.
    if strategy_name in ("random_gaussian", "random_orthogonal"):
        return strategy_fn(d_model, 1, rank).astype(np.float32)

    return strategy_fn(w_k.numpy(), rank).astype(np.float32)


class LogSingularValueResidualMLP(torch.nn.Module):
    def __init__(self, hidden_dim=32, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def make_features(self, singular_values):
        rank = singular_values.shape[0]
        device = singular_values.device
        dtype = singular_values.dtype

        log_sigma = torch.log(singular_values + 1e-8)

        # Normalize log singular values for stable MLP training.
        log_sigma_feature = (
            log_sigma - log_sigma.mean()
        ) / (log_sigma.std(unbiased=False) + 1e-8)

        if rank == 1:
            index_feature = torch.zeros(1, device=device, dtype=dtype)
        else:
            index_feature = torch.linspace(0.0, 1.0, rank, device=device, dtype=dtype)

        features = torch.stack(
            [log_sigma_feature, index_feature],
            dim=1,
        )

        return features, log_sigma

    def forward(self, singular_values):
        """
        singular_values: [rank]

        Returns:
            g: [rank], learned positive scaling values.
            log_g: [rank], learned log scaling values.
        """
        features, log_sigma = self.make_features(singular_values)
        residual = self.net(features).squeeze(-1)

        log_g = 0.5 * log_sigma + self.residual_scale * residual

        log_g = torch.clamp(log_g, min=-10.0, max=10.0)

        g = torch.exp(log_g)

        return g, log_g

def transition_pairs(x, w_init):
    z = x.numpy() @ w_init.T  # (batch, seq, rank)
    z_t = z[:, :-1, :].reshape(-1, z.shape[-1])
    z_next = z[:, 1:, :].reshape(-1, z.shape[-1])
    return z_t, z_next


def fit_koopman_operator(z_t, z_next, seed, max_pairs=MAX_TRAIN_PAIRS):
    rng = np.random.default_rng(seed)
    n_pairs = z_t.shape[0]
    sample_size = min(max_pairs, n_pairs)
    idx = rng.choice(n_pairs, size=sample_size, replace=False)

    x = z_t[idx]
    y = z_next[idx]
    rank = x.shape[1]

    g = (x.T @ x) / sample_size
    m = (y.T @ x) / sample_size
    g_reg = g + RIDGE * np.eye(rank)
    a_w = m @ np.linalg.inv(g_reg)
    return a_w, g_reg


def prediction_metrics(z_t, z_next, a_w):
    pred_next = z_t @ a_w.T
    mse = np.mean((pred_next - z_next) ** 2)
    rel_mse = mse / (np.mean(z_next ** 2) + 1e-12)
    return float(mse), float(rel_mse)


def evaluate_koopman(x_train, x_val, w_init, seed):
    train_z, train_next = transition_pairs(x_train, w_init)
    val_z, val_next = transition_pairs(x_val, w_init)
    a_w, g_reg = fit_koopman_operator(train_z, train_next, seed)

    mse, rel_mse = prediction_metrics(val_z, val_next, a_w)

    eigvals = np.linalg.eigvalsh(g_reg)
    condition = eigvals[-1] / max(eigvals[0], 1e-12)

    singular_values = np.linalg.svd(a_w, compute_uv=False)
    spectral_gap = singular_values[0] / (singular_values[1] + 1e-12)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(a_w)))

    return {
        "val_mse": mse,
        "val_relative_mse": rel_mse,
        "condition": float(condition),
        "spectral_gap": float(spectral_gap),
        "spectral_radius": float(spectral_radius),
    }


def validation_curve(x_train, x_val, w_init, seed):
    train_z, train_next = transition_pairs(x_train, w_init)
    val_z, val_next = transition_pairs(x_val, w_init)

    rows = []
    for n_pairs in CURVE_TRAIN_PAIRS:
        a_w, _ = fit_koopman_operator(train_z, train_next, seed, max_pairs=n_pairs)
        val_mse, val_relative_mse = prediction_metrics(val_z, val_next, a_w)
        rows.append({
            "n_train_pairs": n_pairs,
            "val_mse": val_mse,
            "val_relative_mse": val_relative_mse,
        })
    return rows


def torch_transition_pairs(x, w_comp):
    z = x @ w_comp.T
    z_t = z[:, :-1, :].reshape(-1, z.shape[-1])
    z_next = z[:, 1:, :].reshape(-1, z.shape[-1])
    return z_t, z_next


def sample_pairs(z_t, z_next, max_pairs=MAX_TRAIN_PAIRS, seed=0):
    n_pairs = z_t.shape[0]
    sample_size = min(max_pairs, n_pairs)
    generator = torch.Generator(device=z_t.device)
    generator.manual_seed(seed)
    idx = torch.randperm(n_pairs, generator=generator, device=z_t.device)[:sample_size]
    return z_t[idx], z_next[idx]


def differentiable_koopman_loss(x_train, x_val, w_comp, seed=0, lambda_value=DEFAULT_LAMBDA):
    train_z, train_next = torch_transition_pairs(x_train, w_comp)
    val_z, val_next = torch_transition_pairs(x_val, w_comp)
    train_z, train_next = sample_pairs(train_z, train_next, seed=seed)

    n = train_z.shape[0]
    rank = train_z.shape[1]
    eye = torch.eye(rank, dtype=train_z.dtype, device=train_z.device)

    g = train_z.T @ train_z / n
    m = train_next.T @ train_z / n
    g_reg = g + RIDGE * eye

    a_t = torch.linalg.solve(g_reg.T, m.T)
    a = a_t.T

    pred_next = val_z @ a.T
    mse = torch.mean((pred_next - val_next) ** 2)
    rel_mse = mse / (torch.mean(val_next ** 2) + 1e-12)

    eigvals = torch.linalg.eigvalsh(g_reg)
    condition = eigvals[-1] / torch.clamp(eigvals[0], min=1e-12)
    loss = rel_mse + lambda_value * torch.log(condition)

    return loss, rel_mse, condition


def train_change_a_b_strategy(
        x_train,
        x_val,
        w_k,
        rank,
        seed=100,
        lambda_value=CHANGE_A_B_LAMBDA,
        num_steps=NUM_STEPS,
        lr=1e-2,
):
    torch.manual_seed(seed)

    _, singular_values, vt = torch.linalg.svd(w_k.to(x_train.device), full_matrices=False)
    singular_values = singular_values[:rank]
    vt = vt[:rank, :]

    log_a = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    b = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_a, b], lr=lr)

    for step in range(num_steps):
        a = torch.exp(log_a)
        g = a * singular_values.pow(b)
        w_comp = torch.diag(g) @ vt
        loss, rel_mse, condition = differentiable_koopman_loss(
            x_train,
            x_val,
            w_comp,
            seed=seed,
            lambda_value=lambda_value,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(
                f"strategy={CHANGE_A_B_STRATEGY} "
                f"rank={rank} step={step} "
                f"loss={loss.item():.6f} "
                f"rel_mse={rel_mse.item():.6f} "
                f"cond={condition.item():.2f} "
                f"a={a.item():.4f} b={b.item():.4f}"
            )

    with torch.no_grad():
        a = torch.exp(log_a)
        g = a * singular_values.pow(b)
        w_comp = torch.diag(g) @ vt
        loss, rel_mse, condition = differentiable_koopman_loss(
            x_train,
            x_val,
            w_comp,
            seed=seed,
            lambda_value=lambda_value,
        )

    info = {
        "rank": rank,
        "strategy": CHANGE_A_B_STRATEGY,
        "a": float(a.item()),
        "b": float(b.item()),
        "lambda": lambda_value,
        "loss": float(loss.item()),
        "val_relative_mse": float(rel_mse.item()),
        "condition": float(condition.item()),
    }

    return w_comp.detach().cpu().numpy().astype(np.float32), info


def train_change_b_strategy(
        x_train,
        x_val,
        w_k,
        rank,
        seed=100,
        lambda_value=CHANGE_B_LAMBDA,
        num_steps=NUM_STEPS,
        lr=1e-2,
):
    torch.manual_seed(seed)

    _, singular_values, vt = torch.linalg.svd(w_k.to(x_train.device), full_matrices=False)
    singular_values = singular_values[:rank]
    vt = vt[:rank, :]

    b = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    optimizer = torch.optim.Adam([b], lr=lr)

    for step in range(num_steps):
        g = singular_values.pow(b)
        w_comp = torch.diag(g) @ vt
        loss, rel_mse, condition = differentiable_koopman_loss(
            x_train,
            x_val,
            w_comp,
            seed=seed,
            lambda_value=lambda_value,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(
                f"strategy={CHANGE_B_STRATEGY} "
                f"rank={rank} step={step} "
                f"loss={loss.item():.6f} "
                f"rel_mse={rel_mse.item():.6f} "
                f"cond={condition.item():.2f} "
                f"b={b.item():.4f}"
            )

    with torch.no_grad():
        g = singular_values.pow(b)
        w_comp = torch.diag(g) @ vt
        loss, rel_mse, condition = differentiable_koopman_loss(
            x_train,
            x_val,
            w_comp,
            seed=seed,
            lambda_value=lambda_value,
        )

    info = {
        "rank": rank,
        "strategy": CHANGE_B_STRATEGY,
        "b": float(b.item()),
        "lambda": lambda_value,
        "loss": float(loss.item()),
        "val_relative_mse": float(rel_mse.item()),
        "condition": float(condition.item()),
    }

    return w_comp.detach().cpu().numpy().astype(np.float32), info


def train_mlp_strategy(
        x_train,
        x_val,
        w_k,
        rank,
        seed=100,
        hidden_dim=MLP_HIDDEN_DIM,
        residual_scale=MLP_RESIDUAL_SCALE,
        lambda_value=MLP_LAMBDA,
        num_steps=NUM_STEPS,
        lr=1e-3,
):
    """
    We train an MLP to learn log g_i = f_theta(log sigma_i, i / r), comparing it to
    our old learned g_theta and the fixed baselines.
    """

    torch.manual_seed(seed)

    w_k_raw = w_k.to(x_train.device)

    _, singular_values, vt = torch.linalg.svd(
        w_k_raw,
        full_matrices=False,
    )

    singular_values = singular_values[:rank]
    vt = vt[:rank, :]

    model = LogSingularValueResidualMLP(
        hidden_dim=hidden_dim,
        residual_scale=residual_scale,
    ).to(x_train.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    for step in range(num_steps):
        optimizer.zero_grad()

        g, log_g = model(singular_values)
        w_comp = torch.diag(g) @ vt

        loss, rel_mse, condition = differentiable_koopman_loss(
            x_train,
            x_val,
            w_comp,
            seed=seed,
            lambda_value=lambda_value,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                base_g = torch.sqrt(singular_values + 1e-8)
                rel_g_change = torch.norm(g - base_g) / (torch.norm(base_g) + 1e-12)

            print(
                f"strategy={MLP_STRATEGY} "
                f"rank={rank} step={step} "
                f"loss={loss.item():.6f} "
                f"rel_mse={rel_mse.item():.6f} "
                f"cond={condition.item():.2f} "
                f"rel_g_change={rel_g_change.item():.6f}"
            )

    with torch.no_grad():
        g, log_g = model(singular_values)
        w_comp = torch.diag(g) @ vt

        loss, rel_mse, condition = differentiable_koopman_loss(
            x_train,
            x_val,
            w_comp,
            seed=seed,
            lambda_value=lambda_value,
        )

        base_g = torch.sqrt(singular_values + 1e-8)
        rel_g_change = torch.norm(g - base_g) / (torch.norm(base_g) + 1e-12)

    mlp_info = {
        "rank": rank,
        "strategy": MLP_STRATEGY,
        "loss": float(loss.item()),
        "val_relative_mse": float(rel_mse.item()),
        "condition": float(condition.item()),
        "rel_g_change_from_svd_sqrt": float(rel_g_change.item()),
        "hidden_dim": hidden_dim,
        "residual_scale": residual_scale,
        "lambda": lambda_value,
    }

    return w_comp.detach().cpu().numpy().astype(np.float32), mlp_info


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    sem = stats.sem(values)
    margin = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * sem
    return mean, float(mean - margin), float(mean + margin)


def summarize(raw_df):
    summary_rows = []
    metric_cols = [
        "val_relative_mse",
        "val_mse",
        "condition",
        "spectral_gap",
        "spectral_radius",
    ]

    for (rank, strategy), group in raw_df.groupby(["rank", "strategy"]):
        row = {"rank": rank, "strategy": strategy, "n_seeds": len(group)}
        for metric in metric_cols:
            mean, ci_low, ci_high = confidence_interval(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def paired_t_tests(raw_df, metric="val_relative_mse"):
    rows = []
    for rank, rank_df in raw_df.groupby("rank"):
        for left, right in combinations(sorted(rank_df["strategy"].unique()), 2):
            l_vals = (
                rank_df[rank_df["strategy"] == left]
                .sort_values("seed")[metric]
                .to_numpy()
            )
            r_vals = (
                rank_df[rank_df["strategy"] == right]
                .sort_values("seed")[metric]
                .to_numpy()
            )
            t_stat, p_value = stats.ttest_rel(l_vals, r_vals)
            rows.append({
                "rank": rank,
                "metric": metric,
                "strategy_a": left,
                "strategy_b": right,
                "mean_a": float(l_vals.mean()),
                "mean_b": float(r_vals.mean()),
                "mean_diff_a_minus_b": float((l_vals - r_vals).mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant_p_lt_0_05": bool(p_value < 0.05),
            })
    return pd.DataFrame(rows)


def plot_metric(summary_df, pvals_df, out_dir, metric):
    for rank, rank_df in summary_df.groupby("rank"):
        rank_df = rank_df.set_index("strategy").loc[TOP_STRATEGIES].reset_index()
        means = rank_df[f"{metric}_mean"].to_numpy()
        err_low = means - rank_df[f"{metric}_ci_low"].to_numpy()
        err_high = rank_df[f"{metric}_ci_high"].to_numpy() - means
        x = np.arange(len(rank_df))

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [PLOT_COLORS[s] for s in rank_df["strategy"]]
        ax.bar(x, means, yerr=[err_low, err_high], capsize=5, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(rank_df["strategy"], rotation=20, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Pythia Koopman validation, rank={rank}")
        ax.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        plt.savefig(out_dir / f"pythia_koopman_{metric}_rank{rank}.png", dpi=300)
        plt.close(fig)


def plot_validation_curves(curves_df, out_dir):
    for rank, rank_df in curves_df.groupby("rank"):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for strategy_name in TOP_STRATEGIES:
            group = rank_df[rank_df["strategy"] == strategy_name]
            curve = (
                group.groupby("n_train_pairs")["val_relative_mse"]
                .agg(["mean", "sem"])
                .reset_index()
            )
            ax.plot(
                curve["n_train_pairs"],
                curve["mean"],
                marker="o",
                linewidth=2,
                label=strategy_name,
                color=PLOT_COLORS[strategy_name],
            )
            ax.fill_between(
                curve["n_train_pairs"],
                curve["mean"] - curve["sem"].fillna(0),
                curve["mean"] + curve["sem"].fillna(0),
                color=PLOT_COLORS[strategy_name],
                alpha=0.15,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of transition pairs used to fit Koopman operator")
        ax.set_ylabel("Validation relative MSE")
        ax.set_title(f"Pythia Koopman validation convergence, rank={rank}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"all_strategies_validation_convergence_rank{rank}.png", dpi=300)
        plt.close(fig)


def main():
    x_train, x_val, x_test, w_k = get_pythia_hidden_states()
    print(f"x_train: {tuple(x_train.shape)}")
    print(f"x_val:   {tuple(x_val.shape)}")
    print(f"x_test:  {tuple(x_test.shape)}")
    print(f"W_k:     {tuple(w_k.shape)}")

    raw_rows = []
    curve_rows = []
    curves = {}
    learned_rows = []
    learned_by_strategy_rank = {}
    mlp_by_rank = {}

    for rank in RANKS:
        w_change_a_b_np, change_a_b_info = train_change_a_b_strategy(
            x_train=x_train,
            x_val=x_val,
            w_k=w_k,
            rank=rank,
            seed=100,
            lambda_value=CHANGE_A_B_LAMBDA,
            num_steps=NUM_STEPS,
            lr=1e-2,
        )
        learned_by_strategy_rank[(CHANGE_A_B_STRATEGY, rank)] = w_change_a_b_np
        learned_rows.append(change_a_b_info)

        w_change_b_np, change_b_info = train_change_b_strategy(
            x_train=x_train,
            x_val=x_val,
            w_k=w_k,
            rank=rank,
            seed=100,
            lambda_value=CHANGE_B_LAMBDA,
            num_steps=NUM_STEPS,
            lr=1e-2,
        )
        learned_by_strategy_rank[(CHANGE_B_STRATEGY, rank)] = w_change_b_np
        learned_rows.append(change_b_info)


    for rank in RANKS:
        w_mlp_np, mlp_info = train_mlp_strategy(
            x_train=x_train,
            x_val=x_val,
            w_k=w_k,
            rank=rank,
            seed=100,
            hidden_dim=MLP_HIDDEN_DIM,
            residual_scale=MLP_RESIDUAL_SCALE,
            lambda_value=MLP_LAMBDA,
            num_steps=NUM_STEPS,
            lr=1e-3,
        )

        mlp_by_rank[rank] = w_mlp_np
        learned_rows.append(mlp_info)

    for rank in RANKS:
        curves[str(rank)] = {}
        for strategy_name in TOP_STRATEGIES:
            print(f"rank={rank} strategy={strategy_name}")
            if strategy_name in (CHANGE_A_B_STRATEGY, CHANGE_B_STRATEGY):
                w_init = learned_by_strategy_rank[(strategy_name, rank)]
            elif strategy_name == MLP_STRATEGY:
                w_init = mlp_by_rank[rank]
            else:
                strategy_fn = STRATEGY_FNS[strategy_name]
                w_init = build_init_matrix(w_k, rank, strategy_name, strategy_fn)

            curves[str(rank)][strategy_name] = {}
            for seed in SEEDS:
                metrics = evaluate_koopman(x_train, x_test, w_init, seed)
                raw_rows.append({
                    "rank": rank,
                    "strategy": strategy_name,
                    "seed": seed,
                    **metrics,
                })
                curves[str(rank)][strategy_name][str(seed)] = metrics

                for curve_point in validation_curve(x_train, x_test, w_init, seed):
                    curve_rows.append({
                        "rank": rank,
                        "strategy": strategy_name,
                        "seed": seed,
                        **curve_point,
                    })

    out_dir = Path(__file__).resolve().parent
    learned_df = pd.DataFrame(learned_rows)
    raw_df = pd.DataFrame(raw_rows)
    curves_df = pd.DataFrame(curve_rows)
    summary_df = summarize(raw_df)
    pvals_df = paired_t_tests(raw_df, metric="val_relative_mse")

    learned_df.to_csv(out_dir / "all_learned_koopman_summary.csv", index=False)
    raw_df.to_csv(out_dir / "all_strategies_koopman_raw.csv", index=False)
    curves_df.to_csv(out_dir / "all_strategies_koopman_validation_curves.csv", index=False)
    summary_df.to_csv(out_dir / "all_strategies_koopman_comparison_summary.csv", index=False)
    pvals_df.to_csv(out_dir / "all_strategies_koopman_pvalues.csv", index=False)
    with open(out_dir / "all_strategies_koopman_seed_metrics.json", "w") as f:
        json.dump(curves, f, indent=2)

    plot_metric(summary_df, pvals_df, out_dir, "val_relative_mse")
    plot_metric(summary_df, pvals_df, out_dir, "condition")
    plot_metric(summary_df, pvals_df, out_dir, "spectral_gap")
    plot_validation_curves(curves_df, out_dir)

    print("\nLearned parameters")
    print(learned_df.to_string(index=False))
    print("\nSummary")
    print(summary_df.to_string(index=False))
    print("\nPaired t-tests on val_relative_mse")
    print(pvals_df.to_string(index=False))


if __name__ == "__main__":
    main()
