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
LAMBDA_VALS = [0, 1e-5, 1e-4, 1e-3, 1e-2]
FOLDS = 5


LEARNED_STRATEGY = "learned_gtheta"


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
    if strategy_name in ("random_gaussian", "random_orthogonal"):
        return strategy_fn(d_model, 1, rank).astype(np.float32)
    return strategy_fn(w_k.numpy(), rank).astype(np.float32)

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


def differentiable_koopman_loss(x_fit, x_eval, w_comp, lambda_value, seed=0):
    fit_z, fit_next = torch_transition_pairs(x_fit, w_comp)
    eval_z, eval_next = torch_transition_pairs(x_eval, w_comp)

    fit_z, fit_next = sample_pairs(fit_z, fit_next, seed=seed)

    n = fit_z.shape[0]
    rank = fit_z.shape[1]
    eye = torch.eye(rank, dtype=fit_z.dtype, device=fit_z.device)

    g = fit_z.T @ fit_z / n
    m = fit_next.T @ fit_z / n
    g_reg = g + RIDGE * eye

    a_t = torch.linalg.solve(g_reg.T, m.T)
    a = a_t.T

    pred_next = eval_z @ a.T
    mse = torch.mean((pred_next - eval_next) ** 2)
    rel_mse = mse / (torch.mean(eval_next ** 2) + 1e-12)

    eigvals = torch.linalg.eigvalsh(g_reg)
    condition = eigvals[-1] / torch.clamp(eigvals[0], min=1e-12)
    loss = rel_mse + lambda_value * torch.log(condition)

    return loss, rel_mse, condition


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


def make_folds(num_sequences, k=5, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(num_sequences, generator=generator)
    folds = torch.tensor_split(indices, k)

    return folds



def main():
    x_train, x_val, x_test, w_k = get_pythia_hidden_states()
    print(f"x_train: {tuple(x_train.shape)}")
    print(f"x_val:   {tuple(x_val.shape)}")
    print(f"x_test:  {tuple(x_test.shape)}")
    print(f"W_k:     {tuple(w_k.shape)}")

    x_cv = torch.cat([x_train, x_val], dim=0)
    folds = make_folds(x_cv.shape[0], k=FOLDS, seed=0)
    rows = []
    out_dir = Path(__file__).resolve().parent

    _, full_singular_values, full_vt = torch.linalg.svd(w_k, full_matrices=False)

    for rank in RANKS:
        singular_values = full_singular_values[:rank]
        vt = full_vt[:rank, :]

        for lambda_value in LAMBDA_VALS:
            for fold_idx in range(FOLDS):
                val_idx = folds[fold_idx]
                train_idx = torch.cat([
                    folds[j] for j in range(FOLDS)
                    if j != fold_idx
                ])

                x_fold_train = x_cv[train_idx]
                x_fold_val = x_cv[val_idx]

                log_a = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
                b = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
                optimizer = torch.optim.Adam([log_a, b], lr=1e-2)

                for step in range(NUM_STEPS):
                    a = torch.exp(log_a)
                    g = a * singular_values.pow(b)
                    w_comp = torch.diag(g) @ vt

                    loss, rel_mse, condition = differentiable_koopman_loss(
                        x_fold_train,
                        x_fold_val,
                        w_comp,
                        lambda_value=lambda_value,
                        seed=100 + fold_idx,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    a = torch.exp(log_a)
                    g = a * singular_values.pow(b)
                    w_comp = torch.diag(g) @ vt
                    loss, rel_mse, condition = differentiable_koopman_loss(
                        x_fold_train,
                        x_fold_val,
                        w_comp,
                        lambda_value=lambda_value,
                        seed=100 + fold_idx,
                    )

                rows.append({
                    "rank": rank,
                    "lambda": lambda_value,
                    "fold": fold_idx,
                    "a": float(a.item()),
                    "b": float(b.item()),
                    "loss": float(loss.item()),
                    "val_relative_mse": float(rel_mse.item()),
                    "condition": float(condition.item()),
                })

                print(
                    f"rank={rank} lambda={lambda_value:g} fold={fold_idx} "
                    f"val_rel_mse={rel_mse.item():.6f} "
                    f"condition={condition.item():.2f} "
                    f"a={a.item():.4f} b={b.item():.4f}"
                )

    raw_df = pd.DataFrame(rows)
    summary_df = raw_df.groupby(["rank", "lambda"], as_index=False).agg(
        mean_val_relative_mse=("val_relative_mse", "mean"),
        std_val_relative_mse=("val_relative_mse", "std"),
        mean_condition=("condition", "mean"),
        mean_a=("a", "mean"),
        mean_b=("b", "mean"),
    )

    best_df = (
        summary_df
        .sort_values(["rank", "mean_val_relative_mse"])
        .groupby("rank", as_index=False)
        .first()
    )

    raw_df.to_csv(out_dir / "kfold_lambda_raw.csv", index=False)
    summary_df.to_csv(out_dir / "kfold_lambda_summary.csv", index=False)
    best_df.to_csv(out_dir / "kfold_selected_lambdas.csv", index=False)

    print("\nK-fold lambda summary")
    print(summary_df.to_string(index=False))
    print("\nSelected lambdas")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
