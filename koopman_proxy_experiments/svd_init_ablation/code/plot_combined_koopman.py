import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parents[2] / "results"
RANKS = [8, 16, 32, 64]
STRATEGY_ORDER = [
    "random_orthogonal",
    "svd_sqrt",
    "svd_full",
    "svd_noscale",
    "change_b",
    "change_a_b",
    "mlp_log_singular_values",
]
LABELS = {
    "random_orthogonal": "Random orth",
    "svd_sqrt": "SVD-sqrt",
    "svd_full": "SVD-full",
    "svd_noscale": "SVD-noscale",
    "change_b": "learned b",
    "change_a_b": "learned a,b",
    "mlp_log_singular_values": "MLP",
}
COLORS = {
    "random_orthogonal": "#FF9DA6",
    "svd_sqrt": "#4C78A8",
    "svd_full": "#F58518",
    "svd_noscale": "#54A24B",
    "change_b": "#E45756",
    "change_a_b": "#72B7B2",
    "mlp_log_singular_values": "#B279A2",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_std(values):
    values = list(values)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def plot_validation_curves():
    rows = read_csv(RESULTS_DIR / "all_strategies_koopman_validation_curves.csv")
    for row in rows:
        row["rank"] = int(row["rank"])
        row["seed"] = int(row["seed"])
        row["n_train_pairs"] = int(row["n_train_pairs"])
        row["val_relative_mse"] = float(row["val_relative_mse"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, rank in zip(axes, RANKS):
        rank_rows = [r for r in rows if r["rank"] == rank]
        for strategy in STRATEGY_ORDER:
            subset = [r for r in rank_rows if r["strategy"] == strategy]
            if not subset:
                continue
            by_pairs = defaultdict(list)
            for row in subset:
                by_pairs[row["n_train_pairs"]].append(row["val_relative_mse"])
            xs = sorted(by_pairs)
            ys = [mean(by_pairs[x]) for x in xs]
            es = [stdev(by_pairs[x]) if len(by_pairs[x]) > 1 else 0.0 for x in xs]
            ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.5,
                    label=LABELS.get(strategy, strategy))
            ax.fill_between(
                xs,
                [y - e for y, e in zip(ys, es)],
                [y + e for y, e in zip(ys, es)],
                alpha=0.12,
            )

        ax.set_title(f"rank {rank}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("transition pairs used to fit $A$")
        ax.set_ylabel("validation relative MSE")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Koopman proxy convergence by rank")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))

    out = RESULTS_DIR / "combined_validation_convergence_all_ranks.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_condition_numbers():
    rows = read_csv(RESULTS_DIR / "all_strategies_koopman_raw.csv")
    for row in rows:
        row["rank"] = int(row["rank"])
        row["condition"] = float(row["condition"])

    all_means = []
    for rank in RANKS:
        rank_rows = [r for r in rows if r["rank"] == rank]
        for strategy in STRATEGY_ORDER:
            vals = [r["condition"] for r in rank_rows if r["strategy"] == strategy]
            if vals:
                all_means.append(mean(vals))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    axes = axes.flatten()

    for ax, rank in zip(axes, RANKS):
        rank_rows = [r for r in rows if r["rank"] == rank]
        strategies = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in rank_rows)]
        means, errs, labels = [], [], []
        for strategy in strategies:
            vals = [r["condition"] for r in rank_rows if r["strategy"] == strategy]
            m, sd = mean_std(vals)
            means.append(m)
            errs.append(sd)
            labels.append(LABELS.get(strategy, strategy))

        colors = [COLORS.get(strategy, "#777777") for strategy in strategies]
        ax.bar(range(len(labels)), means, yerr=errs, capsize=3, color=colors)
        ax.set_title(f"rank {rank}")
        ax.set_yscale("log")
        if all_means:
            ax.set_ylim(min(all_means) * 0.5, max(all_means) * 2.0)
        ax.set_ylabel("condition number (log scale)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Koopman Gram matrix conditioning by rank")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = RESULTS_DIR / "combined_condition_numbers_all_ranks.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    plot_validation_curves()
    plot_condition_numbers()


if __name__ == "__main__":
    main()
