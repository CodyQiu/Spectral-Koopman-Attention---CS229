import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_LABELS = {
    "svd_sqrt": "SVD-sqrt",
    "svd_full": "SVD-full",
    "svd_noscale": "SVD-noscale",
    "random_orth": "Random orth",
    "change_b": "learned b",
    "change_a_b": "learned a,b",
    "mlp_log_singular_values": "MLP",
}


STRATEGY_ORDER = [
    "svd_sqrt",
    "svd_full",
    "svd_noscale",
    "random_orth",
    "change_b",
    "change_a_b",
    "mlp_log_singular_values",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="runs/pythia_retrofit_grid",
        help="Result root, or comma-separated result roots to aggregate.",
    )
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def parse_roots(raw):
    return [Path(x.strip()) for x in str(raw).split(",") if x.strip()]


def read_rows(root):
    rows = []
    for root_path in parse_roots(root):
        for path in sorted(root_path.glob("*/metrics.csv")):
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["source"] = str(path)
                    row["init_strategy"] = row["init_strategy"]
                    row["rank"] = int(row["rank"])
                    row["seed"] = int(row["seed"])
                    row["step"] = int(row["step"])
                    row["total_steps"] = int(row["total_steps"])
                    row["lm_loss"] = float(row["lm_loss"])
                    row["ppl"] = float(row["ppl"])
                    rows.append(row)
    return rows


def read_rows_old(root):
    rows = []
    for path in sorted(Path(root).glob("*/metrics.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["source"] = str(path)
                row["init_strategy"] = row["init_strategy"]
                row["rank"] = int(row["rank"])
                row["seed"] = int(row["seed"])
                row["step"] = int(row["step"])
                row["total_steps"] = int(row["total_steps"])
                row["lm_loss"] = float(row["lm_loss"])
                row["ppl"] = float(row["ppl"])
                rows.append(row)
    return rows


def mean_std(values):
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def save_summary(rows, out_dir):
    eval_rows = [r for r in rows if r["stage"] in {"base_eval", "5_init_eval", "5_final_eval"}]
    grouped = defaultdict(list)
    for r in eval_rows:
        grouped[(r["rank"], r["init_strategy"], r["stage"])].append(r)

    summary_path = out_dir / "stage5_eval_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "strategy", "stage", "n",
                "lm_loss_mean", "lm_loss_std",
                "ppl_mean", "ppl_std",
            ],
        )
        writer.writeheader()
        for key in sorted(grouped, key=lambda x: (x[0], STRATEGY_ORDER.index(x[1]), x[2])):
            rank, strategy, stage = key
            vals = grouped[key]
            loss_mean, loss_std = mean_std(r["lm_loss"] for r in vals)
            ppl_mean, ppl_std = mean_std(r["ppl"] for r in vals)
            writer.writerow({
                "rank": rank,
                "strategy": strategy,
                "stage": stage,
                "n": len(vals),
                "lm_loss_mean": loss_mean,
                "lm_loss_std": loss_std,
                "ppl_mean": ppl_mean,
                "ppl_std": ppl_std,
            })
    return summary_path


def save_ranked_tables(rows, out_dir):
    final_rows = [r for r in rows if r["stage"] == "5_final_eval"]
    grouped = defaultdict(list)
    for r in final_rows:
        grouped[(r["rank"], r["init_strategy"])].append(r)

    rank_rows = []
    for rank in sorted({r["rank"] for r in final_rows}):
        strategy_rows = []
        for strategy in STRATEGY_ORDER:
            vals = grouped.get((rank, strategy), [])
            if not vals:
                continue
            loss_mean, loss_std = mean_std(r["lm_loss"] for r in vals)
            ppl_mean, ppl_std = mean_std(r["ppl"] for r in vals)
            strategy_rows.append({
                "rank": rank,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "n": len(vals),
                "lm_loss_mean": loss_mean,
                "lm_loss_std": loss_std,
                "ppl_mean": ppl_mean,
                "ppl_std": ppl_std,
            })

        strategy_rows.sort(key=lambda r: r["lm_loss_mean"])
        for i, row in enumerate(strategy_rows, start=1):
            row["ranking"] = i
            row["winner"] = "yes" if i == 1 else "no"
            rank_rows.append(row)

    csv_path = out_dir / "stage5_ranked_final_eval.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "ranking", "strategy", "strategy_label", "winner", "n",
                "lm_loss_mean", "lm_loss_std", "ppl_mean", "ppl_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rank_rows)

    latex_path = out_dir / "stage5_ranked_final_eval.tex"
    with latex_path.open("w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{c l c c c}\n")
        f.write("\\hline\n")
        f.write("Rank & Strategy & Final LM Loss & Final PPL & Ranking \\\\\\n")
        f.write("\\hline\n")
        last_rank = None
        for row in rank_rows:
            if last_rank is not None and row["rank"] != last_rank:
                f.write("\\hline\n")
            strategy = row["strategy_label"]
            loss = f"${row['lm_loss_mean']:.3f}\\,\\pm\\,{row['lm_loss_std']:.3f}$"
            ppl = f"${row['ppl_mean']:.1f}\\,\\pm\\,{row['ppl_std']:.1f}$"
            ranking = str(row["ranking"])
            if row["ranking"] == 1:
                strategy = f"\\textbf{{{strategy}}}"
                loss = f"\\textbf{{{loss}}}"
                ppl = f"\\textbf{{{ppl}}}"
                ranking = "\\textbf{1}"
            f.write(
                f"{row['rank']} & {strategy} & {loss} & {ppl} & {ranking} \\\\\\n"
            )
            last_rank = row["rank"]
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Final Stage 5 next-token validation performance after SKA retrofit training. "
            "Values are mean $\\pm$ standard deviation over random seeds. "
            "Lower LM loss and perplexity are better.}\n"
        )
        f.write("\\label{tab:stage5-ranked-final-eval}\n")
        f.write("\\end{table}\n")

    paired_latex_path = out_dir / "stage5_ranked_final_eval_paired_ranks.tex"
    by_rank = defaultdict(list)
    for row in rank_rows:
        by_rank[row["rank"]].append(row)

    def fmt_label(row):
        return row["strategy_label"].replace("learned a,b", "learned $a,b$").replace(
            "learned b", "learned $b$"
        )

    def fmt_loss(row):
        return f"${row['lm_loss_mean']:.3f} \\pm {row['lm_loss_std']:.3f}$"

    def fmt_ppl(row):
        return f"${row['ppl_mean']:.1f} \\pm {row['ppl_std']:.1f}$"

    def write_rank_pair(f, left_rank, right_rank):
        left = by_rank.get(left_rank, [])
        right = by_rank.get(right_rank, [])
        n_rows = max(len(left), len(right))
        f.write("\\begin{tabular}{lcc|lcc}\n")
        f.write(f"\\multicolumn{{3}}{{c|}}{{Rank {left_rank}}} &\n")
        f.write(f"\\multicolumn{{3}}{{c}}{{Rank {right_rank}}} \\\\\n")
        f.write("\\hline\n")
        f.write("Strategy & Loss & PPL & Strategy & Loss & PPL \\\\\n")
        f.write("\\hline\n")
        for i in range(n_rows):
            if i < len(left):
                l = left[i]
                left_cells = [fmt_label(l), fmt_loss(l), fmt_ppl(l)]
            else:
                left_cells = ["", "", ""]
            if i < len(right):
                r = right[i]
                right_cells = [fmt_label(r), fmt_loss(r), fmt_ppl(r)]
            else:
                right_cells = ["", "", ""]
            f.write(" & ".join(left_cells + right_cells) + " \\\\\n")
        f.write("\\end{tabular}\n")

    with paired_latex_path.open("w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        write_rank_pair(f, 8, 16)
        f.write("\n\\vspace{0.7cm}\n\n")
        write_rank_pair(f, 32, 64)
        f.write(
            "\n\\caption{Final next-token performance after 300 training steps. "
            "Values are mean $\\pm$ standard deviation over random seeds. "
            "Lower loss and perplexity are better.}\n"
        )
        f.write("\\label{tab:stage5-ranked-final-eval-paired-ranks}\n")
        f.write("\\end{table}\n")

    return csv_path, latex_path, paired_latex_path


def plot_training_curves(rows, out_dir, metric):
    train_rows = [r for r in rows if r["stage"] == "5"]
    ranks = sorted({r["rank"] for r in train_rows})
    if not ranks:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()

    for ax, rank in zip(axes, ranks):
        for strategy in STRATEGY_ORDER:
            subset = [
                r for r in train_rows
                if r["rank"] == rank and r["init_strategy"] == strategy
            ]
            if not subset:
                continue
            by_step = defaultdict(list)
            for r in subset:
                by_step[r["step"]].append(r[metric])

            steps = sorted(by_step)
            means = [mean(by_step[s]) for s in steps]
            stds = [stdev(by_step[s]) if len(by_step[s]) > 1 else 0.0 for s in steps]

            label = STRATEGY_LABELS.get(strategy, strategy)
            ax.plot(steps, means, label=label, linewidth=1.8)
            lower = [m - sd for m, sd in zip(means, stds)]
            upper = [m + sd for m, sd in zip(means, stds)]
            ax.fill_between(steps, lower, upper, alpha=0.12)

        ax.set_title(f"rank {rank}")
        ax.set_xlabel("training step")
        ax.set_ylabel("perplexity" if metric == "ppl" else "LM loss")
        ax.grid(alpha=0.25)
        if metric == "ppl":
            ax.set_yscale("log")

    for ax in axes[len(ranks):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(
        "Stage 5 training curves averaged over seeds"
        + (" (log-scale perplexity)" if metric == "ppl" else "")
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    path = out_dir / f"stage5_training_{metric}_curves_by_rank.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_final_eval(rows, out_dir, metric):
    final_rows = [r for r in rows if r["stage"] == "5_final_eval"]
    ranks = sorted({r["rank"] for r in final_rows})
    strategies = [s for s in STRATEGY_ORDER if any(r["init_strategy"] == s for r in final_rows)]

    width = 0.12
    x = list(range(len(ranks)))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, strategy in enumerate(strategies):
        means, errs = [], []
        for rank in ranks:
            vals = [
                r[metric] for r in final_rows
                if r["rank"] == rank and r["init_strategy"] == strategy
            ]
            m, sd = mean_std(vals)
            means.append(m)
            errs.append(sd)
        offsets = [v + (i - (len(strategies) - 1) / 2) * width for v in x]
        ax.bar(
            offsets,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            label=STRATEGY_LABELS.get(strategy, strategy),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("rank")
    ax.set_ylabel("perplexity" if metric == "ppl" else "LM loss")
    ax.set_title("Stage 5 final evaluation averaged over seeds")
    ax.grid(axis="y", alpha=0.25)
    if metric == "ppl":
        ax.set_yscale("log")
    ax.legend(ncol=3)
    fig.tight_layout()

    path = out_dir / f"stage5_final_eval_{metric}_by_rank.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_gap_to_base(rows, out_dir):
    by_run = defaultdict(dict)
    for r in rows:
        if r["stage"] in {"base_eval", "5_final_eval"}:
            key = (r["rank"], r["init_strategy"], r["seed"])
            by_run[key][r["stage"]] = r["lm_loss"]

    gap_rows = []
    for (rank, strategy, seed), vals in by_run.items():
        if "base_eval" in vals and "5_final_eval" in vals:
            gap_rows.append({
                "rank": rank,
                "strategy": strategy,
                "seed": seed,
                "gap": vals["5_final_eval"] - vals["base_eval"],
            })

    ranks = sorted({r["rank"] for r in gap_rows})
    strategies = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in gap_rows)]
    width = 0.12
    x = list(range(len(ranks)))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, strategy in enumerate(strategies):
        means, errs = [], []
        for rank in ranks:
            vals = [r["gap"] for r in gap_rows if r["rank"] == rank and r["strategy"] == strategy]
            m, sd = mean_std(vals)
            means.append(m)
            errs.append(sd)
        offsets = [v + (i - (len(strategies) - 1) / 2) * width for v in x]
        ax.bar(
            offsets,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            label=STRATEGY_LABELS.get(strategy, strategy),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("rank")
    ax.set_ylabel("final LM loss - base LM loss")
    ax.set_title("Remaining loss gap to original attention model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()

    path = out_dir / "stage5_loss_gap_to_base_by_rank.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _annotated_heatmap(matrix, row_labels, col_labels, title, cbar_label, path,
                       fmt="{:.2f}", cmap="viridis_r"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("rank")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    finite_values = [v for row in matrix for v in row if not math.isnan(v)]
    threshold = (min(finite_values) + max(finite_values)) / 2 if finite_values else 0.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                text = "NA"
                color = "black"
            else:
                text = fmt.format(value)
                color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_heatmaps(rows, out_dir):
    ranks = sorted({r["rank"] for r in rows})
    strategies = [s for s in STRATEGY_ORDER if any(r["init_strategy"] == s for r in rows)]
    strategy_labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    rank_labels = [str(r) for r in ranks]

    eval_by_key = defaultdict(list)
    for r in rows:
        if r["stage"] in {"base_eval", "5_init_eval", "5_final_eval"}:
            eval_by_key[(r["rank"], r["init_strategy"], r["stage"])].append(r["lm_loss"])

    final_matrix = []
    gap_matrix = []
    improvement_matrix = []

    for strategy in strategies:
        final_row = []
        gap_row = []
        improvement_row = []
        for rank in ranks:
            final_vals = eval_by_key.get((rank, strategy, "5_final_eval"), [])
            init_vals = eval_by_key.get((rank, strategy, "5_init_eval"), [])
            base_vals = eval_by_key.get((rank, strategy, "base_eval"), [])

            final_mean = mean(final_vals) if final_vals else float("nan")
            init_mean = mean(init_vals) if init_vals else float("nan")
            base_mean = mean(base_vals) if base_vals else float("nan")

            final_row.append(final_mean)
            gap_row.append(final_mean - base_mean)
            improvement_row.append(init_mean - final_mean)

        final_matrix.append(final_row)
        gap_matrix.append(gap_row)
        improvement_matrix.append(improvement_row)

    return [
        _annotated_heatmap(
            final_matrix,
            strategy_labels,
            rank_labels,
            "Final validation LM loss after Stage 5",
            "LM loss (lower is better)",
            out_dir / "stage5_final_lm_loss_heatmap.png",
            cmap="viridis_r",
        ),
        _annotated_heatmap(
            gap_matrix,
            strategy_labels,
            rank_labels,
            "Loss gap to original attention model",
            "final loss - base loss (lower is better)",
            out_dir / "stage5_loss_gap_heatmap.png",
            cmap="viridis_r",
        ),
        _annotated_heatmap(
            improvement_matrix,
            strategy_labels,
            rank_labels,
            "Training improvement from initialization",
            "init loss - final loss (higher is better)",
            out_dir / "stage5_training_improvement_heatmap.png",
            cmap="viridis",
        ),
    ]


def main():
    args = parse_args()
    roots = parse_roots(args.root)
    if not roots:
        raise SystemExit("No result roots provided")
    out_dir = Path(args.out_dir) if args.out_dir else roots[0] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.root)
    if not rows:
        print(f"No raw metrics.csv files found under {args.root}. "
              "This packaged submission includes generated result tables and plots under results/next_token_plots; "
              "rerun this script with --root pointing to raw experiment run directories to regenerate them.")
        return

    summary_path = save_summary(rows, out_dir)
    ranked_csv_path, ranked_latex_path, paired_latex_path = save_ranked_tables(rows, out_dir)
    paths = [
        plot_training_curves(rows, out_dir, "lm_loss"),
        plot_training_curves(rows, out_dir, "ppl"),
        plot_final_eval(rows, out_dir, "lm_loss"),
        plot_final_eval(rows, out_dir, "ppl"),
        plot_gap_to_base(rows, out_dir),
    ]
    paths.extend(plot_heatmaps(rows, out_dir))

    print(f"Read {len(rows)} metric rows from {args.root}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote ranked CSV: {ranked_csv_path}")
    print(f"Wrote ranked LaTeX: {ranked_latex_path}")
    print(f"Wrote paired-rank LaTeX: {paired_latex_path}")
    for path in paths:
        if path is not None:
            print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
