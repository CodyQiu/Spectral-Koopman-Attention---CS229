import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev


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
        default="runs/pythia_retrofit_grid_layer2",
        help="Result root, or comma-separated result roots to aggregate.",
    )
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def parse_roots(raw):
    return [Path(x.strip()) for x in str(raw).split(",") if x.strip()]


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mean_std_ci(values):
    values = list(values)
    if len(values) == 1:
        return values[0], 0.0, 0.0
    m = mean(values)
    sd = stdev(values)
    # 95% two-sided t critical values for the seed counts used here.
    t_crit_by_n = {3: 4.303, 6: 2.571}
    t_crit = t_crit_by_n.get(len(values), 1.96)
    ci = t_crit * sd / math.sqrt(len(values))
    return m, sd, ci


def paired_t_approx(a_by_seed, b_by_seed):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    diffs = [a_by_seed[s] - b_by_seed[s] for s in seeds]
    if len(diffs) < 2:
        return float("nan"), float("nan"), float("nan"), len(diffs)
    diff_mean = mean(diffs)
    diff_sd = stdev(diffs)
    if diff_sd == 0.0:
        t_stat = 0.0 if diff_mean == 0.0 else math.copysign(float("inf"), diff_mean)
        p_value = 1.0 if diff_mean == 0.0 else 0.0
    else:
        t_stat = diff_mean / (diff_sd / math.sqrt(len(diffs)))
        if len(diffs) == 3:
            # Exact two-sided p-value for Student t with df=2:
            # CDF(t) = 1/2 + t / (2 * sqrt(t^2 + 2)).
            t_abs = abs(t_stat)
            p_value = 1.0 - t_abs / math.sqrt(t_abs * t_abs + 2.0)
        else:
            # Lightweight fallback when not using scipy.
            p_value = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return diff_mean, t_stat, p_value, len(diffs)


def read_final_rows(root):
    rows = []
    for root_path in parse_roots(root):
        for path in sorted(root_path.glob("*/metrics.csv")):
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row["stage"] != "5_final_eval":
                        continue
                    rows.append({
                        "rank": int(row["rank"]),
                        "strategy": row["init_strategy"],
                        "seed": int(row["seed"]),
                        "lm_loss": float(row["lm_loss"]),
                        "ppl": float(row["ppl"]),
                    })
    return rows


def write_ci_summary(rows, out_dir):
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["rank"], row["strategy"])].append(row)

    out = out_dir / "stage5_final_eval_ci_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "strategy", "n",
                "lm_loss_mean", "lm_loss_std", "lm_loss_ci95",
                "lm_loss_ci95_low", "lm_loss_ci95_high",
                "ppl_mean", "ppl_std", "ppl_ci95",
                "ppl_ci95_low", "ppl_ci95_high",
            ],
        )
        writer.writeheader()
        for rank in sorted({r["rank"] for r in rows}):
            for strategy in STRATEGY_ORDER:
                vals = by_group.get((rank, strategy), [])
                if not vals:
                    continue
                loss_mean, loss_std, loss_ci = mean_std_ci(v["lm_loss"] for v in vals)
                ppl_mean, ppl_std, ppl_ci = mean_std_ci(v["ppl"] for v in vals)
                writer.writerow({
                    "rank": rank,
                    "strategy": strategy,
                    "n": len(vals),
                    "lm_loss_mean": loss_mean,
                    "lm_loss_std": loss_std,
                    "lm_loss_ci95": loss_ci,
                    "lm_loss_ci95_low": loss_mean - loss_ci,
                    "lm_loss_ci95_high": loss_mean + loss_ci,
                    "ppl_mean": ppl_mean,
                    "ppl_std": ppl_std,
                    "ppl_ci95": ppl_ci,
                    "ppl_ci95_low": ppl_mean - ppl_ci,
                    "ppl_ci95_high": ppl_mean + ppl_ci,
                })
    return out


def write_pairwise_tests(rows, out_dir):
    by_group = defaultdict(dict)
    for row in rows:
        by_group[(row["rank"], row["strategy"])][row["seed"]] = row["lm_loss"]

    tests = []
    for rank in sorted({r["rank"] for r in rows}):
        strategies = [s for s in STRATEGY_ORDER if (rank, s) in by_group]
        for a, b in combinations(strategies, 2):
            diff, t_stat, p_value, n = paired_t_approx(by_group[(rank, a)], by_group[(rank, b)])
            tests.append({
                "rank": rank,
                "strategy_a": a,
                "strategy_b": b,
                "n_pairs": n,
                "mean_diff_a_minus_b": diff,
                "t_stat": t_stat,
                "p_value_normal_approx": p_value,
            })

    m = len(tests)
    for row in tests:
        p = row["p_value_normal_approx"]
        p_bonf = min(1.0, p * m) if not math.isnan(p) else float("nan")
        row["p_bonferroni"] = p_bonf
        row["significant_p_lt_0_05"] = bool(p < 0.05) if not math.isnan(p) else False
        row["bonferroni_significant"] = bool(p_bonf < 0.05) if not math.isnan(p_bonf) else False

    out = out_dir / "stage5_final_eval_pairwise_pvalues.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
        writer.writeheader()
        writer.writerows(tests)
    return out


def write_compact_significance(rows, out_dir):
    by_group = defaultdict(dict)
    means = {}
    for row in rows:
        by_group[(row["rank"], row["strategy"])][row["seed"]] = row["lm_loss"]

    for key, vals in by_group.items():
        means[key] = mean(vals.values())

    compact_rows = []
    for rank in sorted({r["rank"] for r in rows}):
        strategies = [s for s in STRATEGY_ORDER if (rank, s) in by_group]
        ranked = sorted(strategies, key=lambda s: means[(rank, s)])
        winner = ranked[0]
        comparisons = []
        if len(ranked) > 1:
            comparisons.append(("winner_vs_runner_up", winner, ranked[1]))
        if winner != "svd_sqrt" and "svd_sqrt" in strategies:
            comparisons.append(("winner_vs_svd_sqrt", winner, "svd_sqrt"))
        if winner != "svd_sqrt" and len(ranked) > 1 and ranked[1] != "svd_sqrt" and "svd_sqrt" in strategies:
            comparisons.append(("runner_up_vs_svd_sqrt", ranked[1], "svd_sqrt"))

        for comparison, a, b in comparisons:
            diff, t_stat, p_value, n = paired_t_approx(by_group[(rank, a)], by_group[(rank, b)])
            compact_rows.append({
                "rank": rank,
                "comparison": comparison,
                "strategy_a": a,
                "strategy_b": b,
                "mean_a": means[(rank, a)],
                "mean_b": means[(rank, b)],
                "mean_diff_a_minus_b": diff,
                "p_value": p_value,
                "significant_p_lt_0_05": bool(p_value < 0.05),
            })

    csv_path = out_dir / "stage5_compact_significance.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compact_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compact_rows)

    tex_path = out_dir / "stage5_compact_significance.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{c l l c c}\n")
        f.write("\\hline\n")
        f.write("Rank & Comparison & Mean LM Losses & $p$-value & Significant? \\\\" + "\n")
        f.write("\\hline\n")
        for row in compact_rows:
            comp = row["comparison"].replace("_", " ")
            losses = (
                f"{row['strategy_a']} {row['mean_a']:.3f} vs. "
                f"{row['strategy_b']} {row['mean_b']:.3f}"
            )
            p = row["p_value"]
            p_text = f"${p:.2e}$" if p < 0.001 else f"${p:.3f}$"
            sig = "Yes" if row["significant_p_lt_0_05"] else "No"
            f.write(f"{row['rank']} & {comp} & {losses} & {p_text} & {sig} \\\\" + "\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Compact paired significance tests for final Stage 5 LM loss. "
            "Tests are paired by random seed and are not Bonferroni-corrected.}\n"
        )
        f.write("\\label{tab:stage5-compact-significance}\n")
        f.write("\\end{table}\n")

    return csv_path, tex_path


def main():
    args = parse_args()
    roots = parse_roots(args.root)
    if not roots:
        raise SystemExit("No result roots provided")
    out_dir = Path(args.out_dir) if args.out_dir else roots[0] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_final_rows(args.root)
    if not rows:
        print(f"No raw 5_final_eval rows found under {args.root}. "
              "This packaged submission includes generated significance tables under results/next_token_plots; "
              "rerun this script with --root pointing to raw experiment run directories to recompute them.")
        return

    ci_path = write_ci_summary(rows, out_dir)
    pval_path = write_pairwise_tests(rows, out_dir)
    compact_csv, compact_tex = write_compact_significance(rows, out_dir)
    print(f"Read {len(rows)} final-eval rows")
    print(f"Wrote CI summary: {ci_path}")
    print(f"Wrote pairwise tests: {pval_path}")
    print(f"Wrote compact significance CSV: {compact_csv}")
    print(f"Wrote compact significance LaTeX: {compact_tex}")


if __name__ == "__main__":
    main()
