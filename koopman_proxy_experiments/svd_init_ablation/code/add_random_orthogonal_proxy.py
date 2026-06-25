from pathlib import Path
import pandas as pd
import all_learned_koopman as proxy


STRATEGY = "random_orthogonal"
HERE = Path(__file__).resolve().parent


def main():
    raw_path = HERE / "all_strategies_koopman_raw.csv"
    curves_path = HERE / "all_strategies_koopman_validation_curves.csv"

    raw_df = pd.read_csv(raw_path)
    curves_df = pd.read_csv(curves_path)

    raw_df = raw_df[raw_df["strategy"] != STRATEGY].copy()
    curves_df = curves_df[curves_df["strategy"] != STRATEGY].copy()

    x_train, _, x_test, w_k = proxy.get_pythia_hidden_states()
    strategy_fn = proxy.STRATEGY_FNS[STRATEGY]

    raw_rows = []
    curve_rows = []
    for rank in proxy.RANKS:
        print(f"rank={rank} strategy={STRATEGY}")
        w_init = proxy.build_init_matrix(w_k, rank, STRATEGY, strategy_fn)

        for seed in proxy.SEEDS:
            metrics = proxy.evaluate_koopman(x_train, x_test, w_init, seed)
            raw_rows.append({
                "rank": rank,
                "strategy": STRATEGY,
                "seed": seed,
                **metrics,
            })

            for curve_point in proxy.validation_curve(x_train, x_test, w_init, seed):
                curve_rows.append({
                    "rank": rank,
                    "strategy": STRATEGY,
                    "seed": seed,
                    **curve_point,
                })

    raw_df = pd.concat([raw_df, pd.DataFrame(raw_rows)], ignore_index=True)
    curves_df = pd.concat([curves_df, pd.DataFrame(curve_rows)], ignore_index=True)
    summary_df = proxy.summarize(raw_df)
    pvals_df = proxy.paired_t_tests(raw_df, metric="val_relative_mse")

    raw_df.to_csv(raw_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    summary_df.to_csv(HERE / "all_strategies_koopman_comparison_summary.csv", index=False)
    pvals_df.to_csv(HERE / "all_strategies_koopman_pvalues.csv", index=False)

    proxy.plot_metric(summary_df, pvals_df, HERE, "val_relative_mse")
    proxy.plot_metric(summary_df, pvals_df, HERE, "condition")
    proxy.plot_metric(summary_df, pvals_df, HERE, "spectral_gap")
    proxy.plot_validation_curves(curves_df, HERE)

    print("\nRandom orthogonal rows")
    print(pd.DataFrame(raw_rows).to_string(index=False))
    print("\nUpdated summary")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
