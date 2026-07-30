# CS229 SKA Initialization Project

This submission contains code for our Koopman proxy experiment, our next-token prediction experiment, and the result files used in the report.

Note: The `next_token_experiments/ska_distill/` code is adapted from a shared SKA-Distill research codebase used for CS197/SKA integration work, which contains mostly architecture code. My additions for this project focus on initialization strategies, learned spectral scaling rules, experiment runners, statistical analysis, and result plotting.

## Relationship to CS197 Work

This CS229 project builds on earlier CS197/SKA work but asks a different question. The CS197 work focused on fixed low-rank initialization strategies, including PCA, SVD-sqrt, SVD-noscale, SVD-full, and random orthogonal initialization. Those experiments emphasized reconstruction error and spectral diagnostics such as condition number and spectral gap.

In contrast, this CS229 project introduces data-driven initialization methods, including learned \(b\), learned \(a,b\), and an MLP-based singular-value scaling rule. We also evaluate the strategies using predictive objectives: a Koopman proxy objective based on next-state relative MSE, and a downstream next-token prediction experiment using language-model loss and perplexity.

## Structure

- `koopman_proxy_experiments/`: Code for the Koopman proxy experiments.
- `next_token_experiments/`: Code for the downstream next-token prediction experiments.
- `papers/`: The project poster and final report.
- `results/`: CSVs, LaTeX tables, and plots used in the report.

## Papers

- [Project poster: Learning Spectral Initializations for Low-Rank Attention](papers/Learning_Spectral_Initializations_Poster.pdf)
- [CS229 project final report](papers/CS229_Project_Final_Report.pdf)

## Main Scripts

Koopman proxy:

- `koopman_proxy_experiments/svd_init_ablation/code/all_learned_koopman.py`: runs the main proxy comparison over fixed SVD strategies and learned spectral strategies.
- `koopman_proxy_experiments/svd_init_ablation/code/add_random_orthogonal_proxy.py`: adds the random orthogonal baseline to the proxy results.
- `koopman_proxy_experiments/svd_init_ablation/code/plot_combined_koopman.py`: produces the combined proxy plots.

Next-token:

- `next_token_experiments/scripts/run_pythia_retrofit_grid.py`: runs the next-token retrofit grid over ranks, seeds, and initialization strategies.
- `next_token_experiments/scripts/plot_stage5_results.py`: produces next-token plots and result tables.
- `next_token_experiments/scripts/significance_stage5_results.py`: computes paired significance tests for final next-token LM loss.
- `next_token_experiments/scripts/export_mlp_scaler_checkpoint.py`: exports trained MLP scaler checkpoints used by the MLP initialization strategy.

## Dependencies

The experiments use PyTorch, Transformers, Datasets, NumPy, Pandas, Matplotlib, and SciPy. Full reruns require downloading Pythia-160M and WikiText-103 from Hugging Face. The included `results/` folder contains the outputs used in the report.

## Relationship to CS197 Work

This CS229 project builds on earlier CS197/SKA work but asks a different question. The CS197 work focused on fixed low-rank initialization strategies, including PCA, SVD-sqrt, SVD-noscale, SVD-full, and random orthogonal initialization. Those experiments emphasized reconstruction error and spectral diagnostics such as condition number and spectral gap.

In contrast, this CS229 project introduces data-driven initialization methods, including learned \(b\), learned \(a,b\), and an MLP-based singular-value scaling rule. We also evaluate the strategies using predictive objectives: a Koopman proxy objective based on next-state relative MSE, and a downstream next-token prediction experiment using language-model loss and perplexity.


## Reproducing Included Results

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The plotting scripts force Matplotlib's non-interactive `Agg` backend, so they can run on headless machines.

The proxy plots can be regenerated from the packaged CSVs with:

```bash
python koopman_proxy_experiments/svd_init_ablation/code/plot_combined_koopman.py
```

The next-token result tables and plots in `results/next_token_plots/` are precomputed outputs. To regenerate them from scratch, run the next-token experiments to produce per-run `metrics.csv` files, then pass the raw run directory or comma-separated raw run directories to:

```bash
python next_token_experiments/scripts/plot_stage5_results.py --root <raw_run_dir_or_dirs>
python next_token_experiments/scripts/significance_stage5_results.py --root <raw_run_dir_or_dirs>
```

For a non-executing command preview of the next-token grid, use:

```bash
python next_token_experiments/scripts/run_pythia_retrofit_grid.py --dry_run
```
