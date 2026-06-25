import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-14m")
    parser.add_argument("--output_root", default="runs/pythia_retrofit_grid")
    parser.add_argument("--strategies", default="svd_sqrt,svd_full,svd_noscale,change_b,change_a_b")
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n_ska_layers", type=int, default=1)
    parser.add_argument("--ska_layer_indices", default=None)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=1000000)
    parser.add_argument("--ska_lr", type=float, default=1e-4)
    parser.add_argument("--change_b_power", type=float, default=4.0)
    parser.add_argument("--change_b_by_rank", default=None,
                        help='Optional JSON rank map, e.g. {"16":4.0,"32":3.5}')
    parser.add_argument("--change_ab_scale", type=float, default=1.0)
    parser.add_argument("--change_ab_power", type=float, default=4.0)
    parser.add_argument("--change_ab_by_rank", default=None,
                        help='Optional JSON rank map, e.g. {"16":{"a":2.8,"b":4.0}}')
    parser.add_argument("--mlp_state_path", default=None)
    parser.add_argument("--mlp_hidden_dim", type=int, default=32)
    parser.add_argument("--mlp_residual_scale", type=float, default=0.3)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def parse_int_list(raw):
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_str_list(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def strategy_args(strategy, rank, args, change_b_by_rank, change_ab_by_rank):
    if strategy == "change_b":
        power = change_b_by_rank.get(str(rank), args.change_b_power) if change_b_by_rank else args.change_b_power
        return ["--power_b", str(power)]

    if strategy == "change_a_b":
        params = change_ab_by_rank.get(str(rank), {}) if change_ab_by_rank else {}
        scale = params.get("a", args.change_ab_scale)
        power = params.get("b", args.change_ab_power)
        return ["--scale_a", str(scale), "--power_b", str(power)]

    if strategy == "mlp_log_singular_values":
        if args.mlp_state_path is None:
            raise ValueError(
                "mlp_log_singular_values requires --mlp_state_path. "
                "Run scripts/export_mlp_scaler_checkpoint.py first."
            )
        return [
            "--mlp_state_path", args.mlp_state_path,
            "--mlp_hidden_dim", str(args.mlp_hidden_dim),
            "--mlp_residual_scale", str(args.mlp_residual_scale),
        ]

    return []


def main():
    args = parse_args()
    strategies = parse_str_list(args.strategies)
    ranks = parse_int_list(args.ranks)
    seeds = parse_int_list(args.seeds)
    change_b_by_rank = json.loads(args.change_b_by_rank) if args.change_b_by_rank else None
    change_ab_by_rank = json.loads(args.change_ab_by_rank) if args.change_ab_by_rank else None

    experiment_dir = Path(__file__).resolve().parents[1]
    train_script = experiment_dir / "train_ska_retrofit.py"

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    total = len(strategies) * len(ranks) * len(seeds)
    run_idx = 0

    for strategy in strategies:
        for rank in ranks:
            for seed in seeds:
                run_idx += 1
                out = root / f"{strategy}_rank{rank}_seed{seed}"
                cmd = [
                    sys.executable, str(train_script),
                    "--output_dir", str(out),
                    "--stage", "5",
                    "--model", args.model,
                    "--init_strategy", strategy,
                    "--ska_rank", str(rank),
                    "--n_ska_layers", str(args.n_ska_layers),
                    "--max_seq_len", str(args.max_seq_len),
                    "--batch_size", str(args.batch_size),
                    "--grad_accum", str(args.grad_accum),
                    "--single_stage_steps", str(args.steps),
                    "--warmup_steps", str(args.warmup_steps),
                    "--logging_steps", str(args.logging_steps),
                    "--save_steps", str(args.save_steps),
                    "--ska_lr", str(args.ska_lr),
                    "--seed", str(seed),
                ]
                if args.ska_layer_indices is not None:
                    cmd.extend(["--ska_layer_indices", args.ska_layer_indices])
                cmd.extend(strategy_args(strategy, rank, args, change_b_by_rank, change_ab_by_rank))

                print(f"\n[{run_idx}/{total}] {' '.join(cmd)}", flush=True)
                if args.dry_run:
                    continue
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
