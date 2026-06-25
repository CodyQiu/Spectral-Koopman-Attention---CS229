#!/bin/bash
set -e

uv run python scripts/run_pythia_retrofit_grid.py \
  --output_root runs/pythia_retrofit_grid_layer2_moreseeds \
  --model EleutherAI/pythia-160m \
  --strategies svd_sqrt,svd_full,svd_noscale,change_b,change_a_b,mlp_log_singular_values \
  --ranks 8,16,32,64 \
  --seeds 3,4,5 \
  --steps 300 \
  --max_seq_len 256 \
  --batch_size 4 \
  --grad_accum 4 \
  --n_ska_layers 1 \
  --ska_layer_indices 2 \
  --change_b_by_rank '{"8":4.0,"16":4.0384,"32":4.0025,"64":3.9938}' \
  --change_ab_by_rank '{"8":{"a":0.0216,"b":4.0},"16":{"a":2.8645,"b":4.0384},"32":{"a":9.4683,"b":4.0026},"64":{"a":0.0928,"b":3.9935}}' \
  --mlp_state_path runs/mlp_scaler_pythia160m_layer2.pt \
  --mlp_hidden_dim 32 \
  --mlp_residual_scale 0.3
