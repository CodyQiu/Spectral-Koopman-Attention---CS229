#!/bin/bash
set -e

/home/users/cody1212/.local/bin/uv run python scripts/run_pythia_retrofit_grid.py \
  --output_root runs/pythia_retrofit_grid_layer2_random_orth_moreseeds \
  --model EleutherAI/pythia-160m \
  --strategies random_orth \
  --ranks 8,16,32,64 \
  --seeds 3,4,5 \
  --steps 300 \
  --max_seq_len 256 \
  --batch_size 4 \
  --grad_accum 4 \
  --n_ska_layers 1 \
  --ska_layer_indices 2
