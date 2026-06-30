#!/usr/bin/env bash
set -euo pipefail
DATA_IN=${1:-data/raw_uspto_mit}
OUT=${2:-data/uspto_mit}
python scripts/convert_uspto_mit.py --input "$DATA_IN" --format auto --out "$OUT" --include_reagents false --split predefined
for mode in train valid test; do
  python canonicalize_prod.py --dataset uspto_mit --mode "$mode"
  python preprocess.py --dataset uspto_mit --mode "$mode" --input_stage auto --lg_min_count 50 --print_every 10000
  python prepare_data.py --dataset uspto_mit --mode "$mode" --batch_size 256 --max_steps 9 --print_every 10000
done
python train.py --dataset uspto_mit --epochs 150 --lr 0.0001 --num_workers 6 --print_every 200
# Full USPTO-MIT/480k training is expected to use a GPU.
# Example evaluation command after training:
# python eval.py --dataset uspto_mit --experiments <timestamp> --checkpoint best --beam_size 10 --max_steps 9
