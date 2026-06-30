#!/usr/bin/env bash
set -euo pipefail
rm -rf data/uspto_mit_tiny data/uspto_mit
python scripts/convert_uspto_mit.py --input tests/fixtures/uspto_mit_tiny/deepchem.csv --format deepchem --out data/uspto_mit_tiny --split random --limit_rows 3
python scripts/convert_uspto_mit.py --input tests/fixtures/uspto_mit_tiny/graph2edits.csv --format graph2edits --out data/uspto_mit_tiny --split random --limit_rows 3
python scripts/convert_uspto_mit.py --input tests/fixtures/uspto_mit_tiny --format jin --out data/uspto_mit_tiny --split predefined --limit_rows 3
cp -r data/uspto_mit_tiny data/uspto_mit
for mode in train valid test; do
  python canonicalize_prod.py --dataset uspto_mit --mode "$mode"
  python preprocess.py --dataset uspto_mit --mode "$mode" --input_stage auto --lg_min_count 1 --print_every 10000
  python prepare_data.py --dataset uspto_mit --mode "$mode" --batch_size 2 --max_steps 9 --print_every 10000
done
python -m py_compile scripts/convert_uspto_mit.py canonicalize_prod.py preprocess.py prepare_data.py train.py eval.py utils/datasets.py utils/dataset_config.py
python train.py --help >/dev/null
python eval.py --help >/dev/null
