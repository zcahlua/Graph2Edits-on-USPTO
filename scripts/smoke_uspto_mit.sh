#!/usr/bin/env bash
set -euo pipefail

changed_py=(
  scripts/convert_uspto_mit.py
  canonicalize_prod.py
  preprocess.py
  prepare_data.py
  train.py
  eval.py
  utils/datasets.py
  utils/dataset_config.py
)

assert_nonempty_csv() {
  local file="$1"
  test -s "$file"
  local rows
  rows=$(python - "$file" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as fh:
    print(max(0, sum(1 for _ in csv.reader(fh)) - 1))
PY
)
  if [ "$rows" -le 0 ]; then
    echo "Expected non-empty CSV: $file" >&2
    exit 1
  fi
}

rm -rf data/uspto_mit_tiny_jin data/uspto_mit_tiny_deepchem data/uspto_mit_tiny_graph2edits
python scripts/convert_uspto_mit.py --input tests/fixtures/uspto_mit_tiny --format jin --out data/uspto_mit_tiny_jin --split predefined
python scripts/convert_uspto_mit.py --input tests/fixtures/uspto_mit_tiny/deepchem.csv --format deepchem --out data/uspto_mit_tiny_deepchem --split random --limit_rows 3
python scripts/convert_uspto_mit.py --input tests/fixtures/uspto_mit_tiny/graph2edits.csv --format graph2edits --out data/uspto_mit_tiny_graph2edits --split random --limit_rows 3

for out in data/uspto_mit_tiny_jin data/uspto_mit_tiny_deepchem data/uspto_mit_tiny_graph2edits; do
  for mode in train valid test; do
    assert_nonempty_csv "$out/raw_${mode}.csv"
  done
done

for mode in train valid test; do
  python canonicalize_prod.py --dataset uspto_mit_tiny_jin --mode "$mode"
  python preprocess.py --dataset uspto_mit_tiny_jin --mode "$mode" --input_stage auto --lg_min_count 1 --print_every 10000
  python prepare_data.py --dataset uspto_mit_tiny_jin --mode "$mode" --batch_size 2 --max_steps 9 --print_every 10000
done

python -m py_compile "${changed_py[@]}"
python scripts/convert_uspto_mit.py --help >/dev/null
python canonicalize_prod.py --help >/dev/null
python preprocess.py --help >/dev/null
python prepare_data.py --help >/dev/null
python train.py --help >/dev/null
python eval.py --help >/dev/null
