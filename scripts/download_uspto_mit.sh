#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-data/raw_uspto_mit}
URL="https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip"
OUT_FILE="${OUT_DIR}/data.zip"

mkdir -p "$OUT_DIR"

echo "Downloading USPTO-MIT / USPTO-480K"
echo "Source: $URL"
echo "Output: $OUT_FILE"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "$OUT_FILE" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT_FILE" "$URL"
else
  echo "ERROR: neither curl nor wget is available." >&2
  exit 1
fi

python - "$OUT_FILE" <<'PY'
import sys
import zipfile

path = sys.argv[1]
try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError("corrupt member: {}".format(bad))
        names = zf.namelist()
except Exception as exc:
    raise SystemExit("ERROR: downloaded file is not a valid zip archive: {}".format(exc))

print("Zip archive verified.")
print("Members:")
for name in names[:20]:
    print("  " + name)
PY

cat <<EOF

Next step:

python scripts/convert_uspto_mit.py \\
  --input "$OUT_FILE" \\
  --format jin \\
  --out data/uspto_mit \\
  --include_reagents false \\
  --split predefined

EOF
