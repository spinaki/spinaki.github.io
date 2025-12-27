#!/usr/bin/env bash
# usage: scripts/nb_to_md.sh notebooks/demo.ipynb _posts/2025-12-02-demo.md
set -euo pipefail
NB="$1"
OUT="$2"

jupyter nbconvert --to markdown "$NB" --output "$(basename "$OUT")"

# Move the generated file and images into place
GEN_DIR="$(dirname "$NB")"
GEN_MD="$(basename "$OUT")"
mv "$GEN_DIR/$GEN_MD" "$OUT"

# You may need to adjust image paths manually to point into /assets/notebooks/
echo "Converted $NB -> $OUT (check image paths)."
