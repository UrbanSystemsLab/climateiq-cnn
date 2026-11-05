#!/usr/bin/env bash
# Stage the post-processing Cloud Functions source into a flat directory ready
# for `gcloud functions deploy --source`.
#
# Usage: [REPO_ROOT=<checkout>] scripts/stage_postprocessing.sh [OUT_DIR]
#   REPO_ROOT defaults to this script's repository.
#   OUT_DIR defaults to build/postprocessing_source (gitignored).
# Prints OUT_DIR on success.
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$ROOT/usl_pipeline/4_postprocessing"
LIB="$ROOT/usl_pipeline/usl_lib/usl_lib"
OUT="${1:-$ROOT/build/postprocessing_source}"

rm -rf "$OUT"
mkdir -p "$OUT/wheels" "$OUT/usl_lib"

cp "$SRC/main.py" "$SRC/requirements.txt" "$SRC/cities_config.yaml" "$OUT/"
cp "$SRC"/wheels/*.whl "$OUT/wheels/"
rsync -a --include='*/' --include='*.py' --exclude='*' --prune-empty-dirs \
  "$LIB/" "$OUT/usl_lib/"
rsync -a --include='*/' --include='*.py' --exclude='*' --prune-empty-dirs \
  "$SRC/postprocessing/" "$OUT/postprocessing/"

# Stop gcloud from generating its own .gcloudignore and keep caches out.
printf '__pycache__/\n*.pyc\n' > "$OUT/.gcloudignore"

echo "$OUT"
