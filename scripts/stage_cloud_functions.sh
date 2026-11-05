#!/usr/bin/env bash
# Stage the data-pipeline Cloud Functions source into a flat directory ready
# for `gcloud functions deploy --source`.
#
# The file list mirrors climateiq-terraform/modules/data_pipeline/main.tf
#
# Usage: [REPO_ROOT=<checkout>] scripts/stage_cloud_functions.sh [OUT_DIR]
#   REPO_ROOT defaults to this script's repository.
#   OUT_DIR defaults to build/cloud_functions_source (gitignored).
# Prints OUT_DIR on success.
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$ROOT/usl_pipeline/cloud_functions"
LIB="$ROOT/usl_pipeline/usl_lib/usl_lib"
OUT="${1:-$ROOT/build/cloud_functions_source}"

rm -rf "$OUT"
mkdir -p "$OUT/wheels" "$OUT/usl_lib"

cp "$SRC/main.py" "$SRC/requirements.txt" "$OUT/"
cp "$SRC"/wheels/*.whl "$OUT/wheels/"
rsync -a --include='*/' --include='*.py' --exclude='*' --prune-empty-dirs \
  "$LIB/" "$OUT/usl_lib/"

# Stop gcloud from generating its own .gcloudignore and keep caches out.
printf '__pycache__/\n*.pyc\n' > "$OUT/.gcloudignore"

echo "$OUT"
