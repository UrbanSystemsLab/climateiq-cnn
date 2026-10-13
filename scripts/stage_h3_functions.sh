#!/usr/bin/env bash
# Stage the H3 pipeline Cloud Function source into a flat directory ready
# for `gcloud functions deploy --source`.
#
# Usage: [REPO_ROOT=<checkout>] scripts/stage_h3_functions.sh [OUT_DIR]
#   REPO_ROOT defaults to this script's repository.
#   OUT_DIR defaults to build/h3_functions_source (gitignored).
# Prints OUT_DIR on success.
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$ROOT/usl_pipeline/h3_cloud_functions"
OUT="${1:-$ROOT/build/h3_functions_source}"

rm -rf "$OUT"
mkdir -p "$OUT/usl_h3_tools"

cp "$SRC/main.py" "$SRC/requirements.txt" "$SRC/cities_config.yaml" "$OUT/"
cp "$SRC/cli_unified_pipeline.py" "$SRC/cli_add_admin_breadcrumbs.py" "$SRC/cli_preprocess_tiff_to_h3csv.py" "$OUT/"
rsync -a --include='*/' --include='*.py' --exclude='*' --prune-empty-dirs \
  "$SRC/usl_h3_tools/" "$OUT/usl_h3_tools/"

printf '__pycache__/\n*.pyc\n' > "$OUT/.gcloudignore"

echo "$OUT"
