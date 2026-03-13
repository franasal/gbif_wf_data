#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${ROOT_DIR}/tools/fetch_protected_areas.py" \
  --tmp-dir "${ROOT_DIR}/tmp/protected_areas_raw" \
  --output "${ROOT_DIR}/data/protected_areas_de.json.gz"

ls -lh "${ROOT_DIR}/data/protected_areas_de.json.gz"
