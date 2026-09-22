#!/bin/bash
# Run the per-draw arm-agreement analysis for every row of a frozen batch.
#   arm_prefix_batch.sh <batch_tag> [out_dir]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-refreeze_2026-09-08}"
OUT="${2:-$HERE/logs_frozen/$TAG/agreement}"
mkdir -p "$OUT"
for s in "$HERE/logs_frozen/$TAG"/*_sicm.out.json; do
  [ -e "$s" ] || continue
  base="$(basename "$s" _sicm.out.json)"
  n="$HERE/logs_frozen/$TAG/${base}_nosicm.out.json"
  [ -e "$n" ] || { echo "== $base: no nosicm arm"; continue; }
  echo "== $base"
  python3 "$HERE/arm_prefix.py" "$s" "$n" --json "$OUT/${base}.json" | tail -4
done
