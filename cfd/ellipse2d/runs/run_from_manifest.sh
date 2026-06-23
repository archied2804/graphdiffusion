#!/bin/bash
# Launch one OpenLB run per manifest row, using that row's own Re and H so
# the simulated domain always matches the dataset sample. The body polygon
# (.dat) determines the shape; --SEMI_MINOR_B is deliberately omitted.
#
# Usage:
#   run_from_manifest.sh <shapes_dir> [RESOLUTION] [MAX_PHYS_T]
# Expects <shapes_dir>/manifest.csv with a header containing idx, Re, H (and
# an optional name column) and a sibling <shapes_dir>/<name>.dat or
# sim_<idx>.dat per row.
#
# Column order is resolved from the header, so any column ordering works.
# If a name column is present, the run subdir and .dat file use that name;
# otherwise sim_<idx> is used. If <name>.dat does not exist, falls back to
# sim_<idx>.dat.
set -euo pipefail

BIN="$(cd "$(dirname "$0")/.." && pwd)/ellipse2d"
RUNDIR="$(cd "$(dirname "$0")" && pwd)"
SHAPES_DIR="${1:?usage: run_from_manifest.sh <shapes_dir> [RES] [MAX_T]}"
RES="${2:-100}"
MAX_T="${3:-50}"
MANIFEST="$SHAPES_DIR/manifest.csv"

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }
[ -x "$BIN" ]      || { echo "ERROR: binary not found or not executable: $BIN"; exit 1; }

# Resolve column indices from the header (robust to any column ordering).
# col() returns the 1-based column number for a given field name.
header=$(head -1 "$MANIFEST")
col() { echo "$header" | tr ',' '\n' | grep -nx "$1" | cut -d: -f1; }

c_idx=$(col idx)
c_re=$(col Re)
c_h=$(col H)
# name column is optional; the || true prevents set -e from aborting when absent.
c_name=$(col name || true)

# Process every data row (skip header).
# NOTE: the while loop runs inside a pipe subshell. Variables it sets are not
# visible outside, which is fine — we never collect results after the loop.
tail -n +2 "$MANIFEST" | while IFS=',' read -r -a f; do
  idx="${f[$((c_idx-1))]}"; re="${f[$((c_re-1))]}"; h="${f[$((c_h-1))]}"

  # Determine run name and .dat path.
  if [ -n "${c_name:-}" ]; then
    name="${f[$((c_name-1))]}"
    dat="$SHAPES_DIR/${name}.dat"
  else
    name="sim_${idx}"
    dat="$SHAPES_DIR/sim_${idx}.dat"
  fi
  # Fallback: if the preferred .dat doesn't exist, try sim_<idx>.dat.
  [ -f "$dat" ] || dat="$SHAPES_DIR/sim_${idx}.dat"

  d="$RUNDIR/$name"
  mkdir -p "$d"
  rm -rf "$d/tmp" "$d/run.log"

  echo "launching $name: Re=$re H=$h RES=$RES"
  ( cd "$d" && "$BIN" "$dat" \
      --REYNOLDS "$re" \
      --CHANNEL_HEIGHT "$h" \
      --RESOLUTION "$RES" \
      --MAX_PHYS_T "$MAX_T" \
      --AVG_WINDOW_CHORDS 25 \
      > run.log 2>&1 ) || echo "  WARN: $name exited non-zero"
  echo "  $name done: $(tail -1 "$d/tmp/ellipse2d_force.csv" 2>/dev/null || echo '(no force.csv)')"
done
echo ALL_DONE
