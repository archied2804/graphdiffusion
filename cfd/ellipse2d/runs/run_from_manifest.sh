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
#
# RESOLUTION: if omitted, it is auto-scaled PER ROW as RES = max(100, 0.2*Re)
# rounded to the nearest integer. This follows the validated stability rule —
# cells-per-chord must grow with Re or high-Re / thicker bodies exceed the Mach
# limit and diverge (e.g. Re754->160, Re1000->200). Pass an explicit value to
# override the auto-scaling and use one fixed resolution for every row.
set -euo pipefail

BIN="$(cd "$(dirname "$0")/.." && pwd)/ellipse2d"
RUNDIR="$(cd "$(dirname "$0")" && pwd)"
SHAPES_DIR="${1:?usage: run_from_manifest.sh <shapes_dir> [RES] [MAX_T]}"
RES_OVERRIDE="${2:-}"   # empty => auto-scale RES per row from Re (see header)
MAX_T="${3:-50}"

# res_for_re <Re> -> echoes max(100, round(0.2*Re)); the stability rule.
res_for_re() { awk -v r="$1" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }'; }
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

  # Resolution: explicit override if given, else auto-scale from Re.
  if [ -n "$RES_OVERRIDE" ]; then
    res="$RES_OVERRIDE"
  else
    res=$(res_for_re "$re")
  fi

  d="$RUNDIR/$name"
  mkdir -p "$d"
  rm -rf "$d/tmp" "$d/run.log"

  echo "launching $name: Re=$re H=$h RES=$res"
  ( cd "$d" && "$BIN" "$dat" \
      --REYNOLDS "$re" \
      --CHANNEL_HEIGHT "$h" \
      --RESOLUTION "$res" \
      --MAX_PHYS_T "$MAX_T" \
      --AVG_WINDOW_CHORDS 25 \
      > run.log 2>&1 ) || echo "  WARN: $name exited non-zero"
  echo "  $name done: $(tail -1 "$d/tmp/ellipse2d_force.csv" 2>/dev/null || echo '(no force.csv)')"
done
echo ALL_DONE
