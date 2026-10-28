#!/bin/bash
# P1-F03 -- solver instrument qualification.
#
# Two parts, per the spec (step 2, the external benchmark, moved to P1-F06):
#
#   Part A  envelope gate      -- all 15 eval_grid cases at the stability-rule
#                                 resolution RES = max(100, 0.2*Re).  15 runs.
#   Part B  resolution sweep   -- 3 cases spanning the Re envelope, each at
#                                 {0.5, 1.0, 1.5, 2.0} x the rule.  12 runs.
#
# 27 solver runs total.  Runs are launched serially by run_from_manifest.sh;
# a diverged run is recorded, not hidden (spec method step 4).
#
# Usage:
#   scripts/paper1/run_p1_f03.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHAPES="$ROOT/cfd/ellipse2d/shapes/eval_grid"
RUNNER="$ROOT/cfd/ellipse2d/runs/run_from_manifest.sh"
WORK="$ROOT/outputs/paper1/P1-F03_solver-qualification"
STAGE="$WORK/stage"

mkdir -p "$WORK" "$STAGE"
LOG="$WORK/f03.log"
: > "$LOG"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "P1-F03 starting"
say "git: $(git -C "$ROOT" rev-parse --short HEAD) dirty=$(git -C "$ROOT" status --porcelain -- train.py src/ configs/ cfd/ | wc -l)"

# --- Part A: envelope gate, stability-rule resolution ----------------------
say "Part A -- envelope gate, 15 cases at RES = max(100, 0.2*Re)"
"$RUNNER" "$SHAPES" "" 50 >> "$LOG" 2>&1
say "Part A done"

# Park Part A results so Part B's reruns of the same names cannot clobber them.
mkdir -p "$WORK/envelope"
while IFS=',' read -r name _rest; do
  [ "$name" = "name" ] && continue
  src="$ROOT/cfd/ellipse2d/runs/$name"
  [ -d "$src" ] && cp -r "$src" "$WORK/envelope/$name"
done < "$SHAPES/manifest.csv"
say "Part A archived to $WORK/envelope"

# --- Part B: resolution sensitivity ----------------------------------------
# Three cases spanning the Re envelope: lowest, middle, highest.
CASES=("re0501_h50_idx0018" "re0751_h50_idx2833" "re0998_h50_idx5642")
say "Part B -- resolution sweep on ${CASES[*]}"

for name in "${CASES[@]}"; do
  row=$(grep "^$name," "$SHAPES/manifest.csv")
  re=$(echo "$row" | cut -d, -f3)
  h=$(echo "$row" | cut -d, -f4)
  base=$(awk -v r="$re" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }')

  for mult in 0.5 1.0 1.5 2.0; do
    res=$(awk -v b="$base" -v m="$mult" 'BEGIN { printf "%d", b * m + 0.5 }')
    tag="${name}_res${res}"
    # One-row manifest so the shared runner handles the actual invocation.
    printf 'name,idx,Re,H\n%s,%s,%s,%s\n' "$tag" "$(echo "$row" | cut -d, -f2)" "$re" "$h" \
      > "$STAGE/manifest.csv"
    # Shape files are named sim_<idx>.dat, not <name>.dat.
  idx=$(echo "$row" | cut -d, -f2)
  cp "$SHAPES/sim_${idx}.dat" "$STAGE/${tag}.dat"
    say "  $tag (Re=$re, RES=$res, ${mult}x rule)"
    "$RUNNER" "$STAGE" "$res" 50 >> "$LOG" 2>&1
    src="$ROOT/cfd/ellipse2d/runs/$tag"
    if [ -d "$src" ]; then
      mkdir -p "$WORK/resolution"
      cp -r "$src" "$WORK/resolution/$tag"
    else
      say "  WARN: no output for $tag"
    fi
  done
done

say "Part B done"
say "P1-F03 solver runs complete -- grade with scripts/paper1/p1_f03_grade.py"
