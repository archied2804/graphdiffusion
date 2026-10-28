#!/bin/bash
# P1-F06 follow-up -- repeat P1-F03's Part B resolution sweep in fp64.
#
# THE DISCRIMINATING TEST. F03 found the affine slope `a` moving by up to 25.9%
# from numerical resolution alone at fixed Reynolds number, and concluded that
# `a` is not a stable calibration constant. F06 then found that in fp32 the lift
# coefficient stops converging under refinement -- 29.5% scatter at N=80 against
# 0.1% in fp64 -- because LBM recovers pressure from density fluctuations of
# order Ma^2 and the surface integrals near-cancel.
#
# If the fp32 round-off is the cause, repeating the identical sweep in fp64
# should collapse the spread in `a`. If the spread survives, the resolution
# dependence is physical (or a case defect) and precision is a side issue.
#
# The sweep matches F03 exactly -- same three cases, same four multipliers of
# the stability rule, same geometry, same averaging window -- so the only
# difference is the floating-point type. Anything else would confound it.
#
# fp64 costs ~1.8x wall clock here, not the 32-64x a compute-bound code would
# pay: LBM is memory-bandwidth bound, so fp64 mostly doubles memory traffic.
#
# Usage:
#   scripts/paper1/run_f03_fp64_sweep.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
D=/home/m22729ad/Software/OpenLB-1.9-0-double/olb-release
BIN="$D/examples/ellipse2d_fp64/ellipse2d_fp64"
SHAPES="$ROOT/cfd/ellipse2d/shapes/eval_grid"
WORK="$ROOT/outputs/paper1/P1-F06_schafer-turek/f03_fp64_sweep"
LOG="$WORK/sweep.log"
MIN_FREE_GB=40

mkdir -p "$WORK"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[ -x "$BIN" ] || { say "ERROR: fp64 binary not built: $BIN"; exit 1; }

CASES=("re0501_h50_idx0018" "re0751_h50_idx2833" "re0998_h50_idx5642")

say "=========================================================="
say "F03 Part B repeated in fp64 -- does the spread in the affine slope collapse?"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

for name in "${CASES[@]}"; do
  row=$(grep "^$name," "$SHAPES/manifest.csv") || { say "  no manifest row for $name"; continue; }
  idx=$(echo "$row" | cut -d, -f2)
  re=$(echo "$row"  | cut -d, -f3)
  h=$(echo "$row"   | cut -d, -f4)
  base=$(awk -v r="$re" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }')

  for mult in 0.5 1.0 1.5 2.0; do
    res=$(awk -v b="$base" -v m="$mult" 'BEGIN { printf "%d", b * m + 0.5 }')
    d="$WORK/${name}_res${res}"
    if [ -s "$d/tmp/ellipse2d_fp64_force.csv" ] || [ -s "$d/tmp/ellipse2d_force.csv" ]; then
      say "  skip ${name}_res${res} (done)"; continue
    fi
    free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
    [ "$free" -lt "$MIN_FREE_GB" ] && { say "ABORT: only ${free}G free"; exit 1; }

    rm -rf "$d"; mkdir -p "$d"
    cp "$SHAPES/sim_${idx}.dat" "$d/body.dat"
    say "  ${name}_res${res} (Re=$re RES=$res ${mult}x rule)"
    ( cd "$d" && "$BIN" body.dat \
        --REYNOLDS "$re" --CHANNEL_HEIGHT "$h" --RESOLUTION "$res" \
        --MAX_PHYS_T 50 --AVG_WINDOW_CHORDS 25 > run.log 2>&1 )
    if ls "$d"/tmp/*force.csv >/dev/null 2>&1; then
      say "    OK"
      "$(dirname "$0")/prune_run_output.sh" "$d" >> "$LOG" 2>&1
    else
      reason=$(grep -i "launch timed out" "$d/run.log" >/dev/null 2>&1 \
               && echo "GPU-WATCHDOG" || echo "see run.log")
      say "    FAILED -- $reason"
    fi
  done
done

say "fp64 sweep complete -- grade with p1_f03_grade.py against the fp32 numbers"
