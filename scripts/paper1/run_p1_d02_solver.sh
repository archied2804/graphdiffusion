#!/bin/bash
# P1-D02 -- node-count transfer.
#
# Every generated body is run at its TARGET's Re and H and graded against the
# TARGET's dataset Cp curve. The ladder spans 24-384 boundary nodes, against a
# trained range of 52-96.
#
# RESOLUTION follows the Re rule and does NOT co-vary with the node count. That
# is the spec's own requirement: if the solver grid tracked N, the experiment
# would confound model behaviour with solver discretisation.
#
# 12 rungs x 4 targets x 3 seeds = 144 runs.
#
# Usage:
#   scripts/paper1/run_p1_d02_solver.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/cfd/ellipse2d/ellipse2d"
# Two body sets go through this runner:
#   g01_holdout  the generated shapes           (default)
#   g01_truth    the TARGET geometries themselves, which give the reference Cd
#                the spec's "Cd generated vs target" measurement needs. Both
#                sides then pass through the same solver, so F07's constant
#                9.35% offset cancels.
SHAPES="${SHAPES:-$ROOT/cfd/ellipse2d/shapes/d02_nodes}"
WORK="${WORK:-$ROOT/outputs/paper1/P1-D02_node-count-transfer/solver}"
LOG="$WORK/d02_solver.log"
MIN_FREE_GB=20

MAXT=50
AVG=25

mkdir -p "$WORK"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[ -x "$BIN" ]    || { say "ERROR: $BIN not built"; exit 1; }
[ -d "$SHAPES" ] || { say "ERROR: run p1_d02_sample.py first"; exit 1; }

run() {
  local tag="$1" dat="$2" res="$3" re="$4" h="$5"
  local d="$WORK/$tag"
  if ls "$d"/tmp/*_force.csv >/dev/null 2>&1; then say "  skip $tag (done)"; return 0; fi
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  [ "$free" -lt "$MIN_FREE_GB" ] && { say "ABORT: only ${free}G free"; exit 1; }

  local attempt
  for attempt in 1 2 3; do
    rm -rf "$d"; mkdir -p "$d"
    cp "$dat" "$d/body.dat"
    [ "$attempt" -eq 1 ] \
      && say "  $tag (RES=$res Re=$re H=$h)" \
      || say "  $tag retry $attempt/3"
    ( cd "$d" && "$BIN" body.dat --REYNOLDS "$re" --CHANNEL_HEIGHT "$h" \
        --RESOLUTION "$res" --MAX_PHYS_T "$MAXT" --AVG_WINDOW_CHORDS "$AVG" \
        > run.log 2>&1 )
    if ls "$d"/tmp/*_force.csv >/dev/null 2>&1; then
      say "    OK$([ "$attempt" -gt 1 ] && echo " (attempt $attempt)")"
      "$(dirname "$0")/prune_run_output.sh" "$d" >> "$LOG" 2>&1
      return 0
    fi
    if grep -qi "launch timed out" "$d/run.log" 2>/dev/null; then
      say "    GPU-WATCHDOG on attempt $attempt"; sleep 20; continue
    fi
    # A generated body can be geometrically invalid in a way the solver refuses.
    # That is a RESULT (the spec measures a validity rate), not a harness fault.
    say "    FAILED -- $(grep -iE 'exception|error' "$d/run.log" | head -1 | cut -c1-70)"
    return 0
  done
  say "    FAILED -- GPU-WATCHDOG after 3 attempts"
}

say "=========================================================="
say "P1-D02 node-count transfer"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

tail -n +2 "$SHAPES/manifest.csv" | while IFS=, read -r tag label nn zone seed tidx re h ng; do
  [ -z "${tag:-}" ] && continue
  dat="$SHAPES/${tag}.dat"
  [ -f "$dat" ] || { say "  MISSING $dat"; continue; }
  rule=$(awk -v r="$re" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }')
  run "$tag" "$dat" "$rule" "$re" "$h"
done

say "P1-G01 solver runs complete -- grade with scripts/paper1/p1_d02_grade.py"
say "free disk: $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
