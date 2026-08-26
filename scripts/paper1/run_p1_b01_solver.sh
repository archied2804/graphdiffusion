#!/bin/bash
# P1-B01 -- solver half. Does generation beat a lookup table?
#
# For each evaluation target, p1_b01_retrieval_index.py found the nearest
# TRAINING shape in K=8 DCT-Cp space -- the exact vector the model is
# conditioned on, so retrieval is handed precisely the information the model
# gets. This runs that retrieved geometry at the TARGET's Re and H, which is the
# honest comparison: the retrieved shape is asked to perform at a condition it
# was not drawn from, exactly as a generated shape is.
#
#   20  in-distribution targets   the set P1-G01 will use, leaked rows excluded
#   15  OOD targets               3 each from lowre, highre, thin, thick, aoa10
#
# 35 runs, the spec's budget. Grading is against the TARGET's dataset reference
# curve, not the retrieved sample's.
#
# Usage:
#   scripts/paper1/run_p1_b01_solver.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/cfd/ellipse2d/ellipse2d"
SHAPES="$ROOT/cfd/ellipse2d/shapes/b01_retrieval"
WORK="$ROOT/outputs/paper1/P1-B01_retrieval-baseline/solver"
LOG="$WORK/b01_solver.log"
MIN_FREE_GB=20

MAXT=50
AVG=25

mkdir -p "$WORK"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[ -x "$BIN" ]    || { say "ERROR: $BIN not built"; exit 1; }
[ -d "$SHAPES" ] || { say "ERROR: run p1_b01_make_targets.py first"; exit 1; }

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
    say "    FAILED -- see run.log"; return 0
  done
  say "    FAILED -- GPU-WATCHDOG after 3 attempts"
}

say "=========================================================="
say "P1-B01 solver half -- nearest-neighbour retrieval baseline"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

tail -n +2 "$SHAPES/manifest.csv" | while IFS=, read -r tag set tidx ridx nn re h nt nr; do
  [ -z "${tag:-}" ] && continue
  dat="$SHAPES/${tag}_retrieved.dat"
  [ -f "$dat" ] || { say "  MISSING $dat"; continue; }
  rule=$(awk -v r="$re" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }')
  run "$tag" "$dat" "$rule" "$re" "$h"
done

say "P1-B01 solver runs complete -- grade with scripts/paper1/p1_b01_grade.py"
say "free disk: $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
