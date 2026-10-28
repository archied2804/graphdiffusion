#!/bin/bash
# P1-B03 -- solver half. Convert PCA reconstruction error into a physical
# tolerance.
#
# The analytic half measured reconstruction error in units of normalised radius.
# The spec is explicit that this is not a physically meaningful tolerance, and
# asks for the conversion:
#
#   "Take 4 held-out shapes, reconstruct each at M = 3 and at M = 10, and push
#    original and reconstructed through the solver. Measure the change in Cp."
#
# Four held-out shapes on a 2x2 (thickness, Reynolds) factorial, six bodies each:
#
#   orig        the shape as the dataset stores it
#   roundtrip   resampled to the 128-point theta grid and back, no truncation
#   pca01..10   PCA reconstruction at M = 1, 2, 3, 10
#
# 24 runs. The roundtrip control and M = 1, 2 are additions to the registered
# design -- see p1_b03_make_reconstructions.py for why, and the results file
# under "Protocol deviations".
#
# Every run uses the PROTOCOL section 6 standard settings, so the frozen F04
# sigmas apply directly:
#   RESOLUTION = max(100, round(0.2*Re))   MAX_PHYS_T = 50   AVG_WINDOW = 25
#
# Usage:
#   scripts/paper1/run_p1_b03_solver.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/cfd/ellipse2d/ellipse2d"
SHAPES="$ROOT/cfd/ellipse2d/shapes/b03_recon"
WORK="$ROOT/outputs/paper1/P1-B03_basis-reachability/solver"
LOG="$WORK/b03_solver.log"
MIN_FREE_GB=20

MAXT=50
AVG=25

mkdir -p "$WORK"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[ -x "$BIN" ]    || { say "ERROR: $BIN not built"; exit 1; }
[ -d "$SHAPES" ] || { say "ERROR: run p1_b03_make_reconstructions.py first"; exit 1; }

# The display watchdog kills long CUDA kernels intermittently -- at RES>~190 it
# took 2 of 6 runs on the first pass while the other 4 completed at identical
# settings. It is a race against whatever else is using the GPU, not a property
# of the case, so a retry is the correct response. Anything else is reported and
# left alone.
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
      say "    GPU-WATCHDOG on attempt $attempt"
      sleep 20
      continue
    fi
    say "    FAILED -- see run.log"
    return 0
  done
  say "    FAILED -- GPU-WATCHDOG after 3 attempts"
}

say "=========================================================="
say "P1-B03 solver half -- reconstruction tolerance"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

# Drive straight off the manifest so the runner cannot drift from the bodies.
tail -n +2 "$SHAPES/manifest.csv" | while IFS=, read -r label idx variant re h b n dev; do
  [ -z "${label:-}" ] && continue
  dat="$SHAPES/${label}_${variant}.dat"
  [ -f "$dat" ] || { say "  MISSING $dat"; continue; }
  rule=$(awk -v r="$re" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }')
  run "${label}_${variant}" "$dat" "$rule" "$re" "$h"
done

say "P1-B03 solver runs complete -- grade with scripts/paper1/p1_b03_grade_solver.py"
say "free disk: $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
