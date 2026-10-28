#!/bin/bash
# P1-F04 -- grading noise floor.
#
# Holds the geometry fixed and varies only nuisance parameters that ought not to
# matter. The pooled sigmas become the significance thresholds for every
# subsequent experiment in the suite, so this is the cheapest thing in the plan
# that raises the paper's defensibility.
#
# Three shapes spanning the envelope (low-Re thin, mid, high-Re thick), and for
# each, one factor varied at a time from a common baseline:
#
#   RESOLUTION          rule, rule +/- one step   above the rule it should be converged
#   node phase          3 rolls of the point list same body, different node offset
#   AVG_WINDOW_CHORDS   15, 25, 35                time-averaging window
#   MAX_PHYS_T          40, 50, 60                statistical convergence
#
# 9 runs per shape (baseline + 8 variations), 27 total. VTK output is pruned as
# each run completes.
#
# Usage:
#   scripts/paper1/run_p1_f04.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/cfd/ellipse2d/ellipse2d"
SHAPES="$ROOT/cfd/ellipse2d/shapes/f04_variants"
WORK="$ROOT/outputs/paper1/P1-F04_noise-floor"
LOG="$WORK/f04.log"
MIN_FREE_GB=20

mkdir -p "$WORK"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[ -x "$BIN" ]    || { say "ERROR: $BIN not built"; exit 1; }
[ -d "$SHAPES" ] || { say "ERROR: run p1_f04_make_variants.py first"; exit 1; }

# label:Re:H  -- must match p1_f04_make_variants.py SHAPES
CASES=(
  "lowre_thin:502.0:5.989999771118164"
  "mid:750.0:5.5"
  "highre_thick:880.0:6.0"
)

# run <tag> <dat> <RES> <AVG> <MAXT> <Re> <H>
run() {
  local tag="$1" dat="$2" res="$3" avg="$4" maxt="$5" re="$6" h="$7"
  local d="$WORK/$tag"
  if ls "$d"/tmp/*_force.csv >/dev/null 2>&1; then say "  skip $tag (done)"; return 0; fi
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  [ "$free" -lt "$MIN_FREE_GB" ] && { say "ABORT: only ${free}G free"; exit 1; }

  rm -rf "$d"; mkdir -p "$d"
  cp "$dat" "$d/body.dat"
  say "  $tag (RES=$res AVG=$avg T=$maxt)"
  ( cd "$d" && "$BIN" body.dat --REYNOLDS "$re" --CHANNEL_HEIGHT "$h" \
      --RESOLUTION "$res" --MAX_PHYS_T "$maxt" --AVG_WINDOW_CHORDS "$avg" \
      > run.log 2>&1 )
  if ls "$d"/tmp/*_force.csv >/dev/null 2>&1; then
    say "    OK"
    "$(dirname "$0")/prune_run_output.sh" "$d" >> "$LOG" 2>&1
  else
    local why="see run.log"
    grep -qi "launch timed out" "$d/run.log" 2>/dev/null && why="GPU-WATCHDOG"
    say "    FAILED -- $why"
  fi
}

say "=========================================================="
say "P1-F04 noise floor"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

for entry in "${CASES[@]}"; do
  IFS=':' read -r label re h <<< "$entry"
  rule=$(awk -v r="$re" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }')
  step=$(awk -v b="$rule" 'BEGIN { printf "%d", b * 0.25 + 0.5 }')   # one step = 25%
  base="$SHAPES/${label}.dat"
  say "-- $label (Re=$re H=$h rule=$rule step=$step)"

  # Baseline, and the node-phase variants at that baseline.
  for k in 0 1 2; do
    run "${label}_phase${k}" "$SHAPES/${label}_phase${k}.dat" "$rule" 25 50 "$re" "$h"
  done
  # RESOLUTION, one step either side of the rule.
  run "${label}_res$((rule - step))" "$base" "$((rule - step))" 25 50 "$re" "$h"
  run "${label}_res$((rule + step))" "$base" "$((rule + step))" 25 50 "$re" "$h"
  # Averaging window.
  run "${label}_avg15" "$base" "$rule" 15 50 "$re" "$h"
  run "${label}_avg35" "$base" "$rule" 35 50 "$re" "$h"
  # Statistical convergence.
  run "${label}_t40" "$base" "$rule" 25 40 "$re" "$h"
  run "${label}_t60" "$base" "$rule" 25 60 "$re" "$h"
done

say "P1-F04 runs complete -- grade with scripts/paper1/p1_f04_grade.py"
say "free disk: $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
