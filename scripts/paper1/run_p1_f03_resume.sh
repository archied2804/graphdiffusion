#!/bin/bash
# P1-F03 -- resume after an interrupted run.
#
# `run_p1_f03.sh` is not restartable: it truncates its own log (`: > "$LOG"`)
# and `run_from_manifest.sh` does `rm -rf "$d/tmp"` per row, so re-running it
# would redo every completed case and erase the record of the first attempt.
#
# This script is idempotent. It re-runs only what is missing, judged from the
# run directories rather than from the log -- a power loss can drop the log's
# last buffered lines, so the log undercounts what actually finished.
#
# A case counts as complete when both result files exist and are non-empty:
#   <run>/tmp/ellipse2d_cp.csv  and  <run>/tmp/ellipse2d_force.csv
# A run killed mid-simulation has neither, but does leave gigabytes of VTK
# output, so incomplete directories are deleted before being re-run.
#
# Usage:
#   scripts/paper1/run_p1_f03_resume.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHAPES="$ROOT/cfd/ellipse2d/shapes/eval_grid"
RUNS="$ROOT/cfd/ellipse2d/runs"
RUNNER="$RUNS/run_from_manifest.sh"
WORK="$ROOT/outputs/paper1/P1-F03_solver-qualification"
STAGE="$WORK/stage_resume"

# Each Part B case at 2x the stability rule writes several GB of VTK. Abort
# rather than fill the disk and corrupt an otherwise good run.
MIN_FREE_GB=40

mkdir -p "$WORK" "$STAGE"
LOG="$WORK/f03.log"          # appended, never truncated
SENTINEL="$WORK/.f03_complete"
rm -f "$SENTINEL"            # cleared now, written only on a clean finish

# Retries per case. The GPU is display-attached, so the driver watchdog can
# kill a long kernel with cudaErrorLaunchTimeout regardless of the physics --
# runs have died at t=49 of 50. That is an infrastructure failure, not solver
# divergence, and must not be recorded as one.
MAX_TRIES=3
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

check_disk() {
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    say "ABORT: only ${free}G free, need ${MIN_FREE_GB}G"
    exit 1
  fi
}

# complete <run_name> -> 0 if both result files exist and are non-empty
complete() {
  local d="$RUNS/$1"
  [ -s "$d/tmp/ellipse2d_cp.csv" ] && [ -s "$d/tmp/ellipse2d_force.csv" ]
}

# classify <run_name> -> echoes why a run produced no results
classify() {
  local log="$RUNS/$1/run.log"
  local reached
  reached=$(grep -E "^\[getResults\]" "$log" 2>/dev/null | tail -1 |
            grep -oE "t=[0-9.]+" | head -1)
  if grep -qi "launch timed out" "$log" 2>/dev/null; then
    echo "GPU-WATCHDOG (cudaErrorLaunchTimeout) at ${reached:-t=?}"
  elif grep -qiE "Uncaught exception|nan|diverge" "$log" 2>/dev/null; then
    echo "SOLVER at ${reached:-t=?}: $(grep -i "Uncaught exception" "$log" |
         head -1 | cut -c1-60)"
  else
    echo "INCOMPLETE at ${reached:-t=?} (no exception -- killed externally?)"
  fi
}

# run_one <shapes_dir> <name> <res_override>
# Skips a complete case; clears a partial one first so no stale VTK survives.
# Retries transient GPU watchdog kills; a genuine solver failure is recorded
# after the first attempt, since retrying it would only waste time.
run_one() {
  local shapes="$1" name="$2" res="$3" try reason
  if complete "$name"; then
    say "  skip $name (already complete)"
    return 0
  fi
  for try in $(seq 1 "$MAX_TRIES"); do
    if [ -d "$RUNS/$name" ]; then
      rm -rf "${RUNS:?}/$name"
    fi
    check_disk
    say "  running $name (RES=${res:-auto}) attempt $try/$MAX_TRIES"
    "$RUNNER" "$shapes" "$res" 50 >> "$LOG" 2>&1
    if complete "$name"; then
      say "  $name OK"
      # Drop the VTK dumps now: ~99% of the bytes, none of it read downstream.
      "$(dirname "$0")/prune_run_output.sh" "$RUNS/$name" >> "$LOG" 2>&1
      return 0
    fi
    reason=$(classify "$name")
    say "  FAILED $name -- $reason"
    case "$reason" in
      GPU-WATCHDOG*) [ "$try" -lt "$MAX_TRIES" ] && say "  retrying" ;;
      *) say "  not retrying -- recorded as failed, not hidden"; return 1 ;;
    esac
  done
  say "  GAVE UP on $name after $MAX_TRIES watchdog kills"
  return 1
}

say "=========================================================="
say "P1-F03 RESUME after interruption"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

# --- Part A: envelope gate -------------------------------------------------
say "Part A -- envelope gate, 15 cases at RES = max(100, 0.2*Re)"
while IFS=',' read -r name idx re h rest; do
  [ "$name" = "name" ] && continue
  if complete "$name"; then
    say "  skip $name (already complete)"
    continue
  fi
  # One-row manifest so only this case is re-run; the shared runner would
  # otherwise clear and redo every row in the full manifest.
  printf 'name,idx,Re,H\n%s,%s,%s,%s\n' "$name" "$idx" "$re" "$h" > "$STAGE/manifest.csv"
  # Shape files are named sim_<idx>.dat; stage under the run name so the
  # runner resolves <name>.dat directly instead of falling back.
  cp "$SHAPES/sim_${idx}.dat" "$STAGE/${name}.dat" || { say "  ERROR: no sim_${idx}.dat"; continue; }
  run_one "$STAGE" "$name" ""
done < "$SHAPES/manifest.csv"
say "Part A done"

# Park Part A results. Only the result files and log are copied, not the VTK
# output -- the full directories are ~58G and Part B does not overwrite them
# anyway (its tags carry a _res<N> suffix), so a full copy would risk the disk
# for no benefit. Everything the grading script reads is here.
mkdir -p "$WORK/envelope"
while IFS=',' read -r name _rest; do
  [ "$name" = "name" ] && continue
  if complete "$name"; then
    mkdir -p "$WORK/envelope/$name/tmp"
    cp "$RUNS/$name/tmp/ellipse2d_cp.csv" "$WORK/envelope/$name/tmp/" 2>/dev/null
    cp "$RUNS/$name/tmp/ellipse2d_force.csv" "$WORK/envelope/$name/tmp/" 2>/dev/null
    cp "$RUNS/$name/run.log" "$WORK/envelope/$name/" 2>/dev/null
  fi
done < "$SHAPES/manifest.csv"
say "Part A results archived to $WORK/envelope"

# --- Part B: resolution sensitivity ----------------------------------------
CASES=("re0501_h50_idx0018" "re0751_h50_idx2833" "re0998_h50_idx5642")
say "Part B -- resolution sweep on ${CASES[*]}"

for name in "${CASES[@]}"; do
  row=$(grep "^$name," "$SHAPES/manifest.csv")
  idx=$(echo "$row" | cut -d, -f2)
  re=$(echo "$row" | cut -d, -f3)
  h=$(echo "$row" | cut -d, -f4)
  base=$(awk -v r="$re" 'BEGIN { v = int(0.2 * r + 0.5); if (v < 100) v = 100; print v }')

  for mult in 0.5 1.0 1.5 2.0; do
    res=$(awk -v b="$base" -v m="$mult" 'BEGIN { printf "%d", b * m + 0.5 }')
    tag="${name}_res${res}"
    printf 'name,idx,Re,H\n%s,%s,%s,%s\n' "$tag" "$idx" "$re" "$h" > "$STAGE/manifest.csv"
    cp "$SHAPES/sim_${idx}.dat" "$STAGE/${tag}.dat" \
      || { say "  ERROR: no sim_${idx}.dat"; continue; }
    say "  $tag (Re=$re, RES=$res, ${mult}x rule)"
    run_one "$STAGE" "$tag" "$res"
    if complete "$tag"; then
      mkdir -p "$WORK/resolution/$tag/tmp"
      cp "$RUNS/$tag/tmp/ellipse2d_cp.csv" "$WORK/resolution/$tag/tmp/" 2>/dev/null
      cp "$RUNS/$tag/tmp/ellipse2d_force.csv" "$WORK/resolution/$tag/tmp/" 2>/dev/null
      cp "$RUNS/$tag/run.log" "$WORK/resolution/$tag/" 2>/dev/null
    fi
  done
done

say "Part B done"
touch "$SENTINEL"
say "P1-F03 solver runs complete -- grade with scripts/paper1/p1_f03_grade.py"
say "free disk: $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
