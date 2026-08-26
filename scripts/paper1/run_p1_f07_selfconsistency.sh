#!/bin/bash
# P1-F07 step 2 -- solver-internal normalisation self-consistency.
#
# Cp is dimensionless: at fixed Re it must be invariant to the choice of U, and
# invariant to rho outright. The case builds its converter with
# UnitConverterFromResolutionAndRelaxationTime, so the lattice evolution is set
# by (RESOLUTION, tau, chord) alone -- rescaling U at fixed Re rescales the
# physical viscosity in step and leaves the lattice problem identical.
#
# That is what makes this a clean test. Anything that moves in the reported Cp
# is a de-dimensionalisation error in the post-processing, not physics: most
# likely a mismatch between the PHYS_CHAR_DENSITY / PHYS_CHAR_VELOCITY used for
# qInf (results.h:380) and the values the converter applies to
# SuperLatticePhysPressure2D.
#
#   base     U = 1.0  rho = 1.0      the production setting
#   u2       U = 2.0  rho = 1.0      spec step 2
#   uhalf    U = 0.5  rho = 1.0      the other direction, to catch a one-sided bug
#   rho2     U = 1.0  rho = 2.0      spec candidate 9, "caught by step 2"
#
# Four runs of the six the spec budgets.
#
# Usage:
#   scripts/paper1/run_p1_f07_selfconsistency.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/cfd/ellipse2d/ellipse2d"
DAT="$ROOT/cfd/ellipse2d/shapes/eval_grid/sim_2824.dat"
WORK="$ROOT/outputs/paper1/P1-F07_normalisation-audit/selfconsistency"
LOG="$WORK/f07_step2.log"
MIN_FREE_GB=20

# F04's "mid" case -- the best-characterised point in the envelope.
RE=750.0
H=5.5
RES=150
MAXT=50
AVG=25

mkdir -p "$WORK"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[ -x "$BIN" ] || { say "ERROR: $BIN not built"; exit 1; }
[ -f "$DAT" ] || { say "ERROR: $DAT missing"; exit 1; }

run() {
  local tag="$1" u="$2" rho="$3"
  local d="$WORK/$tag"
  if ls "$d"/tmp/*_force.csv >/dev/null 2>&1; then say "  skip $tag (done)"; return 0; fi
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  [ "$free" -lt "$MIN_FREE_GB" ] && { say "ABORT: only ${free}G free"; exit 1; }

  local attempt
  for attempt in 1 2 3; do
    rm -rf "$d"; mkdir -p "$d"
    cp "$DAT" "$d/body.dat"
    [ "$attempt" -eq 1 ] \
      && say "  $tag (U=$u rho=$rho Re=$RE RES=$RES)" \
      || say "  $tag retry $attempt/3"
    ( cd "$d" && "$BIN" body.dat --REYNOLDS "$RE" --CHANNEL_HEIGHT "$H" \
        --RESOLUTION "$RES" --MAX_PHYS_T "$MAXT" --AVG_WINDOW_CHORDS "$AVG" \
        --PHYS_CHAR_VELOCITY "$u" --PHYS_CHAR_DENSITY "$rho" \
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
say "P1-F07 step 2 -- normalisation self-consistency"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

run base   1.0 1.0
run u2     2.0 1.0
run uhalf  0.5 1.0
run rho2   1.0 2.0

say "P1-F07 step 2 complete -- grade with scripts/paper1/p1_f07_audit.py"
