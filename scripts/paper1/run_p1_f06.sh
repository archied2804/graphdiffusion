#!/bin/bash
# P1-F06 -- Schaefer-Turek absolute benchmark.
#
# Three levels, per the spec. The differences between them are the diagnostic:
#
#   L0  stock OpenLB cylinder2d, unmodified      -> the build and the library
#   L1  benchmark geometry through ellipse2d,
#       parabolic inlet + no-slip walls          -> THIS case setup
#   L2  same, but the uniform inlet ellipse2d
#       normally uses                            -> what that simplification costs
#
# L0 -> L1 separates "OpenLB is fine, my case is wrong" from "the library
# behaves differently than I assume". L1 -> L2 quantifies the uniform-inlet
# choice recorded at lattice.h:15, for the first time.
#
# Each level runs a resolution ladder, because a benchmark value hit at one grid
# proves nothing -- this is the one place in the suite where a resolution study
# is genuinely required.
#
# Usage:
#   scripts/paper1/run_p1_f06.sh            # 2D-1 (Re=20, steady)
#   CONFIG=2d2 scripts/paper1/run_p1_f06.sh # 2D-2 (Re=100, unsteady)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/cfd/ellipse2d/ellipse2d"
DAT="$ROOT/cfd/ellipse2d/shapes/schafer_turek/cylinder_d0p1.dat"
OLB_FLOAT="/home/m22729ad/Software/OpenLB-1.9-0-cuda/olb-release"
OLB_DOUBLE="/home/m22729ad/Software/OpenLB-1.9-0-double/olb-release"

CONFIG="${CONFIG:-2d1}"
RESOLUTIONS="${RESOLUTIONS:-20 40 80}"
WORK="$ROOT/outputs/paper1/P1-F06_schafer-turek"
LOG="$WORK/f06_${CONFIG}.log"

mkdir -p "$WORK"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# Benchmark operating point. U_mean and MAX_PHYS_T differ between the two
# configurations; nu = U_mean*D/Re is derived by the case from these.
if [ "$CONFIG" = "2d1" ]; then
  RE=20;  UMEAN=0.2; MAXT=16
else
  RE=100; UMEAN=1.0; MAXT=32
fi

# Geometry, from p1_f06_make_cylinder.py. BODY_Y_OFFSET reproduces the
# benchmark's deliberate 0.005 asymmetry about the channel centreline.
GEOM=(--CHANNEL_HEIGHT 0.41 --BODY_Y_OFFSET -0.005
      --UPSTREAM_CHORDS 1.5 --DOWNSTREAM_CHORDS 19.5
      --SPONGE_CHORDS 0)

[ -f "$DAT" ] || { say "ERROR: $DAT missing -- run p1_f06_make_cylinder.py"; exit 1; }
[ -x "$BIN" ] || { say "ERROR: $BIN not built"; exit 1; }

say "=========================================================="
say "P1-F06 $CONFIG : Re=$RE U_mean=$UMEAN MAX_PHYS_T=$MAXT"
say "git: $(git -C "$ROOT" rev-parse --short HEAD)"

# ---- L0: stock cylinder2d, both precisions --------------------------------
for prec in float double; do
  root=$OLB_FLOAT; [ "$prec" = double ] && root=$OLB_DOUBLE
  stock="$root/examples/laminar/cylinder2d/cylinder2d"
  [ -x "$stock" ] || { say "L0/$prec: not built, skipping"; continue; }
  for N in $RESOLUTIONS; do
    d="$WORK/L0_${prec}_${CONFIG}_N${N}"
    if [ -s "$d/run.log" ] && grep -q "Timer.*measured time" "$d/run.log"; then
      say "  skip L0/$prec N=$N (done)"; continue
    fi
    mkdir -p "$d"; say "  L0/$prec N=$N"
    ( cd "$d" && "$stock" --RESOLUTION "$N" --REYNOLDS "$RE" > run.log 2>&1 )
  done
done

# ---- L1 (parabolic inlet) and L2 (uniform inlet) --------------------------
for level in L1 L2; do
  if [ "$level" = "L1" ]; then INLET=1; WALL=1; else INLET=0; WALL=0; fi
  for N in $RESOLUTIONS; do
    d="$WORK/${level}_${CONFIG}_N${N}"
    if [ -s "$d/tmp/ellipse2d_force.csv" ]; then say "  skip $level N=$N (done)"; continue; fi
    rm -rf "$d"; mkdir -p "$d"
    cp "$DAT" "$d/cylinder.dat"
    say "  $level N=$N (INLET_PROFILE=$INLET WALL_NOSLIP=$WALL)"
    ( cd "$d" && "$BIN" cylinder.dat \
        --REYNOLDS "$RE" --PHYS_CHAR_VELOCITY "$UMEAN" \
        --RESOLUTION "$N" --MAX_PHYS_T "$MAXT" \
        --INLET_PROFILE "$INLET" --WALL_NOSLIP "$WALL" \
        "${GEOM[@]}" > run.log 2>&1 )
    if [ -s "$d/tmp/ellipse2d_force.csv" ]; then
      say "    OK"
      "$(dirname "$0")/prune_run_output.sh" "$d" >> "$LOG" 2>&1
    else
      say "    FAILED: $(grep -iE 'exception|error' "$d/run.log" | head -1 | cut -c1-80)"
    fi
  done
done

say "P1-F06 $CONFIG runs complete -- grade with scripts/paper1/p1_f06_grade.py"
