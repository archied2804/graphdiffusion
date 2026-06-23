#!/bin/bash
# EXP-041 DPS sweep: run the 4 posterior-guided synthetic-OOD shapes through
# OpenLB GPU at the same conditions as the gen_synth baseline (Re 754, RES 160,
# H 5.5). Same target_cp.csv across all weights; only DPS guidance differs.
BIN=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/ellipse2d
SHAPES=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/shapes/dps_sweep
run() { # tag
  d=runs/dps_$1; mkdir -p "$d"; rm -rf "$d/tmp" "$d/run.log"
  ( cd "$d" && "$BIN" "$SHAPES/synth_$1.dat" \
      --REYNOLDS 754 --CHANNEL_HEIGHT 5.5 --RESOLUTION 160 \
      --MAX_PHYS_T 50 --AVG_WINDOW_CHORDS 25 > run.log 2>&1 )
  echo "dps_$1 done: $(tail -1 "$d/tmp/ellipse2d_force.csv" 2>/dev/null)"
}
run dps0
run dps0p3
run dps0p5
run dps1
echo ALL_DONE
