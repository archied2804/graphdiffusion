#!/bin/bash
# EXP-042 node-density / magnitude-creep sweep. Same generated synth_ood
# geometry resampled to increasing node counts (periodic cubic spline), run at
# fixed lattice RES 160 (Re 754, H 5.5). A final RES-240 control on raw_n240
# separates boundary discretisation from lattice resolution.
BIN=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/ellipse2d
SHAPES=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/shapes/node_sweep
run() { # tag resolution
  d=runs/node_$1_res$2; mkdir -p "$d"; rm -rf "$d/tmp" "$d/run.log"
  ( cd "$d" && "$BIN" "$SHAPES/$1.dat" \
      --REYNOLDS 754 --CHANNEL_HEIGHT 5.5 --RESOLUTION $2 \
      --MAX_PHYS_T 50 --AVG_WINDOW_CHORDS 25 > run.log 2>&1 )
  echo "node_$1_res$2 done: $(tail -1 "$d/tmp/ellipse2d_force.csv" 2>/dev/null)"
}
run orig 160
run raw_n120 160
run raw_n240 160
run raw_n480 160
run smooth_n240 160
run raw_n240 240
echo ALL_DONE
