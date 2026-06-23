#!/bin/bash
# EXP-042 follow-up. Two questions:
#  Q2 (roughness vs lattice fidelity): does under-resolving the boundary mask
#      the surface wobble? Run the synth raw_n240 (rough) and smooth_n240 at a
#      ladder of lattice resolutions; the raw-minus-smooth Cd gap should grow as
#      RES rises and the lattice starts to resolve the wobble.
#  Q1 (in-distribution creep): repeat the node-density check on an in-dist shape
#      (gen_4849, Re 960, H 5.63) to confirm node count / RES don't move it.
BIN=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/ellipse2d
SYNTH=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/shapes/node_sweep
IND=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/shapes/node_sweep_indist
run() { # shapefile tag Re H RES
  d=runs/$2; mkdir -p "$d"; rm -rf "$d/tmp" "$d/run.log"
  ( cd "$d" && "$BIN" "$1" \
      --REYNOLDS $3 --CHANNEL_HEIGHT $4 --RESOLUTION $5 \
      --MAX_PHYS_T 50 --AVG_WINDOW_CHORDS 25 > run.log 2>&1 )
  echo "$2 done: $(tail -1 "$d/tmp/ellipse2d_force.csv" 2>/dev/null)"
}

# Q2: synth roughness vs resolution (RES 160 raw/smooth + 240 raw already exist).
for R in 60 80 120; do
  run "$SYNTH/raw_n240.dat"    roughres_raw_n240_res$R    754 5.5 $R
  run "$SYNTH/smooth_n240.dat" roughres_smooth_n240_res$R 754 5.5 $R
done
run "$SYNTH/smooth_n240.dat" roughres_smooth_n240_res240 754 5.5 240

# Q1: in-distribution shape (gen_4849).
run "$IND/orig.dat"     nodeind_orig_res160     960 5.63 160
run "$IND/raw_n240.dat" nodeind_raw_n240_res160 960 5.63 160
run "$IND/raw_n240.dat" nodeind_raw_n240_res240 960 5.63 240
echo ALL_DONE
