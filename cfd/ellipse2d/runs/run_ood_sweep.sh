#!/bin/bash
# EXP-041 OOD spread: run the family of synthetic out-of-distribution targets
# (sin amplitude/sign/frequency/tilt perturbations of the dataset-mean Cp) at
# fixed conditions (Re 754, H 5.5, RES 160) so the target morphology is the only
# variable. Maps measured-Cp fidelity against each target's OOD distance.
BIN=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/ellipse2d
SHAPES=/home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/shapes/ood_sweep
run() { # tag
  d=runs/ood_$1; mkdir -p "$d"; rm -rf "$d/tmp" "$d/run.log"
  ( cd "$d" && "$BIN" "$SHAPES/ood_$1.dat" \
      --REYNOLDS 754 --CHANNEL_HEIGHT 5.5 --RESOLUTION 160 \
      --MAX_PHYS_T 50 --AVG_WINDOW_CHORDS 25 > run.log 2>&1 )
  echo "ood_$1 done: $(tail -1 "$d/tmp/ellipse2d_force.csv" 2>/dev/null)"
}
for tag in sin_p3 sin_p5 sin_m3 sin2_p3 tilt_p4; do run $tag; done
echo ALL_DONE
