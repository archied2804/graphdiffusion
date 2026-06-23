#!/bin/bash
set -e
run() { # idx Re H b
  d=runs/sim_$1; mkdir -p $d; rm -rf $d/tmp $d/run.log
  ( cd $d && mpirun -np 12 /home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/ellipse2d /home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d/shapes/baseline/sim_$1.dat --REYNOLDS $2 --CHANNEL_HEIGHT $3 --RESOLUTION 100 --MAX_PHYS_T 50 --AVG_WINDOW_CHORDS 25 --SEMI_MINOR_B $4 > run.log 2>&1 )
  echo "sim $1 done: $(cat $d/tmp/ellipse2d_force.csv | tail -1)"
}
run 0 500 5.3 0.2555
run 2850 754 5.8 0.3915
run 5700 1000 5.78 0.254
echo ALL_DONE
