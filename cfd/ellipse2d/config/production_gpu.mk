# OpenLB build configuration for every Paper 1 production solver run.
#
# Copy to $OPENLB_ROOT/config.mk before building:
#
#   cp cfd/ellipse2d/config/production_gpu.mk $OPENLB_ROOT/config.mk
#   cd $OPENLB_ROOT && make clean && make
#   cd $OPENLB_ROOT/examples/ellipse2d && rm -f ellipse2d.o && make OLB_ROOT=$OPENLB_ROOT
#
# The `rm -f ellipse2d.o` is not optional: make relinks a stale object without
# recompiling it, which is how an earlier "fp64" sweep silently stayed fp32.
# Verify from the run banner, which prints Platform and Parallel, and from the
# compile line, which prints -DDEFAULT_FLOATING_POINT_TYPE.
#
# PARALLEL_MODE is NONE, not MPI. The case is a single-GPU run; the run banner
# for every P1-F03/F04/F06 result reads "Parallel : None". Building with MPI
# changes the Cp gather path in results.h (MPI_SUM reduce vs straight copy), so
# it is not an interchangeable choice.
#
# CUDA_ARCH 120 targets the Blackwell laptop GPU this was developed on
# (compute capability 12.0). Change it for other hardware; nothing else here is
# machine-specific.
#
# fp32 is the production setting. P1-F06 established that fp64 changes nothing
# for drag or pressure but is required for lift under grid refinement (fp32
# scatters lift by 29.5% at N=80 where fp64 holds 0.1%). Runs whose graded
# quantity is Cl must be rebuilt with FLOATING_POINT_TYPE := double and said so
# in their results file.

CXX             := nvcc
CC              := nvcc
CXXFLAGS        := -O3
CXXFLAGS        += -std=c++20 --forward-unknown-to-host-compiler
PARALLEL_MODE   := NONE
PLATFORMS       := CPU_SISD GPU_CUDA
CUDA_ARCH       := 120
FLOATING_POINT_TYPE := float
USE_EMBEDDED_DEPENDENCIES := ON
