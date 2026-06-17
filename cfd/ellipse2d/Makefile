EXAMPLE = ellipse2d

# This case dir is symlinked into $OPENLB_ROOT/examples/ellipse2d but lives
# physically elsewhere, so the conventional relative `OLB_ROOT := ../../..`
# breaks: GNU make resolves relative paths against the *physical* cwd, which is
# the real case dir, not the examples/ location. Allow OLB_ROOT to be supplied
# from the environment (reproducible in a job script:
#   make OLB_ROOT=$OPENLB_ROOT
# ), and fall back to the relative path for a normally-placed example.
OLB_ROOT ?= ../../..

include $(OLB_ROOT)/default.mk
