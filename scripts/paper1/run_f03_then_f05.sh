#!/bin/bash
# Wait for the P1-F03 solver runs to finish, then start P1-F05 attempt 3.
#
# Both workloads use the GPU -- the OpenLB solver runs at ~91% utilisation, so
# training alongside it would slow both substantially. This serialises them so
# the pair can be left unattended.
#
# F03 is assumed to be already running. If it is not, its completion marker is
# checked first and training starts immediately.
#
# Usage:
#   nohup scripts/paper1/run_f03_then_f05.sh > outputs/paper1/chain.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
F03LOG="outputs/paper1/P1-F03_solver-qualification/f03.log"
MARKER="P1-F03 solver runs complete"

say() { echo "[$(date -Is)] $*"; }

say "chain: waiting for P1-F03 to finish"
while pgrep -f "run_p1_f03_resume.sh" > /dev/null 2>&1; do
  sleep 120
done

# The script exiting is not proof it succeeded -- it may have hit the disk
# guard. Check for the completion marker before committing 7.5 h of training.
if ! grep -q "$MARKER" "$F03LOG" 2>/dev/null; then
  say "ABORT: P1-F03 exited without its completion marker."
  say "       Inspect $F03LOG before starting F05."
  exit 1
fi

n_done=$(ls -1d cfd/ellipse2d/runs/re[0-9]*/ 2>/dev/null | wc -l)
say "P1-F03 complete ($n_done run dirs). Starting P1-F05 attempt 3."
say "conditioning: cond_norm=forward (P1-D01 §6.1)"

exec bash scripts/paper1/run_p1_f05.sh
