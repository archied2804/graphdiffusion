#!/bin/bash
# P1-F05 — reference model under the clean protocol.
#
# Trains the EXP-022 recipe on the frozen P1-F01 partition across 5 seeds.
# The resulting checkpoints are frozen as P1-REF; every A/B/C/D/G/S experiment
# is a delta from these and must reuse these seeds.
#
# Spec: DGN_Docs/paper1/specs/P1-F05_reference-model.md
#
# Usage:
#   scripts/paper1/run_p1_f05.sh              # seeds 0-4, 1000 epochs
#   SEEDS="0 1" EPOCHS=50 scripts/paper1/run_p1_f05.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/paper1/P1-F05_reference.yaml}"
PARTITION="${PARTITION:-data/splits/paper1_partition.json}"
EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EXP_ID="P1-F05"
OUTROOT="outputs/paper1/P1-F05_reference"

[ -f "$PARTITION" ] || { echo "ERROR: run p1_f01_partition.py first — $PARTITION missing"; exit 1; }

mkdir -p "$OUTROOT"
echo "P1-F05 | config=$CONFIG epochs=$EPOCHS device=$DEVICE seeds=[$SEEDS]"
echo "started $(date -Is)"

for s in $SEEDS; do
  d="$OUTROOT/seed$s"
  if [ -f "$d/run_meta.json" ]; then
    echo "[seed $s] already complete, skipping (delete $d to force)"
    continue
  fi
  mkdir -p "$d"
  echo "[seed $s] training -> $d   $(date -Is)"
  python train.py \
    --config "$CONFIG" \
    --partition "$PARTITION" \
    --seed "$s" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --exp-id "$EXP_ID" \
    --output "$d/generated_shapes.png" \
    > "$d/train.log" 2>&1
  echo "[seed $s] done $(date -Is)  best_val=$(python -c "
import json
try:
    log = json.load(open('$d/loss_log.json'))
    v = [e['val_loss'] for e in log if e.get('val_loss') is not None]
    print(f'{min(v):.6f}' if v else 'n/a')
except Exception:
    print('n/a')
")"
done

echo "ALL_DONE $(date -Is)"
