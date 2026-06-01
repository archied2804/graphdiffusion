#!/usr/bin/env bash
# Run EXP-022..025 sequentially, evaluate each, continue on per-experiment
# failures so one bad config doesn't kill the chain. Output:
#   outputs/EXP-NNN_*/loss_log.json, checkpoint_best.pt, checkpoint_ema.pt (if EMA),
#   roughness_report.json, figure_*.png, *.mp4/.gif
#   outputs/exp_series_run.log (overall log; this script also tees per-experiment logs)
#
# Usage:
#   bash scripts/run_exp_series.sh
#
# Override the experiment list:
#   EXPS="EXP-024_ema EXP-025_v_pred" bash scripts/run_exp_series.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda}"
EXPS="${EXPS:-EXP-022_T1000 EXP-023_min_snr EXP-024_ema EXP-025_v_pred}"

mkdir -p outputs

for exp in $EXPS; do
    cfg="configs/${exp}.yaml"
    out="outputs/${exp}"
    log="outputs/${exp}.log"

    if [[ ! -f "$cfg" ]]; then
        echo "[skip] $exp: config $cfg not found" | tee -a outputs/exp_series_run.log
        continue
    fi

    echo "================================================================="
    echo "[$(date -Iseconds)] Starting $exp"
    echo "  config:     $cfg"
    echo "  output dir: $out"
    echo "  log:        $log"
    echo "================================================================="

    mkdir -p "$out"

    # Training
    uv run python train.py \
        --config "$cfg" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --output "$out/generated_shapes.png" \
        2>&1 | tee "$log"
    train_rc=${PIPESTATUS[0]}

    if [[ "$train_rc" -ne 0 ]]; then
        echo "[fail] $exp: train.py exited $train_rc — skipping postprocess" | tee -a outputs/exp_series_run.log
        continue
    fi

    # Pick checkpoint: prefer EMA when present (EXP-024 + anything else with ema_decay).
    if [[ -f "$out/checkpoint_ema.pt" ]]; then
        ckpt="$out/checkpoint_ema.pt"
    else
        ckpt="$out/checkpoint_best.pt"
    fi

    # Evaluation
    uv run python scripts/postprocess_exp020.py \
        --experiment-dir "$out" \
        --config "$cfg" \
        --device "$DEVICE" \
        --n-samples 4 --n-targets 3 --target-seed 0 \
        --checkpoint "$ckpt" \
        2>&1 | tee -a "$log"
    pp_rc=${PIPESTATUS[0]}

    if [[ "$pp_rc" -ne 0 ]]; then
        echo "[fail] $exp: postprocess exited $pp_rc" | tee -a outputs/exp_series_run.log
    fi

    if [[ -f "$out/roughness_report.json" ]]; then
        overall=$(uv run python -c "import json; print(f\"{json.load(open('$out/roughness_report.json'))['overall_mean']:.5f}\")")
        echo "[$(date -Iseconds)] $exp done: overall roughness = $overall" \
            | tee -a outputs/exp_series_run.log
    fi
done

echo
echo "[$(date -Iseconds)] Series complete. Summary:"
for exp in $EXPS; do
    rep="outputs/${exp}/roughness_report.json"
    if [[ -f "$rep" ]]; then
        v=$(uv run python -c "import json; print(f\"{json.load(open('$rep'))['overall_mean']:.5f}\")")
        printf '  %-25s  %s\n' "$exp" "$v"
    else
        printf '  %-25s  (no report)\n' "$exp"
    fi
done | tee -a outputs/exp_series_run.log
