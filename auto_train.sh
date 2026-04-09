#!/usr/bin/env bash
set -euo pipefail

MAX_ITER=10
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# SAM_Baseline per-fold values (Overall, fold 0-4)
BASELINE_DICE="0.8611,0.8748,0.8129,0.8244,0.8447"
BASELINE_IOU="0.7634,0.7828,0.6933,0.7125,0.7374"
BASELINE_PREC="0.8621,0.8801,0.7990,0.8320,0.8284"
BASELINE_RECALL="0.8754,0.8817,0.8475,0.8369,0.8739"
BASELINE_HD95="36.79,41.59,42.80,45.27,39.53"
BASELINE_ASD="5.78,4.99,8.13,7.67,5.26"

# SAM_Gal_PlanC16 per-fold values (Overall, fold 0-4) — reference target
PLANC16_DICE="0.8732,0.8853,0.8347,0.8421,0.8624"
PLANC16_IOU="0.7822,0.7994,0.7249,0.7384,0.7644"
PLANC16_PREC="0.8831,0.8899,0.8229,0.8541,0.8548"
PLANC16_RECALL="0.8784,0.8931,0.8641,0.8492,0.8809"
PLANC16_HD95="36.21,41.20,42.08,44.58,38.47"
PLANC16_ASD="5.47,4.38,7.75,7.03,4.49"

cleanup() {
    :
}
trap cleanup EXIT

for iter in $(seq 2 $MAX_ITER); do
    RUN_DIR="checkpoints_run${iter}"
    echo ""
    echo "=========================================="
    echo "  Iteration ${iter} / ${MAX_ITER}"
    echo "=========================================="

    # 1. Create checkpoint dirs
    for fold in 0 1 2 3 4; do
        mkdir -p "${RUN_DIR}/fold_${fold}"
    done

    # 2. Launch all 5 folds in parallel
    PIDS=()
    for fold in 0 1 2 3 4; do
        port=$((29500 + fold))
        CKPT_DIR="./${RUN_DIR}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup torchrun \
            --nproc_per_node=8 \
            --master_port=${port} \
            src/train.py --fold ${fold} \
            > "${RESULTS_DIR}/run${iter}_fold${fold}.log" 2>&1 &
        PIDS+=($!)
        echo "  Fold ${fold} started (PID $!, port ${port})"
    done

    # 3. Wait for all folds to finish
    echo "  Waiting for all folds to complete..."
    failed=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: A fold process (PID $pid) exited with error."
            failed=1
        fi
    done
    if [ $failed -eq 1 ]; then
        echo "  One or more folds failed. Skipping evaluation for this iteration."
        continue
    fi
    echo "  All folds complete."

    # 4. Evaluate
    METRICS_FILE="${RESULTS_DIR}/run${iter}_metrics.txt"
    echo "  Running evaluation..."
    CKPT_DIR="./${RUN_DIR}" python3 get_final_table_v2.py 2>&1 | tee "$METRICS_FILE"

    # 5. Significance test (inline Python)
    STARS_FILE="${RESULTS_DIR}/run${iter}_stars.txt"
    python3 - <<PYEOF | tee "$STARS_FILE"
import sys
from scipy import stats
import numpy as np

# Baseline values (data_sam from cal_stars.py)
baseline = {
    'Dice':   [${BASELINE_DICE}],
    'IoU':    [${BASELINE_IOU}],
    'Prec':   [${BASELINE_PREC}],
    'Recall': [${BASELINE_RECALL}],
    'HD95':   [${BASELINE_HD95}],
    'ASD':    [${BASELINE_ASD}],
}
planc16 = {
    'Dice':   [${PLANC16_DICE}],
    'IoU':    [${PLANC16_IOU}],
    'Prec':   [${PLANC16_PREC}],
    'Recall': [${PLANC16_RECALL}],
    'HD95':   [${PLANC16_HD95}],
    'ASD':    [${PLANC16_ASD}],
}

# Parse per-fold Overall values (average of Colour+Infrared rows per fold)
# Output format per fold:
#   Fold N Results:
#   Colo     | dice | iou | recall | prec | hd95 | asd
#   Infr     | dice | iou | recall | prec | hd95 | asd
with open("${METRICS_FILE}") as f:
    lines = f.readlines()

# fold_rows[fold] = list of [dice,iou,recall,prec,hd95,asd] rows (one per modality)
fold_rows = {}
cur_fold = -1
for line in lines:
    if line.startswith("Fold ") and "Results:" in line:
        cur_fold = int(line.split()[1])
        fold_rows[cur_fold] = []
    elif "|" in line and cur_fold >= 0:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 7 and parts[0][:4] in ("Colo", "Infr"):
            try:
                fold_rows[cur_fold].append([float(parts[i]) for i in range(1, 7)])
            except ValueError:
                pass

# Average modalities per fold → Overall per-fold
new_data = {'Dice': [], 'IoU': [], 'Recall': [], 'Prec': [], 'HD95': [], 'ASD': []}
for fold in range(5):
    rows = fold_rows.get(fold, [])
    if not rows:
        continue
    avg = np.mean(rows, axis=0)  # [dice, iou, recall, prec, hd95, asd]
    new_data['Dice'].append(avg[0])
    new_data['IoU'].append(avg[1])
    new_data['Recall'].append(avg[2])
    new_data['Prec'].append(avg[3])
    new_data['HD95'].append(avg[4])
    new_data['ASD'].append(avg[5])

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

print("=== vs SAM_Baseline (must all be significant) ===")
all_sig = True
for metric in ['Dice', 'IoU', 'Recall', 'Prec', 'HD95', 'ASD']:
    nd = new_data.get(metric, [])
    bl = baseline[metric]
    if len(nd) != 5:
        print(f"  {metric:<8}: MISSING ({len(nd)} values found)")
        all_sig = False
        continue
    _, p = stats.ttest_rel(nd, bl)
    s = stars(p)
    direction = "better" if np.mean(nd) > np.mean(bl) else "worse"
    if metric in ('HD95', 'ASD'):
        direction = "better" if np.mean(nd) < np.mean(bl) else "worse"
    print(f"  {metric:<8}: p={p:.5f}  {s}  ({direction})")
    if s == "ns":
        all_sig = False

print()
print("=== vs PlanC16 (reference, informational) ===")
planc16_all_sig = True
for metric in ['Dice', 'IoU', 'Recall', 'Prec', 'HD95', 'ASD']:
    nd = new_data.get(metric, [])
    pc = planc16[metric]
    if len(nd) != 5:
        continue
    _, p = stats.ttest_rel(nd, pc)
    s = stars(p)
    diff = np.mean(nd) - np.mean(pc)
    if metric in ('HD95', 'ASD'):
        diff = -diff
    trend = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
    print(f"  {metric:<8}: p={p:.5f}  {s}  (vs PlanC16: {trend})")
    if s == "ns":
        planc16_all_sig = False

print()
if all_sig and planc16_all_sig:
    print("RESULT: SUCCESS - All metrics significant vs baseline AND PlanC16!")
elif all_sig:
    print("RESULT: PARTIAL - Significant vs baseline, but not vs PlanC16.")
else:
    print("RESULT: NOT_SIGNIFICANT - Some metrics are ns vs baseline.")
PYEOF

    # 6. Append to summary
    {
        echo "========== Run ${iter} =========="
        echo "--- Metrics ---"
        cat "$METRICS_FILE"
        echo "--- Stars ---"
        cat "$STARS_FILE"
        echo ""
    } >> "${RESULTS_DIR}/summary.txt"

    # 7. Check success
    if grep -q "^RESULT: SUCCESS" "$STARS_FILE"; then
        echo ""
        echo "SUCCESS on iteration ${iter}! Checkpoints saved in ${RUN_DIR}/"
        exit 0
    fi

    echo "  Not significant yet. Moving to next iteration..."
done

echo ""
echo "DONE: Did not achieve full significance after ${MAX_ITER} iterations."
echo "Check results/summary.txt for all run results."
exit 1
