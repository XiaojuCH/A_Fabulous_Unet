import os
import json
import cv2
import numpy as np

# ==========================================
# 1. 路径配置 (踢掉表现极差的 U-Net)
# ==========================================
JSON_PATH = "data_splits/clean_full_list.json"
GT_ROOT_DIR = "../Unet/dataset"

DIRS = {
    "MedSAM (Expert)": "results/masks_medsam_gt",
    "SAM2-LoRA (Expert)": "results/masks_lora_gt",
    "SAM2-MSA (Expert)": "results/masks_msa_gt",
    "ST-SAM (Expert)": "results/masks_stsam_gt"
}

# 临床容忍阈值：误差超过 15 像素被视为“临床灾难性失效”
CLINICAL_TOLERANCE_PX = 15.0

def extract_mask_properties(mask_path):
    if not os.path.exists(mask_path): return None, None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.max(mask) == 0: return 0.0, 0.0
    binary_mask = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels < 2: return 0.0, 0.0
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    width = stats[largest_idx, cv2.CC_STAT_WIDTH]
    height = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    if width < 10: return 0.0, 0.0
    tmh = float(area / width)
    aspect_ratio = float(width / max(height, 1))
    return tmh, aspect_ratio

def main():
    print("🚀 正在计算核心临床安全性指标...")
    with open(JSON_PATH, 'r') as f:
        data_list = json.load(f)
        
    diffs = {name: [] for name in DIRS.keys()}
    valid_samples = 0
    
    for item in data_list:
        sample_id = item.get("id")
        raw_gt_rel_path = item.get("label")
        gt_rel_path = raw_gt_rel_path.replace("Label", "Cleaned_Label")
        gt_path = os.path.normpath(os.path.join(GT_ROOT_DIR, "..", gt_rel_path))
        
        gt_tmh, ar = extract_mask_properties(gt_path)
        if gt_tmh is None or gt_tmh == 0 or gt_tmh > 200: continue
            
        pred_tmhs = {}
        missing = False
        for name, dir_path in DIRS.items():
            p_path = os.path.join(dir_path, f"{sample_id}.png")
            if not os.path.exists(p_path):
                missing = True; break
            p_tmh, _ = extract_mask_properties(p_path)
            pred_tmhs[name] = p_tmh if p_tmh is not None else 0.0
            
        if missing: continue
            
        for name in DIRS.keys():
            error = pred_tmhs[name] - gt_tmh
            # 如果断裂导致几乎为0，或者发生极大假阳性
            if pred_tmhs[name] > 200 or pred_tmhs[name] == 0:
                error = 100.0 if pred_tmhs[name] > 200 else -gt_tmh
            diffs[name].append(error)
            
        valid_samples += 1

    print(f"\n✅ 分析完成！共计有效样本: {valid_samples}\n")
    print("="*105)
    print(f"{'Model':<15} | {'Median Error':<15} | {'Max Error (Px)':<15} | {'95% LoA Width (Px)':<20} | {'Failure Rate (>15px)':<20}")
    print("-" * 105)
    
    for name in DIRS.keys():
        arr = np.array(diffs[name])
        abs_arr = np.abs(arr)
        
        # 1. Median Error
        median_err = np.median(abs_arr)
        # 2. Max Error
        max_err = np.max(abs_arr)
        # 3. 95% LoA Width (1.96 * StdDev * 2)
        std_diff = np.std(arr)
        loa_width = 1.96 * std_diff * 2
        # 4. Catastrophic Failure Rate
        failure_rate = np.mean(abs_arr > CLINICAL_TOLERANCE_PX) * 100
        
        print(f"{name:<15} | {median_err:<15.2f} | {max_err:<15.2f} | {loa_width:<20.2f} | {failure_rate:.2f}%")
    print("="*105)
    print("* Failure Rate: Percentage of cases with absolute error > 15 pixels (Unacceptable Clinical Risk).")
    print("* 95% LoA Width: The spread of the 95% confidence interval (Smaller = More Reliable).")

if __name__ == "__main__":
    main()