import os
import json
import cv2
import numpy as np

# ================= 配置路径 =================
JSON_PATH = "data_splits/clean_full_list.json"
GT_ROOT_DIR = "../Unet/dataset"
YOLO_MASK_DIR = "results/masks_stsam_yolo"
GT_MASK_DIR = "results/masks_stsam_gt"
IMG_SIZE = 1024

# 1. 加载所有 Fold 的 YOLO 预测框
yolo_boxes = {}
for i in range(5):
    ypath = f"data_splits/yolo_boxes_fold{i}.json"
    if os.path.exists(ypath):
        with open(ypath, 'r') as f:
            yolo_boxes.update(json.load(f))

# ================= 辅助函数 =================
def get_box_iou(box1, box2):
    """计算两个框的 IoU"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection / float(area1 + area2 - intersection + 1e-6)
    return iou

def extract_gt_box_and_tmh(mask_path):
    if not os.path.exists(mask_path): return None, None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.max(mask) == 0: return None, None
    
    binary_mask = (mask > 127).astype(np.uint8)
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0: return None, None
    
    gt_box = [xs.min(), ys.min(), xs.max(), ys.max()]
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels < 2: return gt_box, 0.0
    
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    width = stats[largest_idx, cv2.CC_STAT_WIDTH]
    if width < 10: return gt_box, 0.0
    
    return gt_box, float(area / width)

def extract_pred_tmh(mask_path):
    if not os.path.exists(mask_path): return 0.0, True, False
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.max(mask) == 0: return 0.0, True, False
    
    binary_mask = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels < 2: return 0.0, True, False
    
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    width = stats[largest_idx, cv2.CC_STAT_WIDTH]
    if width < 10: return 0.0, True, False
    
    tmh = float(area / width)
    return tmh, False, (tmh > 200)

# ================= 主程序 =================
def main():
    with open(JSON_PATH, 'r') as f:
        data_list = json.load(f)
        
    failed_cases = []
    
    print("🚀 正在扫描 ST-SAM 全自动模式的灾难性失效样本...")
    
    for item in data_list:
        sample_id = item.get("id")
        raw_gt_rel_path = item.get("label")
        gt_rel_path = raw_gt_rel_path.replace("Label", "Cleaned_Label")
        gt_path = os.path.normpath(os.path.join(GT_ROOT_DIR, "..", gt_rel_path))
        
        gt_box, gt_tmh = extract_gt_box_and_tmh(gt_path)
        if gt_tmh is None or gt_tmh == 0 or gt_tmh > 200: continue
            
        # 提取 YOLO 框
        yolo_box_norm = yolo_boxes.get(sample_id, [0,0,1,1])
        yolo_box = [yolo_box_norm[0]*IMG_SIZE, yolo_box_norm[1]*IMG_SIZE, 
                    yolo_box_norm[2]*IMG_SIZE, yolo_box_norm[3]*IMG_SIZE]
        
        # 计算 Box IoU
        box_iou = get_box_iou(gt_box, yolo_box)
        
        # 提取预测结果
        yolo_pred_path = os.path.join(YOLO_MASK_DIR, f"{sample_id}.png")
        yolo_tmh, is_empty, is_exploded = extract_pred_tmh(yolo_pred_path)
        
        # 计算误差
        error = abs(yolo_tmh - gt_tmh)
        if is_empty or is_exploded: error = 100.0
            
        # 筛选灾难性失效 (> 15 px)
        if error > 15.0:
            # 看看给专家框的话，它能救回来吗？
            gt_pred_path = os.path.join(GT_MASK_DIR, f"{sample_id}.png")
            expert_tmh, _, _ = extract_pred_tmh(gt_pred_path)
            expert_error = abs(expert_tmh - gt_tmh) if expert_tmh > 0 else 100.0
            
            failed_cases.append({
                "id": sample_id,
                "error": error,
                "is_empty": is_empty,
                "is_exploded": is_exploded,
                "box_iou": box_iou,
                "expert_error": expert_error
            })

    # ================= 打印诊断报告 =================
    print(f"\n✅ 扫描完毕。共发现 {len(failed_cases)} 例灾难性失效。")
    print("-" * 110)
    print(f"{'Sample ID':<25} | {'YOLO Error':<12} | {'YOLO-GT Box IoU':<16} | {'Mask Status':<15} | {'Expert Error (GT Box)':<20}")
    print("-" * 110)
    
    empty_count = 0
    low_iou_count = 0
    saved_by_expert = 0
    
    for case in failed_cases:
        status = "Empty (Black)" if case['is_empty'] else "Exploded" if case['is_exploded'] else "Severely Deviated"
        if case['is_empty']: empty_count += 1
        if case['box_iou'] < 0.6: low_iou_count += 1
        if case['expert_error'] < 15.0: saved_by_expert += 1
            
        print(f"{case['id']:<25} | {case['error']:<12.2f} | {case['box_iou']:<16.4f} | {status:<15} | {case['expert_error']:<20.2f}")
    
    print("-" * 110)
    print("📊 【诊断结论】")
    print(f"1. 预测为空 (Empty Mask) 的比例: {empty_count}/{len(failed_cases)}")
    print(f"2. YOLO框存在严重偏差 (IoU < 0.6) 的比例: {low_iou_count}/{len(failed_cases)}")
    print(f"3. 切换专家模式(GT框)后，成功救回(Error < 15)的比例: {saved_by_expert}/{len(failed_cases)}")

if __name__ == "__main__":
    main()