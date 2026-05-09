import os
import json
import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.metrics import compute_dice, compute_hausdorff_distance
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from monai.networks.nets import SwinUNETR


import sys
import argparse
sys.path.append("src") # 确保能找到 dataset 和 model

# 引入你的模型和数据集
from model import ST_SAM, Baseline_SAM2
from dataset import TearDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
MAX_HD95 = math.sqrt(IMG_SIZE**2 + IMG_SIZE**2)

# ==================== 模型加载辅助 ====================
def get_deeplab_p():
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    m = deeplabv3_resnet50(weights=weights)
    m.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    m.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return m

def get_swinunet():
    return SwinUNETR(in_channels=3, out_channels=1, feature_size=48, spatial_dims=2, use_v2=True, window_size=8)

def load_weights(model, path):
    if not os.path.exists(path): return None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
    model.eval()
    return model

# ==================== 核心逻辑 ====================
def is_extreme_hard_case(lbl_tensor, threshold=8.0):
    """筛选真正的地狱级样本：极端细长"""
    lbl_np = lbl_tensor.squeeze().cpu().numpy()
    y_indices, x_indices = np.where(lbl_np > 0)
    if len(y_indices) < 50: return False # 太小的噪点不算
    
    h = np.max(y_indices) - np.min(y_indices)
    w = np.max(x_indices) - np.min(x_indices)
    if h == 0 or w == 0: return False
    
    aspect_ratio = max(h, w) / min(h, w)
    return aspect_ratio > threshold

def compute_robust_metrics(pred_tensor, gt_tensor):
    """严谨计算 Dice 和 HD95，防止出现 NaN"""
    pred = (pred_tensor > 0.5).float()
    lbl = (gt_tensor > 0.5).float()
    
    # 1. Dice
    if lbl.sum() == 0 and pred.sum() == 0:
        dice = 1.0
    else:
        dice = compute_dice(pred.unsqueeze(0), lbl.unsqueeze(0), include_background=False).item()
        if math.isnan(dice): dice = 0.0
        
    # 2. HD95
    if lbl.sum() > 0 and pred.sum() > 0:
        hd95 = compute_hausdorff_distance(pred.unsqueeze(0), lbl.unsqueeze(0), include_background=False, percentile=95).item()
        if math.isnan(hd95): hd95 = MAX_HD95
    elif lbl.sum() > 0 and pred.sum() == 0:
        hd95 = MAX_HD95 
    else:
        hd95 = 0.0 if pred.sum() == 0 else MAX_HD95
        
    return dice, hd95

def main():
    print("🚀 启动 [极端困难样本] 5-Fold 全量评估")
    print("🎯 目标：寻找长宽比 > 8.0 的易断裂拓扑结构，重点对比 HD95 指标\n")
    
    # 存储 5 个 Fold 的所有困难样本结果
    global_results = {
        "ST-SAM (Ours)": {"dice": [], "hd95": []},
        "Baseline SAM2": {"dice": [], "hd95": []},
        "DeepLabV3+":    {"dice": [], "hd95": []},
        "Swin-UNETR":    {"dice": [], "hd95": []}
    }
    
    total_hard_cases = 0

    for fold in range(5):
        print(f"🔄 正在处理 Fold {fold} ...")
        json_path = f"./data_splits/fold_{fold}.json"
        if not os.path.exists(json_path): continue
            
        with open(json_path, 'r') as f: data = json.load(f)
        dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE, yolo_pred_json=f"./data_splits/yolo_boxes_fold{fold}.json")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 加载四个模型
        models = {
            "ST-SAM (Ours)": load_weights(ST_SAM().to(DEVICE), f"./checkpoints_run1/fold_{fold}/best_model.pth"),
            "Baseline SAM2": load_weights(Baseline_SAM2().to(DEVICE), f"./checkpoints_ablation/fold_{fold}/best_model.pth"),
            "DeepLabV3+":    load_weights(get_deeplab_p().to(DEVICE), f"./checkpoints_New_baseline/deeplab_p/fold_{fold}/best_model.pth"),
            "Swin-UNETR":    load_weights(get_swinunet().to(DEVICE), f"./checkpoints_New_baseline/swinunet/fold_{fold}/best_model.pth")
        }
        
        # 剔除加载失败的模型
        models = {k: v for k, v in models.items() if v is not None}
        if not models: continue

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Fold {fold} Hard Cases", leave=False):
                img, lbl, box = batch['image'].to(DEVICE), batch['label'].to(DEVICE), batch['box'].to(DEVICE)
                
                # 🔥 过滤极端困难样本 (长宽比 > 8.0)
                if not is_extreme_hard_case(lbl, threshold=8.0): continue
                total_hard_cases += 1
                
                for name, model in models.items():
                    if "SAM" in name:
                        logits = model(img, box)
                        pred = torch.sigmoid(logits)
                    else:
                        # 1. 正常的 CNN 推理
                        if "DeepLab" in name:
                            logits = model(img)['out']
                        else:
                            logits = model(img)
                        
                        pred = torch.sigmoid(logits)
                        
                        # 2. 引入 YOLO 空间先验：强行抹除 YOLO 框外的假阳性
                        box_tensor = box[0]
                        padding = 20
                        x1, y1, x2, y2 = [int(v.item()) for v in box_tensor]
                        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                        x2, y2 = min(IMG_SIZE, x2 + padding), min(IMG_SIZE, y2 + padding)
                        
                        # 确保是一个合法的检测框
                        if x2 > x1 and y2 > y1 and (x2-x1) >= 10 and (y2-y1) >= 10:
                            spatial_mask = torch.zeros_like(pred)
                            spatial_mask[:, :, y1:y2, x1:x2] = 1.0 
                            pred = pred * spatial_mask 
                            
                    # 3. 计算严谨的指标
                    dice, hd95 = compute_robust_metrics(pred.cpu(), lbl.cpu())
                    
                    global_results[name]["dice"].append(dice)
                    global_results[name]["hd95"].append(hd95)

    # ==================== 打印震撼的结果 ====================
    print("\n" + "═"*75)
    print(f"🎯 最终在 5-Fold 中共筛出极端细长样本数量: {total_hard_cases}")
    print("═"*75)
    print(f"{'模型名称':<16} | {'Dice (↑ 越高越好)':<20} | {'HD95 (↓ 越低越好)':<20}")
    print("─"*75)
    
    for name in global_results.keys():
        if len(global_results[name]["dice"]) == 0: continue
        
        mean_dice = np.mean(global_results[name]["dice"])
        std_dice = np.std(global_results[name]["dice"])
        
        mean_hd95 = np.mean(global_results[name]["hd95"])
        std_hd95 = np.std(global_results[name]["hd95"])
        
        # 排版对齐
        dice_str = f"{mean_dice:.4f} ± {std_dice:.4f}"
        hd95_str = f"{mean_hd95:.2f} ± {std_hd95:.2f}"
        
        print(f"{name:<16} | {dice_str:<20} | {hd95_str:<20}")
    print("═"*75)

if __name__ == "__main__":
    main()