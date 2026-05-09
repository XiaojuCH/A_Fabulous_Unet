import sys
import os
import argparse
sys.path.append("src") 

import torch
import torch.nn.functional as F  # <--- 新增：用于 Resize 图像
import numpy as np
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入计算库
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ 未安装 thop，将跳过 FLOPs 计算")

from monai.metrics import (
    compute_dice, compute_hausdorff_distance, 
    compute_average_surface_distance, compute_iou
)
# 引入所有 Baseline 模型
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, SegResNet, BasicUNetPlusPlus
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, fcn_resnet50, FCN_ResNet50_Weights
from dataset import TearDataset

# ================= 配置区域 (必须与 ST-SAM 一致) =================
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HD95 惩罚值 (对角线长度)
MAX_HD95 = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2) 
# ==============================================================

def get_model(name):
    name = name.lower()
    if name == "unet":
        return UNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2), num_res_units=2,
        )
    elif name == "swinunet":
        return SwinUNETR(
            in_channels=3, out_channels=1,
            feature_size=48, spatial_dims=2,
            use_v2=True,
            window_size=8      # 适配 1024 (1024/32=32, 32%8=0)
        )
    elif name == "attentionunet":
        return AttentionUnet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
    elif name == "segresnet":
        return SegResNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1]
        )
    elif name == "unetplusplus":
        return BasicUNetPlusPlus(
            spatial_dims=2, in_channels=3, out_channels=1,
            features=(16, 32, 64, 128, 256, 256),
            deep_supervision=False
        )
    elif name == "deeplab":
        model = deeplabv3_resnet50(weights=None, num_classes=1)
        model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif name == "deeplab_p":
        m = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        m.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
        m.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
        return m
    elif name == "fcn":
        m = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        m.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1))
        m.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
        return m
    else:
        raise ValueError(f"Unknown model: {name}")

def get_complexity(model_name):
    """计算 Params 和 FLOPs"""
    try:
        model = get_model(model_name).to(DEVICE)
        model.eval()
        input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        
        if THOP_AVAILABLE:
            flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
            return flops / 1e9, params / 1e6
        else:
            return 0, 0
    except Exception as e:
        print(f"⚠️ {model_name} FLOPs 计算失败: {e}")
        return 0, 0

def calculate_metrics_robust(pred, lbl):
    """【核心】与 ST-SAM 的计算逻辑完全一致！"""
    results = {}
    
    # 1. Dice & IoU
    dice_score = compute_dice(pred, lbl, include_background=False).item()
    iou_score = compute_iou(pred, lbl, include_background=False).item()
    
    # 修正全黑情况 (Empty GT & Empty Pred)
    if lbl.sum() == 0 and pred.sum() == 0:
        dice_score = 1.0
        iou_score = 1.0
    
    results['dice'] = dice_score
    results['iou'] = iou_score
    
    # 2. Precision & Recall
    tp = (pred * lbl).sum().item()
    fp = (pred * (1 - lbl)).sum().item()
    fn = ((1 - pred) * lbl).sum().item()
    
    results['recall'] = tp / (tp + fn + 1e-6)
    results['precision'] = tp / (tp + fp + 1e-6)
    
    # 3. HD95 & ASD (带惩罚)
    if lbl.sum() > 0 and pred.sum() > 0:
        results['hd95'] = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        results['asd'] = compute_average_surface_distance(pred, lbl, include_background=False).item()
    elif lbl.sum() > 0 and pred.sum() == 0:
        results['hd95'] = MAX_HD95 
        results['asd'] = MAX_HD95 / 2 
    else:
        if pred.sum() == 0:
            results['hd95'] = 0.0; results['asd'] = 0.0
        else:
            results['hd95'] = MAX_HD95; results['asd'] = MAX_HD95

    return results

def evaluate_fold(model_name, fold):
    split_path = f"./data_splits/fold_{fold}.json"
    if not os.path.exists(split_path): return None

    with open(split_path, 'r') as f: data = json.load(f)
    
    # 🔥【修改 1】：强行传入 YOLO 框的 JSON 路径
    yolo_json_path = f"./data_splits/yolo_boxes_fold{fold}.json"
    dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE, yolo_pred_json=yolo_json_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model = get_model(model_name).to(DEVICE)
    
    # 路径检查
    ckpt_path = f"./checkpoints_New_baseline/{model_name}/fold_{fold}/best_model.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"./checkpoints/{model_name}/fold_{fold}/best_model.pth"
        if not os.path.exists(ckpt_path):
            print(f"⚠️ [Fold {fold}] 权重缺失: {ckpt_path}")
            return None

    # 加载权重
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."): new_state_dict[k[7:]] = v
        else: new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except:
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()
    
    metrics_log = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=f"Eval (Masked) {model_name} F{fold}"):
            img_full, lbl_full = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            box_tensor = batch['box'][0] # [x1, y1, x2, y2]

            # ========================================================
            # 1. 正常全图推理，保证感受野和尺度与训练时绝对一致
            # ========================================================
            output = model(img_full)
            if isinstance(output, dict) and 'out' in output: logits_full = output['out']
            elif isinstance(output, list): logits_full = output[0]
            else: logits_full = output
            
            pred_full = (torch.sigmoid(logits_full) > 0.5).float()

            # ========================================================
            # 2. 引入 YOLO 空间先验：擦除 ROI 之外的假阳性预测
            # ========================================================
            padding = 20
            x1, y1, x2, y2 = [int(v.item()) for v in box_tensor]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(IMG_SIZE, x2 + padding)
            y2 = min(IMG_SIZE, y2 + padding)

            # 只有当 YOLO 成功输出合法框时，才进行背景过滤
            if x2 > x1 and y2 > y1 and (x2-x1) >= 10 and (y2-y1) >= 10:
                spatial_mask = torch.zeros_like(pred_full)
                spatial_mask[:, :, y1:y2, x1:x2] = 1.0  # 框内保留
                pred_full = pred_full * spatial_mask    # 框外置 0 (物理抹除)

            # ========================================================
            # 3. 计算指标
            # ========================================================
            pred_full, lbl_full = pred_full.cpu(), lbl_full.cpu()
            batch_res = calculate_metrics_robust(pred_full, lbl_full)

            for k, v in batch_res.items(): metrics_log[k].append(v)
                
    return {k: np.mean(v) for k, v in metrics_log.items()}

if __name__ == "__main__":
    print("🚀 Baseline [带 YOLO 框截取] 公平评估脚本 (SCI Mode)")
    print(f"📌 Device: {DEVICE} | HD95 Penalty: {MAX_HD95:.2f}")
    
    # 推荐只跑最强的基线作为反驳即可
    models_to_run = ["deeplab_p", "swinunet"]

    for model_name in models_to_run:
        print(f"\n{'='*90}")
        print(f"📋 Processing Model (With YOLO Crop): {model_name.upper()}")
        
        flops, params = get_complexity(model_name)
        
        headers = ["Fold", "Dice", "IoU", "Recall", "Prec", "HD95", "ASD"]
        print("-" * 90)
        print(" | ".join([f"{h:<8}" for h in headers]))
        print("-" * 90)
        
        all_folds_metrics = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
        
        for fold in range(5):
            res = evaluate_fold(model_name, fold)
            if res:
                for k, v in res.items(): all_folds_metrics[k].append(v)
                print(f"{fold:<8} | {res['dice']:.4f}   | {res['iou']:.4f}   | {res['recall']:.4f}   | {res['precision']:.4f} | {res['hd95']:.4f}   | {res['asd']:.4f}")
        
        if len(all_folds_metrics['dice']) > 0:
            print("-" * 90)
            print(f"🏆 {model_name.upper()} (Cropped) Final Average:")
            for k in headers[1:]:
                k_lower = k.lower() if k != "Prec" else "precision"
                avg = np.mean(all_folds_metrics[k_lower])
                std = np.std(all_folds_metrics[k_lower])
                print(f"   ● {k:<10}: {avg:.4f} ± {std:.4f}")
        else:
            print(f"❌ {model_name} 没有产生有效结果。")