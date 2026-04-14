import os
import sys
import json
import torch
import math
import csv
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("src")

# 引入 MONAI 全量严谨指标
from monai.metrics import (
    compute_dice, 
    compute_hausdorff_distance,
    compute_iou,
    compute_average_surface_distance
)

from dataset import TearDataset
from model import ST_SAM, Baseline_SAM2, MSA_Baseline_SAM2, LoRA_SAM2

# ================= 全局配置 =================
IMG_SIZE = 1024
MAX_HD95 = math.sqrt(IMG_SIZE**2 + IMG_SIZE**2)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV = "evaluation_results_5folds_full.csv" # 换了个新名字，防止和之前冲突

# 【请核对并修改你的权重路径规则】
CHECKPOINT_PATHS = {
    "ST-SAM": "./checkpoints_run1/fold_{fold}/best_model.pth", 
    "MSA_SAM2": "./checkpoints_msa/fold_{fold}/best_model.pth",         
    "Baseline_SAM2": "./checkpoints_ablation/fold_{fold}/best_model.pth", 
    "LoRA_SAM2": "./checkpoints_lora/fold_{fold}/best_model.pth"        
}

PADDINGS = [-5, 0, 5, 10, 20, 30, 40, "YOLO"]

# ================= 极其严谨的全量指标计算 (严格对齐 get_final_table_v2) =================
def compute_all_metrics(pred_tensor, gt_tensor):
    """
    输入必须是在 CPU 上的 [1, 1, H, W] Tensor
    """
    if pred_tensor.dim() == 3:
        pred_tensor = pred_tensor.unsqueeze(0)
    if gt_tensor.dim() == 3:
        gt_tensor = gt_tensor.unsqueeze(0)
        
    pred = (pred_tensor > 0.5).float()
    lbl = (gt_tensor > 0.5).float()
    
    # 1. Dice & IoU
    if lbl.sum() == 0 and pred.sum() == 0:
        dice_score = 1.0
        iou_score = 1.0
    else:
        dice_score = compute_dice(pred, lbl, include_background=False).item()
        iou_score = compute_iou(pred, lbl, include_background=False).item()
        if math.isnan(dice_score): dice_score = 0.0
        if math.isnan(iou_score): iou_score = 0.0
        
    # 2. Precision & Recall (像素级硬算，最严谨)
    tp = (pred * lbl).sum().item()
    fp = (pred * (1 - lbl)).sum().item()
    fn = ((1 - pred) * lbl).sum().item()
    
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)

    # 3. HD95 & ASD
    if lbl.sum() > 0 and pred.sum() > 0:
        hd95 = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        asd = compute_average_surface_distance(pred, lbl, include_background=False).item()
        if math.isnan(hd95): hd95 = MAX_HD95
        if math.isnan(asd): asd = MAX_HD95 / 2
    elif lbl.sum() > 0 and pred.sum() == 0:
        hd95 = MAX_HD95 
        asd = MAX_HD95 / 2
    else:
        if pred.sum() == 0:
            hd95 = 0.0
            asd = 0.0
        else:
            hd95 = MAX_HD95
            asd = MAX_HD95

    return dice_score, iou_score, recall, precision, hd95, asd

# ================= 智能数据加载器 =================
class RobustnessDataset(TearDataset):
    def __init__(self, data_list, img_size, yolo_pred_json, padding=0):
        super().__init__(data_list, mode="val", img_size=img_size, yolo_pred_json=yolo_pred_json)
        self.padding = padding

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        if self.padding == "YOLO":
            return item
            
        label_np = item['label'].squeeze().numpy()
        y_indices, x_indices = np.where(label_np > 0)
        
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        else:
            x_min, y_min, x_max, y_max = 0, 0, self.img_size, self.img_size

        p = int(self.padding)
        x1 = max(0, x_min - p)
        y1 = max(0, y_min - p)
        x2 = min(self.img_size, x_max + p)
        y2 = min(self.img_size, y_max + p)
        
        item['box'] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        return item

# ================= 主控制流 =================
def main():
    print(f"🚀 启动 5-Fold 全量指标生成管线...")
    print(f"📄 数据将流式保存至: {OUTPUT_CSV}")
    
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # 🔥 表头加入了所有指标
            writer.writerow(['Fold', 'Model', 'Padding', 'Image_ID', 'Modality', 'Dice', 'IoU', 'Recall', 'Precision', 'HD95', 'ASD'])

        model_instances = {
            "ST-SAM": ST_SAM().to(DEVICE),
            "MSA_SAM2": MSA_Baseline_SAM2().to(DEVICE),
            "Baseline_SAM2": Baseline_SAM2().to(DEVICE),
            "LoRA_SAM2": LoRA_SAM2().to(DEVICE)
        }

        for fold in range(5):
            print(f"\n" + "="*50)
            print(f"🔄 正在处理 Fold {fold} ...")
            
            fold_json_path = f"./data_splits/fold_{fold}.json"
            yolo_json_path = f"./data_splits/yolo_boxes_fold{fold}.json"
            
            if not os.path.exists(fold_json_path):
                print(f"⚠️ 找不到 {fold_json_path}，跳过该 Fold。")
                continue
                
            with open(fold_json_path, 'r') as f:
                split_data = json.load(f)

            active_models = {}
            for model_name, model in model_instances.items():
                ckpt_path = CHECKPOINT_PATHS[model_name].format(fold=fold)
                if os.path.exists(ckpt_path):
                    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict)
                    model.eval()
                    active_models[model_name] = model
                else:
                    print(f"  [跳过] 找不到 {model_name} 的权重: {ckpt_path}")

            if not active_models:
                print(f"❌ Fold {fold} 没有任何模型可用，跳过。")
                continue

            for p in PADDINGS:
                print(f"  📐 评估 Padding = {p}")
                val_dataset = RobustnessDataset(split_data['val'], img_size=IMG_SIZE, yolo_pred_json=yolo_json_path, padding=p)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

                for model_name, model in active_models.items():
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc=f"    {model_name}", leave=False):
                            img = batch['image'].to(DEVICE)
                            lbl = batch['label'].to(DEVICE)
                            box = batch['box'].to(DEVICE)
                            img_id = str(batch['id'][0])
                            
                            modality = 'Colour' if ('colour' in img_id.lower() or 'color' in img_id.lower()) else 'Infrared'

                            logits = model(img, box)
                            preds = torch.sigmoid(logits)
                            
                            # 🔥 接收全部 6 个指标
                            d, iou, rec, prec, h, asd = compute_all_metrics(preds.cpu(), lbl.cpu())
                            
                            # 流式写入 CSV
                            writer.writerow([fold, model_name, p, img_id, modality, d, iou, rec, prec, h, asd])
            
            csvfile.flush()

    print(f"\n🎉 全量评估完毕！所有实例级数据已安全保存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()