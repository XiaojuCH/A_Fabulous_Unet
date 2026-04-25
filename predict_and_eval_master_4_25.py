"""
终极管线：基于 LOCO 划分的全量预测 + 评估 + 掩码保存
1. 直接读取 fold_x.json 的 val 集合。
2. 自动加载对应 Fold 的权重。
3. SAM 模型同时进行 GT Box (专家模式) 和 YOLO Box (自动模式) 评估。
4. 所有指标统一写入 master_evaluation_full.csv。
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF
import math
import csv
from tqdm import tqdm

# 引入 MONAI 严谨指标
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_average_surface_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ================= 配置区域 =================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "master_evaluation_full.csv")
DATA_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "data_splits")

IMG_SIZE = 1024
MAX_HD95 = math.sqrt(IMG_SIZE**2 + IMG_SIZE**2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================

def load_image_and_gt(img_path, label_path):
    """加载原图和清洗后的 GT"""
    # 读取原图
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_tensor = TF.to_tensor(img).unsqueeze(0)
    
    # 替换路径以读取去除瞳孔的 Cleaned_Label
    clean_label_path = label_path.replace("/Label/", "/Cleaned_Label/")
    if not os.path.exists(clean_label_path):
        clean_label_path = label_path # 兜底
        
    mask = Image.open(clean_label_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    mask_np = (np.array(mask) > 127).astype(np.uint8)
    gt_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, mask_np, gt_tensor

def get_gt_box(mask_np):
    """从真实掩码提取 GT 框"""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=torch.float32)
    return torch.tensor([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=torch.float32)

def get_yolo_box(img_id, yolo_preds):
    """提取 YOLO 预测框并还原到 1024 尺度"""
    if img_id in yolo_preds:
        box_norm = yolo_preds[img_id]
        box = [
            box_norm[0] * IMG_SIZE, 
            box_norm[1] * IMG_SIZE, 
            box_norm[2] * IMG_SIZE, 
            box_norm[3] * IMG_SIZE
        ]
        return torch.tensor([box], dtype=torch.float32)
    return torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=torch.float32)

def compute_metrics(pred_np, gt_tensor):
    """计算当前图像的核心指标"""
    pred_tensor = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    pred = (pred_tensor > 0.5).float()
    lbl = (gt_tensor > 0.5).float()
    
    if lbl.sum() == 0 and pred.sum() == 0: return 1.0, 0.0, 0.0
        
    dice = compute_dice(pred, lbl, include_background=False).item()
    if math.isnan(dice): dice = 0.0
        
    if lbl.sum() > 0 and pred.sum() > 0:
        hd95 = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        asd = compute_average_surface_distance(pred, lbl, include_background=False).item()
        if math.isnan(hd95): hd95 = MAX_HD95
        if math.isnan(asd): asd = MAX_HD95 / 2
    elif lbl.sum() > 0 and pred.sum() == 0:
        hd95, asd = MAX_HD95, MAX_HD95 / 2
    else:
        hd95, asd = (0.0, 0.0) if pred.sum() == 0 else (MAX_HD95, MAX_HD95)

    return dice, hd95, asd

def infer(model, img_tensor, box_tensor, is_sam):
    """执行模型推理"""
    with torch.no_grad():
        img = img_tensor.to(DEVICE)
        if is_sam:
            out = model(img, box_tensor.to(DEVICE))
        else:
            out = model(img)
            if isinstance(out, dict): out = out["out"]
            elif isinstance(out, list): out = out[0]
        pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy()
    return (pred * 255).astype(np.uint8)

# ===== 模型加载函数 =====
def load_sam_model(model_class, ckpt_dir, fold, model_kwargs):
    ckpt = os.path.join(ckpt_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(ckpt): return None
    m = model_class(**model_kwargs)
    state = {k.replace("module.", ""): v for k, v in torch.load(ckpt, map_location=DEVICE, weights_only=True).items()}
    m.load_state_dict(state, strict=False)
    m.eval()
    return m.to(DEVICE)

def load_baseline_model(model_name, ckpt_dir, fold):
    from train_baseline import get_model
    ckpt = os.path.join(ckpt_dir, model_name, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(ckpt): return None
    m = get_model(model_name)
    state = {k.replace("module.", ""): v for k, v in torch.load(ckpt, map_location=DEVICE, weights_only=True).items()}
    m.load_state_dict(state, strict=False)
    m.eval()
    return m.to(DEVICE)

# ================= 主流程 =================
def run_model_across_folds(tag, loader_fn, out_name_base, is_sam, writer):
    print(f"\n🚀 启动模型评估: [{tag}]")
    
    for fold in range(5):
        # 1. 加载 Fold 数据列表
        split_path = os.path.join(DATA_SPLITS_DIR, f"fold_{fold}.json")
        yolo_path = os.path.join(DATA_SPLITS_DIR, f"yolo_boxes_fold{fold}.json")
        
        if not os.path.exists(split_path): continue
        with open(split_path, 'r') as f:
            val_list = json.load(f)['val']
            
        yolo_preds = {}
        if os.path.exists(yolo_path):
            with open(yolo_path, 'r') as f:
                yolo_preds = json.load(f)

        # 2. 加载该 Fold 的模型权重
        model = loader_fn(fold)
        if model is None:
            print(f"  ⚠️ [Fold {fold}] 未找到权重，已跳过。")
            continue
            
        print(f"  ✅ [Fold {fold}] 模型已加载，开始预测 {len(val_list)} 张测试图...")
        
        # 定义输出文件夹
        if is_sam:
            dir_gt = os.path.join(RESULTS_DIR, f"{out_name_base}_gt")
            dir_yolo = os.path.join(RESULTS_DIR, f"{out_name_base}_yolo")
            os.makedirs(dir_gt, exist_ok=True)
            os.makedirs(dir_yolo, exist_ok=True)
        else:
            dir_cnn = os.path.join(RESULTS_DIR, f"{out_name_base}")
            os.makedirs(dir_cnn, exist_ok=True)
            
        # 3. 遍历预测
        for item in tqdm(val_list, desc=f"    Fold {fold} Infer", leave=False):
            img_id = item['id']
            modality = 'Colour' if 'Color' in img_id else 'Infrared'
            
            img_tensor, mask_np, gt_tensor = load_image_and_gt(item['image'], item['label'])
            
            if is_sam:
                # ============ SAM 模型：跑两次 (专家 GT + 全自动 YOLO) ============
                # A. 专家模式 (GT Box)
                box_gt = get_gt_box(mask_np)
                pred_gt = infer(model, img_tensor, box_gt, is_sam=True)
                Image.fromarray(pred_gt).save(os.path.join(dir_gt, f"{img_id}.png"))
                dice_gt, hd_gt, asd_gt = compute_metrics(pred_gt, gt_tensor)
                writer.writerow([fold, img_id, modality, tag, 'GT_Box', dice_gt, hd_gt, asd_gt])
                
                # B. 自动模式 (YOLO Box)
                box_yolo = get_yolo_box(img_id, yolo_preds)
                pred_yolo = infer(model, img_tensor, box_yolo, is_sam=True)
                Image.fromarray(pred_yolo).save(os.path.join(dir_yolo, f"{img_id}.png"))
                dice_yolo, hd_yolo, asd_yolo = compute_metrics(pred_yolo, gt_tensor)
                writer.writerow([fold, img_id, modality, tag, 'YOLO_Box', dice_yolo, hd_yolo, asd_yolo])
                
            else:
                # ============ CNN 模型：跑一次 ============
                pred_cnn = infer(model, img_tensor, None, is_sam=False)
                Image.fromarray(pred_cnn).save(os.path.join(dir_cnn, f"{img_id}.png"))
                dice_cnn, hd_cnn, asd_cnn = compute_metrics(pred_cnn, gt_tensor)
                writer.writerow([fold, img_id, modality, tag, 'None', dice_cnn, hd_cnn, asd_cnn])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base_dir = os.path.dirname(__file__)
    sam_kwargs = {
        "model_cfg": "sam2_hiera_l.yaml",
        "checkpoint_path": os.path.join(base_dir, "checkpoints", "sam2_hiera_large.pt"),
    }

    from model import ST_SAM, Baseline_SAM2, LoRA_SAM2, MSA_Baseline_SAM2, MedSAM_SAM2

    # 初始化 CSV
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Fold', 'Image_ID', 'Modality', 'Model', 'Prompt', 'Dice', 'HD95', 'ASD'])

        # 1. 评估 SAM 家族
        sam_tasks = [
            ("ST-SAM",      ST_SAM,             "checkpoints_run1",     "masks_stsam"),
            ("MedSAM",      MedSAM_SAM2,        "checkpoints_medsam",   "masks_medsam"),
            ("MSA",         MSA_Baseline_SAM2,  "checkpoints_msa",      "masks_msa"),
            ("LoRA",        LoRA_SAM2,          "checkpoints_lora",     "masks_lora"),
            ("BaselineSAM", Baseline_SAM2,      "checkpoints_ablation", "masks_baseline_sam"),
        ]
        for tag, cls, ckpt_subdir, out_name in sam_tasks:
            ckpt_dir = os.path.join(base_dir, ckpt_subdir)
            run_model_across_folds(tag, lambda f, cls=cls, cd=ckpt_dir: load_sam_model(cls, cd, f, sam_kwargs),
                                   out_name, is_sam=True, writer=writer)

        # 2. 评估 CNN 家族
        baseline_names = ["unet", "swinunet", "deeplab"] 
        ckpt_dir = os.path.join(base_dir, "checkpoints_New_baseline")
        for bname in baseline_names:
            tag_map = {"unet": "U-Net", "swinunet": "Swin-UNETR", "deeplab": "DeepLabV3+"}
            tag = tag_map.get(bname, bname)
            run_model_across_folds(tag, lambda f, b=bname: load_baseline_model(b, ckpt_dir, f),
                                   f"masks_{bname}", is_sam=False, writer=writer)

    print(f"\n🎉 伟大的胜利！全量预测与评估结束！数据已安全降落至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()