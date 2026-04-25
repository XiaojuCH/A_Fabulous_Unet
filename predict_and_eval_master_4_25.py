"""
终极管线：全量预测 + 全量指标评估 + 掩码保存
将所有模型（CNN + SAM）在所有图像上的表现统一汇总为一份 Master CSV。
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF
import math
import csv

# 引入 MONAI 严谨指标
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_average_surface_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ================= 配置区域 =================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
IMAGES_DIR  = os.path.join(RESULTS_DIR, "images")
MASKS_GT_DIR = os.path.join(RESULTS_DIR, "masks_gt")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "master_evaluation_full.csv")

IMG_SIZE = 1024
MAX_HD95 = math.sqrt(IMG_SIZE**2 + IMG_SIZE**2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREFIX_TO_FOLD = {"Color2_": 1, "Infrared2_": 3, "Infrared3_": 4}
# ============================================

def get_fold(img_name):
    for prefix, fold in PREFIX_TO_FOLD.items():
        if img_name.startswith(prefix):
            return fold
    return 0 # 默认兜底

def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return TF.to_tensor(img).unsqueeze(0)

def get_gt_box_and_tensor(img_name):
    mask_path = os.path.join(MASKS_GT_DIR, img_name)
    if not os.path.exists(mask_path):
        return torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=torch.float32), torch.zeros((1, 1, IMG_SIZE, IMG_SIZE))
    
    mask = np.array(Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
    mask_bool = (mask > 127).astype(np.uint8)
    
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        box = torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=torch.float32)
    else:
        box = torch.tensor([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=torch.float32)
        
    gt_tensor = torch.tensor(mask_bool, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return box, gt_tensor

def compute_metrics(pred_np, gt_tensor):
    """计算当前图像的 3 大核心指标"""
    pred_tensor = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    pred = (pred_tensor > 0.5).float()
    lbl = (gt_tensor > 0.5).float()
    
    if lbl.sum() == 0 and pred.sum() == 0:
        return 1.0, 0.0, 0.0
        
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
    model.eval()
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

# ===== 模型加载函数 (保持你原来的不变) =====
def load_sam_model(model_class, ckpt_dir, fold, model_kwargs):
    ckpt = os.path.join(ckpt_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(ckpt): raise FileNotFoundError(ckpt)
    m = model_class(**model_kwargs)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    return m.to(DEVICE)

def load_baseline_model(model_name, ckpt_dir, fold):
    from train_baseline import get_model
    ckpt = os.path.join(ckpt_dir, model_name, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(ckpt): raise FileNotFoundError(ckpt)
    m = get_model(model_name)
    state = {k.replace("module.", ""): v for k, v in torch.load(ckpt, map_location=DEVICE).items()}
    m.load_state_dict(state, strict=False)
    return m.to(DEVICE)

# ================= 主流程 =================
def run_model_group(tag, loader_fn, out_dir, is_sam, writer):
    os.makedirs(out_dir, exist_ok=True)
    img_names = sorted(os.listdir(IMAGES_DIR))
    print(f"🚀 [{tag}] 开始全量预测与评估 ({len(img_names)} 张图)...")
    
    model_cache = {} 
    for name in img_names:
        fold = get_fold(name)
        if fold not in model_cache:
            try: model_cache[fold] = loader_fn(fold)
            except FileNotFoundError as e:
                print(f"  [skip] 找不到 fold {fold} 的权重")
                continue
        
        model = model_cache[fold]
        img_id = name.replace('.png', '').replace('.PNG', '')
        modality = 'Colour' if 'Color' in name else 'Infrared'
        
        # 1. 准备数据
        img_tensor = load_image(os.path.join(IMAGES_DIR, name))
        box_tensor, gt_tensor = get_gt_box_and_tensor(name)
        
        # 2. 推理并保存掩码
        mask_np = infer(model, img_tensor, box_tensor, is_sam)
        Image.fromarray(mask_np).save(os.path.join(out_dir, name))
        
        # 3. 计算指标并写入 CSV
        dice, hd95, asd = compute_metrics(mask_np, gt_tensor)
        writer.writerow([img_id, modality, tag, 'GT_Box' if is_sam else 'None', dice, hd95, asd])

def main():
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
            writer.writerow(['Image_ID', 'Modality', 'Model', 'Prompt', 'Dice', 'HD95', 'ASD'])

        # 1. 跑 SAM 系列
        sam_tasks = [
            ("ST-SAM",      ST_SAM,             "checkpoints_run1",     "masks_stsam"),
            ("MedSAM",      MedSAM_SAM2,        "checkpoints_medsam",   "masks_medsam"),
            ("MSA",         MSA_Baseline_SAM2,  "checkpoints_msa",      "masks_msa"),
            ("LoRA",        LoRA_SAM2,          "checkpoints_lora",     "masks_lora"),
            ("BaselineSAM", Baseline_SAM2,      "checkpoints_ablation", "masks_baseline_sam"),
        ]
        for tag, cls, ckpt_subdir, out_name in sam_tasks:
            ckpt_dir = os.path.join(base_dir, ckpt_subdir)
            run_model_group(tag, lambda f, cls=cls, cd=ckpt_dir: load_sam_model(cls, cd, f, sam_kwargs),
                            os.path.join(RESULTS_DIR, out_name), is_sam=True, writer=writer)

        # 2. 跑 CNN 系列
        baseline_names = ["unet", "swinunet", "deeplab"] # 重点跑这三个作为竞品即可，节省时间
        ckpt_dir = os.path.join(base_dir, "checkpoints_New_baseline")
        for bname in baseline_names:
            # tag 格式化一下，好看点
            tag_map = {"unet": "U-Net", "swinunet": "Swin-UNETR", "deeplab": "DeepLabV3+"}
            tag = tag_map.get(bname, bname)
            run_model_group(tag, lambda f, b=bname: load_baseline_model(b, ckpt_dir, f),
                            os.path.join(RESULTS_DIR, f"masks_{bname}"), is_sam=False, writer=writer)

    print(f"\n🎉 伟大的胜利！全量预测与评估结束！数据已安全降落至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()