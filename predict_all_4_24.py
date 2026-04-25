"""
批量预测脚本：对 results/images 中的图片用各模型预测，保存到对应的 masks_xxx 文件夹。

SAM-based 模型需要 box prompt，这里用 GT mask 提取 bbox（评估用途，允许使用GT框）。
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
IMAGES_DIR  = os.path.join(RESULTS_DIR, "images")
MASKS_GT_DIR = os.path.join(RESULTS_DIR, "masks_gt")
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 文件名前缀 -> fold 编号
PREFIX_TO_FOLD = {
    "Color2_": 1,
    "Infrared2_": 3,
    "Infrared3_": 4,
}

def get_fold(img_name):
    for prefix, fold in PREFIX_TO_FOLD.items():
        if img_name.startswith(prefix):
            return fold
    raise ValueError(f"未知前缀: {img_name}")


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return TF.to_tensor(img).unsqueeze(0)  # [1,3,H,W]


def get_gt_box(img_name):
    mask_path = os.path.join(MASKS_GT_DIR, img_name)
    if not os.path.exists(mask_path):
        return torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=torch.float32)
    mask = np.array(Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
    mask = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=torch.float32)
    return torch.tensor([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=torch.float32)


def infer(model, img_tensor, box_tensor, is_sam):
    model.eval()
    with torch.no_grad():
        img = img_tensor.to(DEVICE)
        if is_sam:
            out = model(img, box_tensor.to(DEVICE))
        else:
            out = model(img)
            if isinstance(out, dict):
                out = out["out"]
            elif isinstance(out, list):
                out = out[0]
        pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy()
    return (pred * 255).astype(np.uint8)


def load_sam_model(model_class, ckpt_dir, fold, model_kwargs):
    ckpt = os.path.join(ckpt_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    m = model_class(**model_kwargs)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    return m.to(DEVICE)


def load_baseline_model(model_name, ckpt_dir, fold):
    from train_baseline import get_model
    ckpt = os.path.join(ckpt_dir, model_name, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    m = get_model(model_name)
    state = {k.replace("module.", ""): v for k, v in torch.load(ckpt, map_location=DEVICE).items()}
    m.load_state_dict(state, strict=False)
    return m.to(DEVICE)


# ============================================================
# 主流程
# ============================================================

def run_model_group(tag, loader_fn, out_dir, is_sam):
    os.makedirs(out_dir, exist_ok=True)
    img_names = sorted(os.listdir(IMAGES_DIR))
    print(f"[{tag}] 预测 {len(img_names)} 张图片 -> {out_dir}")
    model_cache = {}  # fold -> model，避免重复加载
    for name in img_names:
        fold = get_fold(name)
        if fold not in model_cache:
            try:
                model_cache[fold] = loader_fn(fold)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue
        model = model_cache[fold]
        img_tensor = load_image(os.path.join(IMAGES_DIR, name))
        box_tensor = get_gt_box(name)
        mask = infer(model, img_tensor, box_tensor, is_sam)
        Image.fromarray(mask).save(os.path.join(out_dir, name))
    print(f"[{tag}] 完成。")


def main():
    base_dir = os.path.dirname(__file__)
    sam_kwargs = {
        "model_cfg": "sam2_hiera_l.yaml",
        "checkpoint_path": os.path.join(base_dir, "checkpoints", "sam2_hiera_large.pt"),
    }

    from model import ST_SAM, Baseline_SAM2, LoRA_SAM2, MSA_Baseline_SAM2, MedSAM_SAM2

    sam_tasks = [
        ("ST_SAM",     ST_SAM,             "checkpoints_run1", "masks_stsam"),
        ("MedSAM",     MedSAM_SAM2,        "checkpoints_medsam",       "masks_medsam"),
        ("MSA",        MSA_Baseline_SAM2,  "checkpoints_msa",          "masks_msa"),
        ("LoRA",       LoRA_SAM2,          "checkpoints_lora",         "masks_lora"),
        ("BaselineSAM",Baseline_SAM2,      "checkpoints_ablation",     "masks_baseline_sam"),
    ]
    for tag, cls, ckpt_subdir, out_name in sam_tasks:
        ckpt_dir = os.path.join(base_dir, ckpt_subdir)
        run_model_group(
            tag,
            lambda fold, cls=cls, ckpt_dir=ckpt_dir: load_sam_model(cls, ckpt_dir, fold, sam_kwargs),
            os.path.join(RESULTS_DIR, out_name),
            is_sam=True,
        )

    baseline_names = ["unet", "swinunet", "attentionunet", "segresnet",
                      "unetplusplus", "deeplab", "deeplab_p", "fcn"]
    ckpt_dir = os.path.join(base_dir, "checkpoints_New_baseline")
    for bname in baseline_names:
        run_model_group(
            bname,
            lambda fold, bname=bname: load_baseline_model(bname, ckpt_dir, fold),
            os.path.join(RESULTS_DIR, f"masks_{bname}"),
            is_sam=False,
        )

    print("全部预测完成。")


if __name__ == "__main__":
    main()
