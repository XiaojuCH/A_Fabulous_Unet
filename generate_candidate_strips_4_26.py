import cv2
import numpy as np
import pandas as pd
import os
import json
import glob

# ================= 核心配置 =================
CSV_PATH = "./results/master_evaluation_fixed.csv"
DATA_ROOT = "./results"
JSON_DIR = "./data_splits"
OUT_DIR = "./results/candidate_strips"
os.makedirs(OUT_DIR, exist_ok=True)

TRACK1_MODELS = [("U-Net", "masks_unet"), ("Swin-UNETR", "masks_swinunet"), ("DeepLabV3+", "masks_deeplab"), ("SAM2 Base", "masks_baseline_sam_yolo"), ("ST-SAM (Ours)", "masks_stsam_yolo")]
TRACK2_MODELS = [("MedSAM", "masks_medsam_gt"), ("SAM2 Base", "masks_baseline_sam_gt"), ("SAM2 LoRA", "masks_lora_gt"), ("SAM2 MSA", "masks_msa_gt"), ("ST-SAM (Ours)", "masks_stsam_gt")]
# ============================================

def build_image_dict():
    """解析 5 个 Fold 的 JSON，建立 ID 到原图和真实 GT 的映射字典"""
    img_dict = {}
    for fold in range(5):
        json_path = os.path.join(JSON_DIR, f"fold_{fold}.json")
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f:
            data = json.load(f)
            for split in ['train', 'val']:
                for item in data[split]:
                    img_dict[item['id']] = {
                        'image': item['image'],
                        'label': item['label'].replace("/Label/", "/Cleaned_Label/")
                    }
    return img_dict

def find_mask_path(folder_path, img_id):
    search_pattern = os.path.join(folder_path, f"{img_id}.*")
    matches = glob.glob(search_pattern)
    return matches[0] if matches else None

def make_panel(img_rgb, mask_path, title, color=(0, 255, 0)):
    """将掩码盖在原图上，并在左上角贴上模型名称标签"""
    overlay = img_rgb.copy()
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 127] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.55, 0)
    
    (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(overlay, (0, 0), (w + 40, h + 30), (0, 0, 0), -1)
    cv2.putText(overlay, title, (20, h + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return overlay

def create_strip(img_id, track_name, models, img_dict):
    if img_id not in img_dict: return
    img_bgr = cv2.imread(img_dict[img_id]['image'])
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (1024, 1024)), cv2.COLOR_BGR2RGB)
    
    panels = []
    panels.append(make_panel(img_rgb, None, "Input"))
    
    gt_path = img_dict[img_id]['label']
    panels.append(make_panel(img_rgb, gt_path if os.path.exists(gt_path) else None, "GT", color=(255, 255, 0)))
    
    for model_name, folder in models:
        m_path = find_mask_path(os.path.join(DATA_ROOT, folder), img_id)
        color = (0, 255, 0) if "ST-SAM" in model_name else (255, 0, 0) 
        panels.append(make_panel(img_rgb, m_path, model_name, color))
        
    strip = np.hstack(panels)
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    
    out_path = os.path.join(OUT_DIR, f"{track_name}_{img_id}.png")
    cv2.imwrite(out_path, strip_bgr)
    print(f"✅ 生成对比图: {out_path}")

def generate_candidates():
    print("🔍 正在开启海选模式，读取大表...")
    df = pd.read_csv(CSV_PATH)
    img_dict = build_image_dict()

    df_auto = df[(df['Prompt'] == 'YOLO_Box') | (df['Model'].isin(['U-Net', 'Swin-UNETR', 'DeepLabV3+']))].copy()
    p_auto = df_auto.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95']).reset_index()
    # 【修复1】：完美的展平逻辑，不再破坏原列名
    p_auto.columns = [f"{model}_{metric}" if model else metric for metric, model in p_auto.columns]

    df_expert = df[df['Prompt'] == 'GT_Box'].copy()
    p_exp = df_expert.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95']).reset_index()
    p_exp.columns = [f"{model}_{metric}" if model else metric for metric, model in p_exp.columns]

    # 【修复2】：统一使用 "ST-SAM_Dice" 而不是带 (Ours) 的名字进行过滤
# 把代码里的 auto_c, auto_i 等列表直接写死为我们挑出的极品：
    auto_c = ['Color1_000362', 'Color1_000563']
    auto_i = ['Infrared3_000083']
    exp_c = ['Color2_000660']
    exp_i = ['Infrared2_000203', 'Infrared2_000233']
    
    print(f"🚀 开始生成 Track 1 (全自动) 候选图...")
    for img_id in auto_c + auto_i:
        create_strip(img_id, "Track1_Auto", TRACK1_MODELS, img_dict)
        
    print(f"🚀 开始生成 Track 2 (专家) 候选图...")
    for img_id in exp_c + exp_i:
        create_strip(img_id, "Track2_Expert", TRACK2_MODELS, img_dict)

    print(f"\n🎉 全部 40 张海选对比图已生成！请前往 {OUT_DIR} 挑选！")

if __name__ == "__main__":
    generate_candidates()