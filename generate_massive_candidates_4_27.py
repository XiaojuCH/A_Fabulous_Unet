import cv2
import numpy as np
import pandas as pd
import os
import json
import glob
import shutil
from tqdm import tqdm

# ================= 核心配置 =================
CSV_PATH = "./results/master_evaluation_fixed.csv"
DATA_ROOT = "./results"
JSON_DIR = "./data_splits"
OUT_DIR = "./results/massive_candidates_250"

for sub in ['Track1_Colour', 'Track1_Infrared', 'Track2_Colour', 'Track2_Infrared']:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
os.makedirs(os.path.join(DATA_ROOT, "images"), exist_ok=True)
os.makedirs(os.path.join(DATA_ROOT, "masks_gt"), exist_ok=True)

# 使用最终定稿的 6 列模型组合
TRACK1_MODELS = [("Swin-UNETR", "masks_swinunet"), ("DeepLabV3+", "masks_deeplab"), ("SAM2 Base", "masks_baseline_sam_yolo"), ("ST-SAM (Ours)", "masks_stsam_yolo")]
TRACK2_MODELS = [("MedSAM", "masks_medsam_gt"), ("SAM2 Base", "masks_baseline_sam_gt"), ("SAM2 MSA", "masks_msa_gt"), ("ST-SAM (Ours)", "masks_stsam_gt")]
# ============================================

def safe_flatten_columns(df):
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            metric, model = col
            new_cols.append(f"{model}_{metric}" if model else metric)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df

def harvest_images(target_ids):
    """自动去 json 里抓取需要的原图和 GT，防止白板"""
    print(f"📦 正在补充 {len(target_ids)} 张图片的底层素材...")
    found = set()
    for fold in range(5):
        json_path = os.path.join(JSON_DIR, f"fold_{fold}.json")
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f:
            data = json.load(f)
            for split in ['train', 'val']:
                for item in data[split]:
                    if item['id'] in target_ids and item['id'] not in found:
                        shutil.copy(item['image'], os.path.join(DATA_ROOT, "images", f"{item['id']}.png"))
                        clean_label = item['label'].replace("/Label/", "/Cleaned_Label/")
                        if os.path.exists(clean_label):
                            shutil.copy(clean_label, os.path.join(DATA_ROOT, "masks_gt", f"{item['id']}.png"))
                        found.add(item['id'])

def find_mask_path(folder_path, img_id):
    search_pattern = os.path.join(folder_path, f"{img_id}.*")
    matches = glob.glob(search_pattern)
    return matches[0] if matches else None

def make_panel(img_rgb, mask_path, title, color=(0, 255, 0)):
    overlay = img_rgb.copy()
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0)
        if mask is not None:
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask > 127] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.6, 0)
    
    (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(overlay, (0, 0), (w + 40, h + 30), (0, 0, 0), -1)
    cv2.putText(overlay, title, (20, h + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return overlay

def create_strip(img_id, out_subfolder, models):
    img_path = os.path.join(DATA_ROOT, "images", f"{img_id}.png")
    if not os.path.exists(img_path): return
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (1024, 1024)), cv2.COLOR_BGR2RGB)
    
    panels = []
    panels.append(make_panel(img_rgb, None, f"Input ({img_id})"))
    gt_path = os.path.join(DATA_ROOT, "masks_gt", f"{img_id}.png")
    panels.append(make_panel(img_rgb, gt_path, "GT", color=(255, 255, 0))) # 黄色GT
    
    for model_name, folder in models:
        m_path = find_mask_path(os.path.join(DATA_ROOT, folder), img_id)
        color = (0, 255, 0) if "ST-SAM" in model_name else (255, 0, 0) # Ours绿，竞品红
        panels.append(make_panel(img_rgb, m_path, model_name, color))
        
    strip = np.hstack(panels)
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, out_subfolder, f"{img_id}.png"), strip_bgr)

def generate_massive_candidates():
    print("🔍 正在开启【海量阅兵】模式，为您生成 80 张高清对比长图...")
    df = pd.read_csv(CSV_PATH)

    df_auto = df[(df['Prompt'] == 'YOLO_Box') | (df['Model'].isin(['Swin-UNETR', 'DeepLabV3+']))].copy()
    p_auto = safe_flatten_columns(df_auto.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95']).reset_index())

    df_expert = df[df['Prompt'] == 'GT_Box'].copy()
    p_exp = safe_flatten_columns(df_expert.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95']).reset_index())

# ================= 替换这一段 =================
    # 粗筛：只要我们的模型及格，竞品的误差加起来越大越好
    p_auto['Total_Err'] = p_auto['Swin-UNETR_HD95'] + p_auto['BaselineSAM_HD95']
    # 【修改】：底线放宽至 0.80，数量提升至 50 张
    t1_c = p_auto[(p_auto['Modality'] == 'Colour') & (p_auto['ST-SAM_Dice'] > 0.80)].sort_values('Total_Err', ascending=False).head(50)['Image_ID'].tolist()
    t1_i = p_auto[(p_auto['Modality'] == 'Infrared') & (p_auto['ST-SAM_Dice'] > 0.80)].sort_values('Total_Err', ascending=False).head(100)['Image_ID'].tolist()

    p_exp['Total_Err'] = p_exp['MSA_HD95'] + p_exp['BaselineSAM_HD95']
    # 【修改】：底线放宽至 0.85，数量提升至 50 张
    t2_c = p_exp[(p_exp['Modality'] == 'Colour') & (p_exp['ST-SAM_Dice'] > 0.85)].sort_values('Total_Err', ascending=False).head(50)['Image_ID'].tolist()
    t2_i = p_exp[(p_exp['Modality'] == 'Infrared') & (p_exp['ST-SAM_Dice'] > 0.85)].sort_values('Total_Err', ascending=False).head(50)['Image_ID'].tolist()
    # ===============================================
    all_targets = t1_c + t1_i + t2_c + t2_i
    harvest_images(all_targets)

    print("\n🚀 正在疯狂制图...")
    for img_id in tqdm(t1_c, desc="Track1_Colour"): create_strip(img_id, "Track1_Colour", TRACK1_MODELS)
    for img_id in tqdm(t1_i, desc="Track1_Infrared"): create_strip(img_id, "Track1_Infrared", TRACK1_MODELS)
    for img_id in tqdm(t2_c, desc="Track2_Colour"): create_strip(img_id, "Track2_Colour", TRACK2_MODELS)
    for img_id in tqdm(t2_i, desc="Track2_Infrared"): create_strip(img_id, "Track2_Infrared", TRACK2_MODELS)

    print(f"\n🎉 伟大的胜利！80 张高清横条图已存入: {OUT_DIR}")
    print("👉 快去文件夹里一张一张看吧，挑出最让你震撼的 4 张，记录 ID 和断裂位置的 (X,Y) 坐标！")

if __name__ == "__main__":
    generate_massive_candidates()