import pandas as pd
import cv2
import numpy as np
import os
from tqdm import tqdm

CSV_PATH = "./results/master_evaluation_fixed.csv"
DATA_ROOT = "./results"

def count_connected_components(mask_path):
    if not os.path.exists(mask_path): return -1
    mask = cv2.imread(mask_path, 0)
    if mask is None: return -1
    mask_bool = (mask > 127).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool)
    valid_cc = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 15)
    return valid_cc

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

def find_elegant_cases():
    print("🔍 正在寻找【优雅的失败】(模型大概找对了，但发生了关键性断裂或溢出)...")
    df = pd.read_csv(CSV_PATH)

    df_auto = df[(df['Prompt'] == 'YOLO_Box') | (df['Model'].isin(['U-Net', 'Swin-UNETR', 'DeepLabV3+']))].copy()
    p_auto = safe_flatten_columns(df_auto.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95']).reset_index())

    print("\n🚀 [Track 1] 正在扫描全自动彩色图...")
    # 【优雅过滤条件】：
    # 1. 我们的 Dice > 0.85 (我们很好)
    # 2. Swin 的 Dice > 0.5 且 HD95 < 50 (Swin 也在努力，没有乱飞)
    candidates_auto = p_auto[(p_auto['Modality'] == 'Colour') & 
                             (p_auto['ST-SAM_Dice'] > 0.85) & 
                             (p_auto['Swin-UNETR_Dice'] > 0.5) & 
                             (p_auto['Swin-UNETR_HD95'] < 50)]
    
    track1_color = []
    for _, row in tqdm(candidates_auto.iterrows(), total=len(candidates_auto)):
        img_id = row['Image_ID']
        gt_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_gt", f"{img_id}.png"))
        st_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_stsam_yolo", f"{img_id}.png"))
        swin_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_swinunet", f"{img_id}.png"))
        
        # 寻找断裂：真实是1，我们是1，Swin 断成了 2 截或更多！
        if gt_cc == 1 and st_cc == 1 and swin_cc >= 2:
            track1_color.append({
                'Image_ID': img_id,
                'Swin_Fragments': swin_cc,
                'ST-SAM_Dice': row['ST-SAM_Dice']
            })

    df_t1 = pd.DataFrame(track1_color)
    print("\n" + "🔥"*30)
    print("🏆 【全自动模式 - 彩色图 (优雅断裂榜单)】")
    if not df_t1.empty:
        # 按 Swin 碎的段数降序，同段数按我们的 Dice 降序
        top = df_t1.sort_values(['Swin_Fragments', 'ST-SAM_Dice'], ascending=[False, False]).head(5)
        print(top.to_string(index=False))
    else:
        print("没找到断裂图，建议直接去之前生成的海选图里挑一张视觉效果最好的！")

if __name__ == "__main__":
    find_elegant_cases()