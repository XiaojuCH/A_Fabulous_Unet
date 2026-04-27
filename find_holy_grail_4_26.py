import pandas as pd
import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= 核心配置 =================
CSV_PATH = "./results/master_evaluation_fixed.csv"
DATA_ROOT = "./results"
# ============================================

def count_connected_components(mask_path):
    """带抗噪雷达的连通域计算（过滤 <15 像素的微小噪点）"""
    if not os.path.exists(mask_path): return -1
    mask = cv2.imread(mask_path, 0)
    if mask is None: return -1
    
    mask_bool = (mask > 127).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool)
    
    valid_cc = 0
    # 从 1 开始遍历，因为 0 是黑色背景
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 15: # 忽略小于 15 像素的噪点
            valid_cc += 1
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

def find_holy_grail():
    print("🔍 正在开启【抗噪版】双重终极挖掘模式...")
    df = pd.read_csv(CSV_PATH)

    df_auto = df[(df['Prompt'] == 'YOLO_Box') | (df['Model'].isin(['U-Net', 'Swin-UNETR', 'DeepLabV3+']))].copy()
    p_auto = safe_flatten_columns(df_auto.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95']).reset_index())

    df_expert = df[df['Prompt'] == 'GT_Box'].copy()
    p_exp = safe_flatten_columns(df_expert.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95']).reset_index())

    # ================= 1. 赛道一 (全自动模式) =================
    print("\n🚀 [Track 1] 正在进行第一层 HD95 漏斗快筛...")
    p_auto['Max_Comp_HD95'] = p_auto[['Swin-UNETR_HD95', 'BaselineSAM_HD95']].max(axis=1)
    p_auto['HD95_Gap'] = p_auto['Max_Comp_HD95'] - p_auto['ST-SAM_HD95']
    
    # 放宽快筛条件，保证有底池
    candidates_auto = p_auto[(p_auto['ST-SAM_HD95'] < 20) & (p_auto['HD95_Gap'] > 8)]
    
    track1_results = []
    for _, row in tqdm(candidates_auto.iterrows(), total=len(candidates_auto), desc="拓扑深描 (Auto)"):
        img_id = row['Image_ID']
        gt_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_gt", f"{img_id}.png"))
        st_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_stsam_yolo", f"{img_id}.png"))
        swin_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_swinunet", f"{img_id}.png"))
        base_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_baseline_sam_yolo", f"{img_id}.png"))
        
        # 拓扑错误度：距离 GT 差了几个碎片 (缺漏或碎裂都算错误)
        st_err = abs(st_cc - gt_cc)
        swin_err = abs(swin_cc - gt_cc)
        base_err = abs(base_cc - gt_cc)
        max_comp_err = max(swin_err, base_err)
        
        # 条件：GT有目标，ST-SAM允许最多1个小失误，且竞品错得比ST-SAM更离谱
        if gt_cc > 0 and st_err <= 1 and max_comp_err > st_err:
            track1_results.append({
                'Image_ID': img_id, 'Modality': row['Modality'],
                'Comp_Topo_Error': max_comp_err, 'HD95_Gap': row['HD95_Gap'],
                'GT_CC': gt_cc, 'ST_CC': st_cc, 'Swin_CC': swin_cc, 'Base_CC': base_cc
            })

    # ================= 2. 赛道二 (专家模式) =================
    print("\n🚀 [Track 2] 正在进行第一层 HD95 漏斗快筛...")
    p_exp['Max_Comp_HD95'] = p_exp[['MSA_HD95', 'BaselineSAM_HD95']].max(axis=1)
    p_exp['HD95_Gap'] = p_exp['Max_Comp_HD95'] - p_exp['ST-SAM_HD95']
    
    candidates_exp = p_exp[(p_exp['ST-SAM_HD95'] < 15) & (p_exp['HD95_Gap'] > 5)]
    
    track2_results = []
    for _, row in tqdm(candidates_exp.iterrows(), total=len(candidates_exp), desc="拓扑深描 (Expert)"):
        img_id = row['Image_ID']
        gt_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_gt", f"{img_id}.png"))
        st_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_stsam_gt", f"{img_id}.png"))
        msa_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_msa_gt", f"{img_id}.png"))
        base_cc = count_connected_components(os.path.join(DATA_ROOT, "masks_baseline_sam_gt", f"{img_id}.png"))
        
        st_err = abs(st_cc - gt_cc)
        msa_err = abs(msa_cc - gt_cc)
        base_err = abs(base_cc - gt_cc)
        max_comp_err = max(msa_err, base_err)
        
        if gt_cc > 0 and st_err <= 1 and max_comp_err > st_err:
            track2_results.append({
                'Image_ID': img_id, 'Modality': row['Modality'],
                'Comp_Topo_Error': max_comp_err, 'HD95_Gap': row['HD95_Gap'],
                'GT_CC': gt_cc, 'ST_CC': st_cc, 'MSA_CC': msa_cc, 'Base_CC': base_cc
            })

    # ================= 3. 打印终极榜单与兜底 =================
    print("\n" + "🔥"*30)
    print("🏆 【全自动模式 - 圣杯级图库】")
    df_t1 = pd.DataFrame(track1_results)
    for mod in ['Colour', 'Infrared']:
        print(f"\n--- {mod} ---")
        if not df_t1.empty and len(df_t1[df_t1['Modality'] == mod]) > 0:
            top = df_t1[df_t1['Modality'] == mod].sort_values(['Comp_Topo_Error', 'HD95_Gap'], ascending=[False, False]).head(5)
            print(top[['Image_ID', 'Comp_Topo_Error', 'HD95_Gap', 'GT_CC', 'ST_CC', 'Swin_CC', 'Base_CC']].to_string(index=False))
        else:
            print("⚠️ 拓扑深描无结果，触发【智能兜底】，直接输出边缘崩溃（HD95差距最大）的极品：")
            fallback = candidates_auto[candidates_auto['Modality'] == mod].sort_values('HD95_Gap', ascending=False).head(5)
            print(fallback[['Image_ID', 'ST-SAM_HD95', 'HD95_Gap']].to_string(index=False))

    print("\n" + "🔥"*30)
    print("🏆 【专家模式 - 圣杯级图库】")
    df_t2 = pd.DataFrame(track2_results)
    for mod in ['Colour', 'Infrared']:
        print(f"\n--- {mod} ---")
        if not df_t2.empty and len(df_t2[df_t2['Modality'] == mod]) > 0:
            top = df_t2[df_t2['Modality'] == mod].sort_values(['Comp_Topo_Error', 'HD95_Gap'], ascending=[False, False]).head(5)
            print(top[['Image_ID', 'Comp_Topo_Error', 'HD95_Gap', 'GT_CC', 'ST_CC', 'MSA_CC', 'Base_CC']].to_string(index=False))
        else:
            print("⚠️ 拓扑深描无结果，触发【智能兜底】，直接输出边缘崩溃（HD95差距最大）的极品：")
            fallback = candidates_exp[candidates_exp['Modality'] == mod].sort_values('HD95_Gap', ascending=False).head(5)
            print(fallback[['Image_ID', 'ST-SAM_HD95', 'HD95_Gap']].to_string(index=False))

if __name__ == "__main__":
    find_holy_grail()