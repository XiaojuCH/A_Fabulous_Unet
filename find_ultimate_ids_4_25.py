import pandas as pd

# ================= 配置区域 =================
CSV_PATH = "./results/master_evaluation_full.csv"
# ============================================

def find_ultimate_cases():
    print("🔍 正在开启上帝视角，从全量评估库中狙击天选神图...")
    df = pd.read_csv(CSV_PATH)

    # ---------------------------------------------------------
    # 1. 数据预处理与透视
    # ---------------------------------------------------------
    # 赛道一：全自动模式 (提取 Prompt 为 YOLO_Box 或 None 的行)
    df_auto = df[(df['Prompt'] == 'YOLO_Box') | (df['Model'].isin(['U-Net', 'Swin-UNETR', 'DeepLabV3+']))].copy()
    pivot_auto = df_auto.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95'])
    pivot_auto.columns = [f"{model}_{metric}" for metric, model in pivot_auto.columns]
    pivot_auto = pivot_auto.reset_index()

    # 赛道二：专家引导模式 (提取 Prompt 为 GT_Box 的行)
    df_expert = df[df['Prompt'] == 'GT_Box'].copy()
    pivot_expert = df_expert.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95'])
    pivot_expert.columns = [f"{model}_{metric}" for metric, model in pivot_expert.columns]
    pivot_expert = pivot_expert.reset_index()

    # ---------------------------------------------------------
    # 2. 赛道一 (全自动模式) 严苛筛选
    # ---------------------------------------------------------
    track1_final = {}
    for mod in ['Colour', 'Infrared']:
        # 筛选条件：
        # 1. 我们的 Dice 很高 (>0.85)，且 HD95 极低 (<15，不断裂)
        # 2. 我们的 Dice 必须比 Swin-UNETR 和 DeepLabV3+ 至少高 0.05（绝对碾压端到端）
        # 3. 我们的 Dice 必须比 SAM 2 Baseline 至少高 0.05
        cond = (pivot_auto['Modality'] == mod) & \
               (pivot_auto['ST-SAM_Dice'] > 0.85) & \
               (pivot_auto['ST-SAM_HD95'] < 15) & \
               (pivot_auto['ST-SAM_Dice'] > pivot_auto['Swin-UNETR_Dice'] + 0.05) & \
               (pivot_auto['ST-SAM_Dice'] > pivot_auto['DeepLabV3+_Dice'] + 0.05) & \
               (pivot_auto['ST-SAM_Dice'] > pivot_auto['BaselineSAM_Dice'] + 0.05)
        
        # 按照竞品的溃败程度（比如 Swin 的 HD95）降序排列，取最惨烈的第一名
        candidates = pivot_auto[cond].sort_values('Swin-UNETR_HD95', ascending=False)
        track1_final[mod] = candidates.iloc[0]['Image_ID'] if not candidates.empty else f"No perfect {mod} match"

    # ---------------------------------------------------------
    # 3. 赛道二 (专家引导模式) 严苛筛选
    # ---------------------------------------------------------
    track2_final = {}
    for mod in ['Colour', 'Infrared']:
        # 筛选条件：
        # 1. 我们的 Dice 极高 (>0.90)，且 HD95 极低 (<10)
        # 2. 我们的 Dice 必须比 MSA、LoRA、MedSAM 和 BaselineSAM 都高出一定幅度
        cond = (pivot_expert['Modality'] == mod) & \
               (pivot_expert['ST-SAM_Dice'] > 0.90) & \
               (pivot_expert['ST-SAM_HD95'] < 10) & \
               (pivot_expert['ST-SAM_Dice'] > pivot_expert['MSA_Dice'] + 0.01) & \
               (pivot_expert['ST-SAM_Dice'] > pivot_expert['LoRA_Dice'] + 0.01) & \
               (pivot_expert['ST-SAM_Dice'] > pivot_expert['BaselineSAM_Dice'] + 0.02)
        
        # 按照我们的 Dice 降序排列，取表现最完美的第一名
        candidates = pivot_expert[cond].sort_values('ST-SAM_Dice', ascending=False)
        track2_final[mod] = candidates.iloc[0]['Image_ID'] if not candidates.empty else f"No perfect {mod} match"

    # ---------------------------------------------------------
    # 4. 打印最终结果
    # ---------------------------------------------------------
    print("\n🎯 狙击完成！以下是经受住全系毒打的绝对统治级图像 ID：")
    print("=" * 60)
    print(f"TRACK1_IDS (全自动赛道) = ['{track1_final['Colour']}', '{track1_final['Infrared']}']")
    print(f"TRACK2_IDS (专家引导赛道) = ['{track2_final['Colour']}', '{track2_final['Infrared']}']")
    print("=" * 60)
    print("👉 下一步：将这 4 个 ID 填入 generate_merged_figure.py 中，然后去原图看一眼填入 ARROWS 坐标，大图即刻出炉！")

if __name__ == "__main__":
    find_ultimate_cases()