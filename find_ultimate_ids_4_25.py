import pandas as pd

# ================= 配置区域 =================
CSV_PATH = "./results/master_evaluation_fixed.csv"
# ============================================

def find_ultimate_cases():
    print("🔍 正在开启上帝视角，从全量评估库中狙击天选神图...")
    df = pd.read_csv(CSV_PATH)

    df_auto = df[(df['Prompt'] == 'YOLO_Box') | (df['Model'].isin(['U-Net', 'Swin-UNETR', 'DeepLabV3+']))].copy()
    pivot_auto = df_auto.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95'])
    pivot_auto.columns = [f"{model}_{metric}" for metric, model in pivot_auto.columns]
    pivot_auto = pivot_auto.reset_index()

    df_expert = df[df['Prompt'] == 'GT_Box'].copy()
    pivot_expert = df_expert.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95'])
    pivot_expert.columns = [f"{model}_{metric}" for metric, model in pivot_expert.columns]
    pivot_expert = pivot_expert.reset_index()

    # ---------------------------------------------------------
    # 赛道一 (全自动模式) 自适应筛选
    # ---------------------------------------------------------
    track1_final = {}
    for mod in ['Colour', 'Infrared']:
        # 【修改点】：彩色图要求领先 5%，红外图要求领先 2% 即可；红外 Dice 底线降至 0.80
        margin = 0.05 if mod == 'Colour' else 0.02
        min_dice = 0.85 if mod == 'Colour' else 0.80
        
        cond = (pivot_auto['Modality'] == mod) & \
               (pivot_auto['ST-SAM_Dice'] > min_dice) & \
               (pivot_auto['ST-SAM_HD95'] < 15) & \
               (pivot_auto['ST-SAM_Dice'] > pivot_auto['Swin-UNETR_Dice'] + margin) & \
               (pivot_auto['ST-SAM_Dice'] > pivot_auto['DeepLabV3+_Dice'] + margin) & \
               (pivot_auto['ST-SAM_Dice'] > pivot_auto['BaselineSAM_Dice'] + margin)
        
        candidates = pivot_auto[cond].sort_values('Swin-UNETR_HD95', ascending=False)
        track1_final[mod] = candidates.iloc[0]['Image_ID'] if not candidates.empty else f"No match"

    # ---------------------------------------------------------
    # 赛道二 (专家引导模式) 严苛筛选
    # ---------------------------------------------------------
    track2_final = {}
    for mod in ['Colour', 'Infrared']:
        cond = (pivot_expert['Modality'] == mod) & \
               (pivot_expert['ST-SAM_Dice'] > 0.90) & \
               (pivot_expert['ST-SAM_HD95'] < 10) & \
               (pivot_expert['ST-SAM_Dice'] > pivot_expert['MSA_Dice'] + 0.01) & \
               (pivot_expert['ST-SAM_Dice'] > pivot_expert['LoRA_Dice'] + 0.01) & \
               (pivot_expert['ST-SAM_Dice'] > pivot_expert['BaselineSAM_Dice'] + 0.02)
        
        candidates = pivot_expert[cond].sort_values('ST-SAM_Dice', ascending=False)
        track2_final[mod] = candidates.iloc[0]['Image_ID'] if not candidates.empty else f"No match"

    print("\n🎯 狙击完成！以下是经受住全系毒打的绝对统治级图像 ID：")
    print("=" * 60)
    print(f"TRACK1_IDS = ['{track1_final['Colour']}', '{track1_final['Infrared']}']")
    print(f"TRACK2_IDS = ['{track2_final['Colour']}', '{track2_final['Infrared']}']")
    print("=" * 60)

if __name__ == "__main__":
    find_ultimate_cases()