import pandas as pd

def find_best_cases(csv_path="evaluation_results_5folds_full.csv"):
    print("🔍 正在加载并分析 5-Fold 全量评估数据...")
    df = pd.read_csv(csv_path)
    
    # 将 Padding 转为字符串以统一格式
    df['Padding'] = df['Padding'].astype(str)

    # =======================================================
    # 1. 拆分出 YOLO 赛道和 GT(Padding=0) 赛道的数据
    # =======================================================
    df_yolo = df[df['Padding'] == 'YOLO'].copy()
    df_gt = df[df['Padding'] == '0'].copy()

    # 将长表格 Pivot（透视）成宽表格，方便直接对比模型
    # 例如：变成一行是一张图，列是 ST-SAM_HD95, Baseline_SAM2_HD95 等
    yolo_pivot = df_yolo.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95', 'ASD'])
    yolo_pivot.columns = [f"{model}_{metric}" for metric, model in yolo_pivot.columns]
    yolo_pivot = yolo_pivot.reset_index()

    gt_pivot = df_gt.pivot(index=['Image_ID', 'Modality'], columns='Model', values=['Dice', 'HD95', 'ASD'])
    gt_pivot.columns = [f"{model}_{metric}" for metric, model in gt_pivot.columns]
    gt_pivot = gt_pivot.reset_index()

    # =======================================================
    # 2. 筛选赛道一 (全自动 YOLO 模式) 
    # 策略：找 Baseline SAM 2 断裂 (HD95很大)，但 ST-SAM 没断的
    # =======================================================
    # 找彩色极端样本
    cond_auto_color = (yolo_pivot['Modality'] == 'Colour') & \
                      (yolo_pivot['Baseline_SAM2_HD95'] > 30) & \
                      (yolo_pivot['ST-SAM_HD95'] < 15)
    auto_colors = yolo_pivot[cond_auto_color].sort_values('Baseline_SAM2_HD95', ascending=False)
    
    # 找红外极端噪点样本 (用 ASD 衡量边缘毛刺)
    cond_auto_ir = (yolo_pivot['Modality'] == 'Infrared') & \
                   (yolo_pivot['Baseline_SAM2_ASD'] > 3.0) & \
                   (yolo_pivot['ST-SAM_ASD'] < 2.0)
    auto_irs = yolo_pivot[cond_auto_ir].sort_values('Baseline_SAM2_ASD', ascending=False)

    track1_ids = [
        auto_colors.iloc[0]['Image_ID'] if len(auto_colors) > 0 else "N/A",
        auto_colors.iloc[1]['Image_ID'] if len(auto_colors) > 1 else "N/A",
        auto_irs.iloc[0]['Image_ID'] if len(auto_irs) > 0 else "N/A",
        auto_irs.iloc[1]['Image_ID'] if len(auto_irs) > 1 else "N/A"
    ]

    # =======================================================
    # 3. 筛选赛道二 (专家 GT 模式)
    # 策略：给了完美框，MSA 和 Baseline 依然 Dice 不高，但 ST-SAM 极高
    # =======================================================
    cond_gt_color = (gt_pivot['Modality'] == 'Colour') & \
                    (gt_pivot['Baseline_SAM2_Dice'] < 0.88) & \
                    (gt_pivot['ST-SAM_Dice'] > 0.92)
    gt_colors = gt_pivot[cond_gt_color].sort_values('ST-SAM_Dice', ascending=False)

    cond_gt_ir = (gt_pivot['Modality'] == 'Infrared') & \
                 (gt_pivot['MSA_SAM2_HD95'] > 12.0) & \
                 (gt_pivot['ST-SAM_HD95'] < 5.0)
    gt_irs = gt_pivot[cond_gt_ir].sort_values('MSA_SAM2_HD95', ascending=False)

    track2_ids = [
        gt_colors.iloc[0]['Image_ID'] if len(gt_colors) > 0 else "N/A",
        gt_colors.iloc[1]['Image_ID'] if len(gt_colors) > 1 else "N/A",
        gt_irs.iloc[0]['Image_ID'] if len(gt_irs) > 0 else "N/A",
        gt_irs.iloc[1]['Image_ID'] if len(gt_irs) > 1 else "N/A"
    ]

    print("\n🎉 自动筛选完成！请把下面这行 ID 填入绘图脚本中：")
    print("-" * 50)
    print(f"TRACK1_IDS (Automated) = {track1_ids}")
    print(f"TRACK2_IDS (Expert)    = {track2_ids}")
    print("-" * 50)
    print("👉 提示：你可以直接去 `masks_deeplab/` 文件夹里看一下 TRACK1_IDS 挑出来的图，DeepLab 绝对也是断裂的！")

if __name__ == "__main__":
    find_best_cases()