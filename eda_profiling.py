"""Step 1: EDA — 数据诊断与核心特征提取"""
import pandas as pd
import numpy as np

df = pd.read_csv('evaluation_results_5folds_full.csv')
yolo = df[(df['Padding'] == 'YOLO') & (df['Dice'] > 0)].copy()

MODEL_ORDER = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']

# ── 1. Dice 分位数 ────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Dice 分位数 (YOLO, Dice>0)")
print("=" * 60)
dice_q = (yolo.groupby('Model')['Dice']
              .quantile([0.25, 0.50, 0.75, 0.90])
              .unstack()
              .loc[MODEL_ORDER])
dice_q.columns = ['Q25', 'Q50(median)', 'Q75', 'Q90']
print(dice_q.round(4).to_string())

# ── 2. ASD / HD95 统计 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ASD 统计 (mean / median / Q90)")
print("=" * 60)
asd_stats = yolo.groupby('Model')['ASD'].agg(
    mean='mean', median='median',
    Q90=lambda x: x.quantile(0.90)
).loc[MODEL_ORDER]
print(asd_stats.round(3).to_string())

print("\n" + "=" * 60)
print("2b. HD95 统计 (mean / median / Q90)")
print("=" * 60)
hd_stats = yolo.groupby('Model')['HD95'].agg(
    mean='mean', median='median',
    Q90=lambda x: x.quantile(0.90)
).loc[MODEL_ORDER]
print(hd_stats.round(3).to_string())

# ── 3. 困难样本 Delta 分析 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 困难样本 (Baseline Dice<0.75 OR ASD>10) 上的 ST-SAM 挽救量")
print("=" * 60)

base = yolo[yolo['Model'] == 'Baseline_SAM2'][['Image_ID', 'Dice', 'ASD']].rename(
    columns={'Dice': 'Base_Dice', 'ASD': 'Base_ASD'})
stsam = yolo[yolo['Model'] == 'ST-SAM'][['Image_ID', 'Dice', 'ASD']].rename(
    columns={'Dice': 'ST_Dice', 'ASD': 'ST_ASD'})

paired = base.merge(stsam, on='Image_ID')
hard = paired[(paired['Base_Dice'] < 0.75) | (paired['Base_ASD'] > 10.0)].copy()
hard['Delta_Dice'] = hard['ST_Dice'] - hard['Base_Dice']
hard['Delta_ASD']  = hard['Base_ASD'] - hard['ST_ASD']   # 正值 = ASD 降低 = 好

print(f"困难样本数量: {len(hard)}")
print(f"  Delta_Dice  mean={hard['Delta_Dice'].mean():.4f}  "
      f"median={hard['Delta_Dice'].median():.4f}  "
      f"Q90={hard['Delta_Dice'].quantile(0.90):.4f}")
print(f"  Delta_ASD   mean={hard['Delta_ASD'].mean():.4f}  "
      f"median={hard['Delta_ASD'].median():.4f}  "
      f"Q90={hard['Delta_ASD'].quantile(0.90):.4f}")
print(f"  ST-SAM 在困难样本上 Dice>Baseline 的比例: "
      f"{(hard['Delta_Dice']>0).mean()*100:.1f}%")

# ── 4. 为 Step 2 提供坐标轴建议 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 坐标轴裁剪建议 (95% 数据范围)")
print("=" * 60)
dice_lo = yolo['Dice'].quantile(0.02)
dice_hi = yolo['Dice'].quantile(0.98)
asd_lo  = yolo['ASD'].quantile(0.02)
asd_hi  = yolo['ASD'].quantile(0.95)
print(f"  Dice xlim: [{dice_lo:.3f}, {dice_hi:.3f}]")
print(f"  ASD  xlim: [{asd_lo:.3f}, {asd_hi:.3f}]")

# ── 5. 困难子集 (Baseline_Dice < 0.80) 各模型 Dice 分布 ──────────────────────
print("\n" + "=" * 60)
print("5. 困难子集 (Baseline_Dice<0.80) 各模型 Dice 分布")
print("=" * 60)
hard_ids = paired[paired['Base_Dice'] < 0.80]['Image_ID']
hard_sub = yolo[yolo['Image_ID'].isin(hard_ids)]
print(f"困难子集样本数 (per model): {hard_sub.groupby('Model').size().to_dict()}")
hq = (hard_sub.groupby('Model')['Dice']
               .quantile([0.25, 0.50, 0.75])
               .unstack()
               .loc[MODEL_ORDER])
hq.columns = ['Q25', 'Q50', 'Q75']
print(hq.round(4).to_string())
