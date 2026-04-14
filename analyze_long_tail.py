import pandas as pd

def main():
    csv_file = "evaluation_results_5folds_full.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"找不到 {csv_file}！")
        return

    # 只分析真实临床的 YOLO 场景
    df_yolo = df[df['Padding'] == 'YOLO'].copy()
    models = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']

    print("="*60)
    print("🚀 YOLO 真实场景下 HD95 核心防线与长尾分布分析")
    print("="*60)

    for model in models:
        data = df_yolo[df_yolo['Model'] == model]['HD95']
        
        mean_val = data.mean()
        median_val = data.median()
        q3_val = data.quantile(0.75)
        p90_val = data.quantile(0.90)
        p95_val = data.quantile(0.95)
        max_val = data.max()
        
        # 统计灾难性医疗事故（例如边界误差超过 100 和 500 像素的样本数）
        fail_100 = (data > 100).sum()
        fail_500 = (data > 500).sum()
        total_samples = len(data)

        print(f"\n[{model}] (样本总数: {total_samples})")
        print(f"  ▶ 均值 (Mean)          : {mean_val:.2f} 像素")
        print(f"  ▶ 中位数 (Median)      : {median_val:.2f} 像素 (大家水平接近)")
        print(f"  ▶ 75分位数 (Q3箱体顶端): {q3_val:.2f} 像素")
        print(f"  🔥 90分位数 (核心长尾) : {p90_val:.2f} 像素 (决胜点)")
        print(f"  🔥 95分位数 (极限防线) : {p95_val:.2f} 像素 (决胜点)")
        print(f"  ▶ 最大误差 (Max)       : {max_val:.2f} 像素")
        print(f"  🚨 灾难崩溃 (误差>100) : {fail_100} 例 (占比 {fail_100/total_samples*100:.1f}%)")
        print(f"  🚨 彻底漏割 (误差>500) : {fail_500} 例 (占比 {fail_500/total_samples*100:.1f}%)")

if __name__ == "__main__":
    main()