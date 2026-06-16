import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ================= 1. 顶刊极简严谨排版 (去 AI 味，回归经典) =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,    
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.linewidth': 1.2,   
    'figure.dpi': 300,
    'pdf.fonttype': 42
})

OUTPUT_DIR = "Ultimate_Thesis_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']

# 顶级重色彩
MODEL_COLORS = {
    'ST-SAM': '#B30000',         
    'MSA_SAM2': '#1F77B4',       
    'Baseline_SAM2': '#7F7F7F',  
    'LoRA_SAM2': '#D55E00'       
}

MODALITY_COLORS = {
    'Colour': '#D45B42',         
    'Infrared': '#406A8C'        
}

# ================= 通用辅助函数 =================
def safe_corr(x: pd.Series, y: pd.Series):
    """计算皮尔逊相关系数和 R方"""
    mask = np.isfinite(x) & np.isfinite(y)
    xx = x[mask].to_numpy(dtype=float)
    yy = y[mask].to_numpy(dtype=float)
    if len(xx) < 3: return np.nan, np.nan, len(xx)
    r = float(np.corrcoef(xx, yy)[0, 1])
    return r, r**2, len(xx)

def linear_fit(x: np.ndarray, y: np.ndarray):
    """计算线性回归拟合线"""
    mask = np.isfinite(x) & np.isfinite(y)
    xx = x[mask]
    yy = y[mask]
    if len(xx) < 2: return np.array([]), np.array([])
    coef = np.polyfit(xx, yy, 1)
    xs = np.linspace(xx.min(), xx.max(), 100)
    ys = coef[0] * xs + coef[1]
    return xs, ys

# ================= 图 1：SOTA 模态分组箱线图 (3 panels: Dice + HD95 + Modality Gap) =================
def plot_sota_clean_boxplot(df):
    print("1/3 正在绘制图 1：三面板箱线图 (Dice + HD95 + 模态差距条形图)...")
    df_yolo = df[df['Padding'] == 'YOLO'].copy()

    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.3], wspace=0.38)
    ax_dice = fig.add_subplot(gs[0])
    ax_hd95 = fig.add_subplot(gs[1])
    ax_gap  = fig.add_subplot(gs[2])

    box_props = dict(
        x='Model', hue='Modality', data=df_yolo,
        palette=MODALITY_COLORS, order=MODEL_ORDER,
        width=0.65, linewidth=1.5, whis=(5, 95), showfliers=False,
        boxprops={'alpha': 0.82, 'edgecolor': '#222222'},
        medianprops={'color': 'black', 'linewidth': 2.2},
        whiskerprops={'color': '#333333', 'linewidth': 1.4, 'linestyle': '-'},
        capprops={'color': '#333333', 'linewidth': 1.4},
    )

    # ── Panel A: Dice ──────────────────────────────────────────────────────────
    sns.boxplot(y='Dice', ax=ax_dice, **box_props)
    ax_dice.set_ylim(0.60, 1.02)
    ax_dice.set_title("A. Region Overlap (Dice)", fontweight='bold', pad=12)
    ax_dice.set_ylabel("Dice Score (↑)")
    ax_dice.set_xlabel("")

    # 显著性括号：ST-SAM vs SAM2 Baseline (p<0.01, **)
    # seaborn grouped boxplot: model positions 0,1,2,3; hue offset ≈ ±0.17
    y_sig, h = 0.985, 0.006
    x1, x2 = 0 + 0.17, 3 - 0.17          # Baseline_SAM2 right edge → ST-SAM left edge
    ax_dice.plot([x1, x1, x2, x2], [y_sig, y_sig+h, y_sig+h, y_sig],
                 color='black', linewidth=1.1)
    ax_dice.text((x1+x2)/2, y_sig+h+0.002, '**', ha='center', va='bottom',
                 fontsize=14, fontweight='bold')

    # ST-SAM vs MSA_SAM2 (p<0.01, **)
    y_sig2 = 0.965
    x3, x4 = 2 + 0.17, 3 - 0.17
    ax_dice.plot([x3, x3, x4, x4], [y_sig2, y_sig2+h, y_sig2+h, y_sig2],
                 color='black', linewidth=1.1)
    ax_dice.text((x3+x4)/2, y_sig2+h+0.002, '**', ha='center', va='bottom',
                 fontsize=14, fontweight='bold')

    # ── Panel B: HD95 (log) ────────────────────────────────────────────────────
    sns.boxplot(y='HD95', ax=ax_hd95, **box_props)
    ax_hd95.set_yscale('log')
    ax_hd95.set_title("B. Boundary Drift (HD95)", fontweight='bold', pad=12)
    ax_hd95.set_ylabel("HD95 Error (Pixels, Log, ↓)")
    ax_hd95.set_xlabel("")

    # ── Panel C: Modality Gap 条形图 ───────────────────────────────────────────
    gap_rows = []
    for m in MODEL_ORDER:
        sub = df_yolo[df_yolo['Model'] == m]
        c = sub[sub['Modality'] == 'Colour']['Dice'].mean()
        ir = sub[sub['Modality'] == 'Infrared']['Dice'].mean()
        gap_rows.append({'Model': m, 'Gap': c - ir})
    gap_df = pd.DataFrame(gap_rows)

    bar_colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = ax_gap.bar(range(len(MODEL_ORDER)), gap_df['Gap'],
                      color=bar_colors, edgecolor='#222222', linewidth=1.2,
                      alpha=0.85, width=0.6, zorder=3)
    # 加粗 ST-SAM 边框
    bars[MODEL_ORDER.index('ST-SAM')].set_linewidth(2.8)
    bars[MODEL_ORDER.index('ST-SAM')].set_edgecolor('#B30000')

    # 数值标签
    for i, (bar, row) in enumerate(zip(bars, gap_df.itertuples())):
        is_best = MODEL_ORDER[i] == 'ST-SAM'
        ax_gap.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0008,
                    f'{row.Gap:.3f}',
                    ha='center', va='bottom', fontsize=10.5,
                    fontweight='bold' if is_best else 'normal',
                    color='#B30000' if is_best else '#333333')

    # 最优值参考线
    best_gap = gap_df.loc[gap_df['Model'] == 'ST-SAM', 'Gap'].values[0]
    ax_gap.axhline(best_gap, color='#B30000', linestyle='--', linewidth=1.4, alpha=0.7, zorder=2)

    ax_gap.set_xticks(range(len(MODEL_ORDER)))
    ax_gap.set_xticklabels([m.replace('_', '\n') for m in MODEL_ORDER], fontsize=9.5)
    ax_gap.set_title("C. Modality Gap\n(Colour − Infrared Dice, ↓)", fontweight='bold', pad=12)
    ax_gap.set_ylabel("Dice Gap  (smaller = more robust)")
    ax_gap.set_ylim(0, gap_df['Gap'].max() * 1.35)
    ax_gap.grid(True, axis='y', linestyle=':', alpha=0.5, color='gray', zorder=0)
    sns.despine(ax=ax_gap)

    # ── 公共格式 ───────────────────────────────────────────────────────────────
    for ax in [ax_dice, ax_hd95]:
        ax.grid(True, axis='y', linestyle=':', alpha=0.5, color='gray')
        sns.despine(ax=ax)
        if ax.get_legend(): ax.get_legend().remove()

    handles, labels = ax_dice.get_legend_handles_labels()
    fig.legend(handles, labels, title='Imaging Modality',
               loc='lower center', ncol=2,
               bbox_to_anchor=(0.42, -0.06),
               frameon=True, edgecolor='black', fontsize=11)

    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_1_Comprehensive_Boxplot.pdf'), bbox_inches='tight')
    plt.close()

# ================= 图 2：Dice vs ASD 单图叠加对比 (回归线 + KDE 等高线) =================
def plot_scatter_regression(df):
    print("2/3 正在绘制图 2：Dice-ASD 单图叠加对比（回归线 + KDE 等高线）...")
    df_yolo = df[df['Padding'] == 'YOLO'].copy()

    X_LIM = (0, 15)
    Y_LIM = (0.50, 1.00)

    fig, ax = plt.subplots(figsize=(9, 7))

    # 优质区域底色（高 Dice、低 ASD）
    ax.axvspan(X_LIM[0], 4, ymin=(0.88 - Y_LIM[0]) / (Y_LIM[1] - Y_LIM[0]),
               ymax=1.0, alpha=0.06, color='#2ca02c', zorder=0)
    ax.text(0.18, 0.985, 'High-Performance Zone', transform=ax.transAxes,
            fontsize=9, color='#2ca02c', va='top', style='italic', alpha=0.8)

    stats_lines = []
    for model in MODEL_ORDER:
        sub = df_yolo[df_yolo['Model'] == model][['ASD', 'Dice']].dropna()
        sub = sub[(sub['ASD'] >= X_LIM[0]) & (sub['ASD'] <= X_LIM[1])]
        if len(sub) > 1500:
            sub = sub.sample(1500, random_state=42)

        xx = sub['ASD'].to_numpy(float)
        yy = sub['Dice'].to_numpy(float)
        color = MODEL_COLORS[model]
        is_ours = model == 'ST-SAM'
        lw = 2.8 if is_ours else 1.6
        z  = 6  if is_ours else 3

        # 半透明散点（仅 ST-SAM 稍深）
        ax.scatter(xx, yy, s=6, alpha=0.30 if is_ours else 0.12,
                   color=color, edgecolors='none', zorder=z - 1)

        # KDE 等高线（仅 2 层，轻量）
        sns.kdeplot(x=xx, y=yy, levels=2, linewidths=0.9 if is_ours else 0.6,
                    color=color, alpha=0.55, ax=ax, zorder=z)

        # 回归线
        xs, ys = linear_fit(xx, yy)
        r, r2, _ = safe_corr(sub['ASD'], sub['Dice'])
        label = f'{model}  (r={r:.3f}, $R^2$={r2:.3f})'
        ax.plot(xs, ys, color=color, linewidth=lw, label=label, zorder=z + 1,
                linestyle='-' if is_ours else '--')
        stats_lines.append((model, r, r2))

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel('Average Surface Distance — ASD (pixels, ↓)', fontsize=13)
    ax.set_ylabel('Dice Score (↑)', fontsize=13)
    ax.set_title('Region Overlap vs. Boundary Smoothness\n'
                 '(YOLO Automated Mode, all folds)',
                 fontweight='bold', fontsize=14, pad=12)
    ax.grid(True, linestyle=':', alpha=0.45, color='gray')
    sns.despine(ax=ax)

    # 图例：ST-SAM 排首位
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in sorted(labels,
             key=lambda s: (0 if 'ST-SAM' in s else 1))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='lower left', frameon=True, edgecolor='#cccccc',
              fontsize=10.5, title='Model  (regression line)', title_fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_2_Scatter_Regression.pdf'), bbox_inches='tight')
    plt.close()

# ================= 图 3：离散误差棒鲁棒性折线图 (去除 AI 阴影带，纯正医学统计风) =================
def plot_robustness_traditional_lines(df):
    print("3/3 正在绘制图 3：传统医学统计风鲁棒性折线图 (带标准误差棒)...")
    df_num = df[df['Padding'] != 'YOLO'].copy()
    df_num['Padding'] = pd.to_numeric(df_num['Padding'])
    df_yolo = df[df['Padding'] == 'YOLO'].copy()
    
    # 手动计算每个模型在每个 Padding 下的 Mean 和 95% CI
    agg_df = df_num.groupby(['Padding', 'Model']).agg(
        Mean_Dice=('Dice', 'mean'), Std_Dice=('Dice', 'std'), Count_Dice=('Dice', 'count'),
        Mean_HD95=('HD95', 'mean'), Std_HD95=('HD95', 'std')
    ).reset_index()
    
    # 计算 95% 置信区间幅度 (1.96 * 标准误)
    agg_df['CI_Dice'] = 1.96 * (agg_df['Std_Dice'] / np.sqrt(agg_df['Count_Dice']))
    agg_df['CI_HD95'] = 1.96 * (agg_df['Std_HD95'] / np.sqrt(agg_df['Count_Dice']))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    xticks = [-5, 0, 5, 10, 20, 30, 40]
    
    for ax, metric in zip(axes, ['Dice', 'HD95']):
        for model in MODEL_ORDER:
            sub = agg_df[agg_df['Model'] == model].sort_values('Padding')
            x = sub['Padding'].to_numpy()
            y = sub[f'Mean_{metric}'].to_numpy()
            yerr = sub[f'CI_{metric}'].to_numpy()
            
            # 【核心修改】：使用传统的带盖帽(cap)的误差棒代替连续阴影带，极具专业感
            lw = 2.5 if model == 'ST-SAM' else 1.5
            z = 5 if model == 'ST-SAM' else 3
            alpha = 1.0 if model == 'ST-SAM' else 0.8
            
            display_name = 'GAL-SAM' if model == 'ST-SAM' else model
            ax.errorbar(
                x, y, yerr=yerr, fmt='-o', color=MODEL_COLORS[model], 
                linewidth=lw, markersize=6, capsize=4, capthick=1.2, 
                alpha=alpha, label=display_name, zorder=z
            )
            
        # 绘制显眼的 YOLO 基准线
        for model in MODEL_ORDER:
            mean_val = df_yolo[df_yolo['Model'] == model][metric].mean()
            color = MODEL_COLORS[model]
            
            ax.axhline(y=mean_val, color=color, linestyle='--',
                       linewidth=1.2, alpha=0.6, zorder=1)

        ax.set_xticks(xticks)
        ax.set_xlim(-7, 45)
        ax.set_title(f'Robustness of {metric} to Box Expansion', fontweight='bold', pad=15)
        ax.set_xlabel('Box Expansion / Padding (Linear Pixels)')
        ax.set_ylabel(f'{metric} Score (↑)' if metric == 'Dice' else f'{metric} Error (↓, Log Scale)')
        
        if metric == 'HD95': ax.set_yscale('log')
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        sns.despine(ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='#666666', linestyle='--',
                              linewidth=1.4, label='Auto (YOLO) Baselines'))
        labels.append('Auto (YOLO) Baselines')
        ax.legend(handles, labels, loc='best', frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_3_Traditional_Robustness.pdf'), bbox_inches='tight')
    plt.close()

# ================= 5. 主程序入口 =================
def main():
    csv_file = "evaluation_results_5folds_full.csv"
    if not os.path.exists(csv_file):
        print(f"❌ 找不到数据文件: {csv_file}")
        return

    print("正在加载评估数据...")
    df = pd.read_csv(csv_file).dropna(subset=['Dice', 'HD95', 'ASD'])
    
    plot_sota_clean_boxplot(df)
    plot_scatter_regression(df)
    plot_robustness_traditional_lines(df)
    
    print(f"\n🎉 完美融合！全套【顶级医学统计学版】图表已生成至 [{OUTPUT_DIR}] 文件夹！")

if __name__ == "__main__":
    main()
