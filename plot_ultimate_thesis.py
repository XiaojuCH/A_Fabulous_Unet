import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# ================= 图 1：SOTA 模态分组箱线图 (无散点，极简干净版) =================
def plot_sota_clean_boxplot(df):
    print("1/3 正在绘制图 1：极简模态箱线图 (隐藏散点, 保留 5-95 胡须)...")
    df_yolo = df[df['Padding'] == 'YOLO'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    box_props = {
        'x': 'Model', 'hue': 'Modality', 'data': df_yolo, 
        'palette': MODALITY_COLORS, 'order': MODEL_ORDER,
        'width': 0.65, 
        'linewidth': 1.5, 
        'whis': (5, 95),             # 胡须伸展到 5% 和 95%，代表分布范围
        'showfliers': False,         # 【核心修改】：彻底关闭散点，画面极致干净
        'boxprops': {'alpha': 0.8, 'edgecolor': '#222222'},
        'medianprops': {'color': 'black', 'linewidth': 2.0}, 
        'whiskerprops': {'color': '#222222', 'linewidth': 1.5, 'linestyle': '-'}, # 改为实线胡须，更传统
        'capprops': {'color': '#222222', 'linewidth': 1.5}
    }

    # 左侧：Dice
    sns.boxplot(y='Dice', ax=axes[0], **box_props)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Region Overlap under Automated YOLO Prompts", fontweight='bold', pad=15)
    axes[0].set_ylabel("Dice Score (Higher is Better)")
    axes[0].set_xlabel("")

    # 右侧：HD95 (对数)
    sns.boxplot(y='HD95', ax=axes[1], **box_props)
    axes[1].set_yscale('log') 
    axes[1].set_title("Boundary Drift under Automated YOLO Prompts", fontweight='bold', pad=15)
    axes[1].set_ylabel("HD95 Error (Pixels, Log Scale)", labelpad=10)
    axes[1].set_xlabel("")

    for i, ax in enumerate(axes):
        ax.grid(True, axis='y', linestyle=':', alpha=0.6, color='gray') 
        sns.despine(ax=ax) 
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles=handles, labels=labels, title='', loc='center right', 
                       bbox_to_anchor=(0.98, 0.5), frameon=True, edgecolor='black')
        if ax.get_legend(): ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 0.88, 1]) 
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_1_Clean_Boxplot.pdf'), bbox_inches='tight')
    plt.close()

# ================= 图 2：Dice vs ASD 散点回归图 (融合自 GPT 代码) =================
def plot_scatter_regression(df):
    print("2/3 正在绘制图 2：Dice-ASD 相关性散点分析...")
    # 只取 YOLO 数据，展示真实场景下的相关性
    df_yolo = df[df['Padding'] == 'YOLO'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes = axes.ravel()

    # 统一坐标轴范围
    x_max = np.nanpercentile(df_yolo["ASD"], 98) * 1.05
    
    for ax, model in zip(axes, MODEL_ORDER):
        sub = df_yolo[df_yolo["Model"] == model][["ASD", "Dice"]].dropna()
        if len(sub) == 0: continue

        # 为防止点过于密集，降采样到最多 1500 个点用于展示
        if len(sub) > 1500: sub = sub.sample(1500, random_state=42)

        xx = sub["ASD"].to_numpy(dtype=float)
        yy = sub["Dice"].to_numpy(dtype=float)

        # 绘制半透明散点
        ax.scatter(xx, yy, s=10, alpha=0.25, color=MODEL_COLORS[model], edgecolors="none")

        # 绘制回归线
        xs, ys = linear_fit(xx, yy)
        if len(xs):
            ax.plot(xs, ys, color="#111111", linewidth=2.0, zorder=4)

        # 标注 R方 和 N
        r, r2, n = safe_corr(sub["ASD"], sub["Dice"])
        ax.text(
            0.95, 0.95, f"r = {r:.3f}\n$R^2$ = {r2:.3f}\nN = {n}",
            transform=ax.transAxes, va="top", ha="right", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9)
        )

        ax.set_title(model, pad=10, fontweight="bold")
        ax.set_xlabel("Average Surface Distance (ASD)")
        ax.set_ylabel("Dice Score")
        ax.set_xlim(0, max(10, x_max))
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle=':', alpha=0.5)
        sns.despine(ax=ax)

    fig.suptitle("Correlation: Region Overlap vs. Boundary Smoothness", fontsize=16, fontweight="bold")
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
            
            ax.errorbar(
                x, y, yerr=yerr, fmt='-o', color=MODEL_COLORS[model], 
                linewidth=lw, markersize=6, capsize=4, capthick=1.2, 
                alpha=alpha, label=model, zorder=z
            )
            
        # 绘制显眼的 YOLO 基准线
        for model in MODEL_ORDER:
            mean_val = df_yolo[df_yolo['Model'] == model][metric].mean()
            color = MODEL_COLORS[model]
            
            if model == 'ST-SAM':
                ax.axhline(y=mean_val, color=color, linestyle='--', linewidth=2.5, zorder=2)
                bbox_props = dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9)
                ax.text(41.5, mean_val, 'Auto (YOLO)', color='white', va='center', ha='left', 
                        fontsize=10, fontweight='bold', bbox=bbox_props, zorder=10)
            else:
                ax.axhline(y=mean_val, color=color, linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)

        ax.set_xticks(xticks)
        ax.set_xlim(-7, 51)
        ax.set_title(f'Robustness of {metric} to Box Expansion', fontweight='bold', pad=15)
        ax.set_xlabel('Box Expansion / Padding (Linear Pixels)')
        ax.set_ylabel(f'{metric} Score (↑)' if metric == 'Dice' else f'{metric} Error (↓, Log Scale)')
        
        if metric == 'HD95': ax.set_yscale('log')
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        sns.despine(ax=ax)
        ax.legend(loc='best', frameon=False)

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