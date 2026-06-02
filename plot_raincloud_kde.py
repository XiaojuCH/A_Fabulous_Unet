"""
plot_raincloud_kde.py
两张顶刊级核心对比图：
  Fig_A_Raincloud.png  —— 雨云图（Dice Score 分布）
  Fig_B_KDE_Contour.png —— 2D KDE 等高线图（ASD vs Dice）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy.stats import gaussian_kde

# ── 全局排版 ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
})

OUTPUT_DIR = "Raincloud_KDE_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER  = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']
MODEL_LABELS = ['Baseline SAM2', 'LoRA SAM2', 'MSA SAM2', 'ST-SAM']

# NPG-inspired palette（Nature Publishing Group 风格）
COLORS = {
    'Baseline_SAM2': '#7F7F7F',
    'LoRA_SAM2':     '#D55E00',
    'MSA_SAM2':      '#1F77B4',
    'ST-SAM':        '#B30000',
}

# ── 真实数据 ──────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

_raw = pd.read_csv('evaluation_results_5folds_full.csv')
# 只取 YOLO 自动提示场景（与原脚本一致），去除 Dice=0 的失败样本
df = _raw[(_raw['Padding'] == 'YOLO') & (_raw['Dice'] > 0)][['Model', 'Dice', 'ASD']].copy()
# ASD 截断 99 百分位，避免极端离群值压缩分布
asd_cap = df['ASD'].quantile(0.99)
df['ASD'] = df['ASD'].clip(upper=asd_cap)

# ── 图 A：雨云图 ──────────────────────────────────────────────────────────────
def plot_raincloud(df):
    fig, ax = plt.subplots(figsize=(9, 6))

    n_models = len(MODEL_ORDER)
    y_positions = np.arange(n_models)   # 每个模型的 Y 中心

    kde_height  = 0.30   # KDE 山峰最大高度（Y 方向）
    box_half    = 0.06   # Boxplot 半宽
    strip_offset = 0.14  # 散点相对中心的偏移（向下）

    for i, model in enumerate(MODEL_ORDER):
        y0    = y_positions[i]
        color = COLORS[model]
        data  = df.loc[df['Model'] == model, 'Dice'].values

        # 1) KDE 山峰（上方）
        kde   = gaussian_kde(data, bw_method=0.15)
        x_kde = np.linspace(data.min(), data.max(), 300)
        k_val = kde(x_kde)
        k_val = k_val / k_val.max() * kde_height
        ax.fill_between(x_kde, y0, y0 + k_val,
                        color=color, alpha=0.55, linewidth=0)
        ax.plot(x_kde, y0 + k_val, color=color, linewidth=1.2)

        # 2) 极窄 Boxplot（中间）
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr  = q3 - q1
        wlo  = max(data.min(), q1 - 1.5 * iqr)
        whi  = min(data.max(), q3 + 1.5 * iqr)
        # 箱体
        rect = mpatches.FancyBboxPatch(
            (q1, y0 - box_half), q3 - q1, 2 * box_half,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor='#222222', facecolor=color, alpha=0.75, zorder=3)
        ax.add_patch(rect)
        # 中位数线
        ax.plot([med, med], [y0 - box_half, y0 + box_half],
                color='white', linewidth=2.0, zorder=4)
        # 胡须
        ax.plot([wlo, q1], [y0, y0], color='#333333', linewidth=1.2, zorder=3)
        ax.plot([q3, whi], [y0, y0], color='#333333', linewidth=1.2, zorder=3)
        ax.plot([wlo, wlo], [y0 - box_half*0.6, y0 + box_half*0.6],
                color='#333333', linewidth=1.2, zorder=3)
        ax.plot([whi, whi], [y0 - box_half*0.6, y0 + box_half*0.6],
                color='#333333', linewidth=1.2, zorder=3)

        # 3) Jitter 散点（下方）
        jitter = rng.uniform(-0.06, 0.06, len(data))
        ax.scatter(data, y0 - strip_offset + jitter,
                   s=4, color=color, alpha=0.20,
                   edgecolors='none', zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(MODEL_LABELS, fontsize=12)
    ax.set_xlabel('Dice Score  (↑ Higher is Better)', fontsize=14)
    ax.set_xlim(0.50, 1.02)
    ax.set_ylim(-0.55, n_models - 0.35)

    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='gray')
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig_A_Raincloud.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Raincloud saved: {out}")


# ── 图 B：2D KDE 等高线图 ─────────────────────────────────────────────────────
def plot_kde_contour(df):
    fig, ax = plt.subplots(figsize=(7, 6))

    levels = [0.20, 0.50, 0.80]   # 对应 20% / 50% / 80% 密度圈（取最核心 2-3 层）

    handles = []
    for model in MODEL_ORDER:
        sub   = df[df['Model'] == model]
        color = COLORS[model]
        label = model.replace('_', ' ')

        sns.kdeplot(
            data=sub, x='ASD', y='Dice',
            levels=levels,
            color=color, linewidths=2.0,
            linestyles=['-', '--', ':'],
            ax=ax, zorder=3,
        )
        handles.append(mpatches.Patch(color=color, label=label))

    ax.set_xlabel('Average Surface Distance — ASD  (↓ Lower is Better)', fontsize=14)
    ax.set_ylabel('Dice Score  (↑ Higher is Better)', fontsize=14)

    ax.grid(True, axis='both', linestyle='--', alpha=0.3, color='gray')
    sns.despine(ax=ax)

    ax.legend(handles=handles, fontsize=12, frameon=True,
              edgecolor='#cccccc', loc='lower left')

    # 注释：ST-SAM 核心圈位置
    ax.annotate('ST-SAM\n(tight cluster,\nhigh Dice / low ASD)',
                xy=(3.2, 0.93), xytext=(8, 0.78),
                fontsize=10, color=COLORS['ST-SAM'],
                arrowprops=dict(arrowstyle='->', color=COLORS['ST-SAM'],
                                lw=1.4, connectionstyle='arc3,rad=0.2'))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig_B_KDE_Contour.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] KDE contour saved: {out}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_raincloud(df)
    plot_kde_contour(df)
    print(f"\n[Done] Both figures saved to [{OUTPUT_DIR}/]")
