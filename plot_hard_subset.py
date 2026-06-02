"""
plot_hard_subset.py
重构自 plot_ultimate_thesis.py
  Fig_Hard_A_Contour.png  —— 局部放大 2D KDE 等高线图（全量 YOLO）
  Fig_Hard_B_Raincloud.png —— 困难子集雨云图（Baseline Dice < 0.75）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde

# ── 保留原始排版与配色 ────────────────────────────────────────────────────────
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
    'pdf.fonttype': 42,
})

OUTPUT_DIR = "Ultimate_Thesis_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER  = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']
MODEL_LABELS = ['Baseline SAM2', 'LoRA SAM2', 'MSA SAM2', 'ST-SAM']

MODEL_COLORS = {
    'ST-SAM':        '#B30000',
    'MSA_SAM2':      '#1F77B4',
    'Baseline_SAM2': '#7F7F7F',
    'LoRA_SAM2':     '#D55E00',
}
MODALITY_COLORS = {'Colour': '#D45B42', 'Infrared': '#406A8C'}

rng = np.random.default_rng(42)


# ── 数据加载与困难子集提取 ────────────────────────────────────────────────────
def load_and_filter_data(csv_path):
    df = pd.read_csv(csv_path)
    df_yolo = df[(df['Padding'] == 'YOLO') & (df['Dice'] > 0)].copy()

    hard_ids = (df_yolo[df_yolo['Model'] == 'Baseline_SAM2']
                .loc[lambda x: x['Dice'] < 0.75, 'Image_ID'])
    df_hard = df_yolo[df_yolo['Image_ID'].isin(hard_ids)].copy()

    n = df_hard['Image_ID'].nunique()
    print(f"[Data] Hard subset: {n} unique images  "
          f"({len(df_hard)} rows across {df_hard['Model'].nunique()} models)")
    return df_yolo, df_hard


# ── 方案 A：局部放大 2D KDE 等高线图 ─────────────────────────────────────────
def plot_zoom_contour(df_yolo):
    fig, ax = plt.subplots(figsize=(7, 6))

    handles = []
    for model in MODEL_ORDER:
        sub   = df_yolo[df_yolo['Model'] == model]
        color = MODEL_COLORS[model]
        sns.kdeplot(
            data=sub, x='ASD', y='Dice',
            fill=True, levels=4, alpha=0.25, thresh=0.2,
            color=color,
            ax=ax, zorder=3,
        )
        handles.append(mpatches.Patch(
            color=color, label=model.replace('_', ' ')))

    ax.set_xlim(0, 20)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel('Average Surface Distance — ASD  (↓ Lower is Better)')
    ax.set_ylabel('Dice Score  (↑ Higher is Better)')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    sns.despine(ax=ax)
    ax.legend(handles=handles, frameon=True, edgecolor='#cccccc', loc='lower left')

    # 标注 ST-SAM 核心圈
    ax.annotate('ST-SAM\n(50% density core)',
                xy=(3.5, 0.885), xytext=(8.5, 0.82),
                fontsize=10, color=MODEL_COLORS['ST-SAM'],
                arrowprops=dict(arrowstyle='->', color=MODEL_COLORS['ST-SAM'],
                                lw=1.4, connectionstyle='arc3,rad=0.25'))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig_Hard_A_Contour.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Zoom contour saved: {out}")


# ── 方案 B：困难子集雨云图 ────────────────────────────────────────────────────
def _draw_raincloud(ax, df, metric, color_map, model_order, model_labels):
    """在单个 Axes 上绘制横向雨云图（KDE + 窄 Boxplot + Jitter）"""
    kde_h       = 0.45   # KDE 山峰最大高度
    box_half    = 0.055  # Boxplot 半宽
    strip_off   = 0.13   # 散点向下偏移

    for i, model in enumerate(model_order):
        data  = df.loc[df['Model'] == model, metric].dropna().values
        if len(data) < 5:
            continue
        color = color_map[model]

        # 1) KDE 山峰（上方）
        kde   = gaussian_kde(data, bw_method=0.18)
        x_kde = np.linspace(data.min(), data.max(), 300)
        k_val = kde(x_kde)
        k_val = k_val / k_val.max() * kde_h
        ax.fill_between(x_kde, i, i + k_val, color=color, alpha=0.50, linewidth=0)
        ax.plot(x_kde, i + k_val, color=color, linewidth=1.2)

        # 2) 极窄定制 Boxplot（中间）
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr  = q3 - q1
        wlo  = max(data.min(), q1 - 1.5 * iqr)
        whi  = min(data.max(), q3 + 1.5 * iqr)
        rect = mpatches.FancyBboxPatch(
            (q1, i - box_half), q3 - q1, 2 * box_half,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor='#222222', facecolor=color, alpha=0.80, zorder=3)
        ax.add_patch(rect)
        ax.plot([med, med], [i - box_half, i + box_half],
                color='white', linewidth=2.0, zorder=4)
        for wx in [wlo, whi]:
            ax.plot([wx, wx], [i - box_half * 0.6, i + box_half * 0.6],
                    color='#333333', linewidth=1.2, zorder=3)
        ax.plot([wlo, q1], [i, i], color='#333333', linewidth=1.2, zorder=3)
        ax.plot([q3, whi], [i, i], color='#333333', linewidth=1.2, zorder=3)

        # 3) Jitter 散点（下方）
        jitter = rng.uniform(-0.055, 0.055, len(data))
        ax.scatter(data, i - strip_off + jitter,
                   s=3, color=color, alpha=0.3, edgecolors='none', zorder=2)

    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_labels, fontsize=12)
    ax.set_ylim(-0.5, len(model_order) - 0.3)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='gray')
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)


def plot_hard_raincloud(df_hard):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    _draw_raincloud(axes[0], df_hard, 'Dice',
                    MODEL_COLORS, MODEL_ORDER, MODEL_LABELS)
    axes[0].set_xlabel('Dice Score  (↑ Higher is Better)')
    axes[0].set_xlim(0.3, 0.98)

    _draw_raincloud(axes[1], df_hard, 'HD95',
                    MODEL_COLORS, MODEL_ORDER, MODEL_LABELS)
    axes[1].set_xlabel('HD95 Error (Pixels, ↓ Lower is Better)')
    axes[1].set_xlim(-2, 100)
    axes[1].set_yticklabels([])   # 右图不重复 Y 轴标签

    # 共享标题
    n_img = df_hard['Image_ID'].nunique()
    fig.suptitle(
        f'Performance on Hard Cases  (Baseline Dice < 0.75,  N = {n_img} images)',
        fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig_Hard_B_Raincloud.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Hard-subset raincloud saved: {out}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_yolo, df_hard = load_and_filter_data('evaluation_results_5folds_full.csv')
    plot_zoom_contour(df_yolo)
    plot_hard_raincloud(df_hard)
    print(f"\n[Done] Figures saved to [{OUTPUT_DIR}/]")
