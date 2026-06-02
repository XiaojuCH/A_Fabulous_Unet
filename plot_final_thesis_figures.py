"""
plot_final_thesis_figures.py
论文最终定稿图表（原位替换图 2 & 图 3）
  Fig_2_Final_Boxplot.png   —— 全量 YOLO 极简模态分组箱线图（2x2）
  Fig_3_Final_KDE.png       —— 全量 2D KDE 靶心等高线 + 回归线
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── 排版与配色（保留原稿设定）────────────────────────────────────────────────
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

MODEL_ORDER = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']
MODEL_COLORS = {
    'ST-SAM':        '#B30000',
    'MSA_SAM2':      '#1F77B4',
    'Baseline_SAM2': '#7F7F7F',
    'LoRA_SAM2':     '#D55E00',
}
MODALITY_COLORS = {'Colour': '#D45B42', 'Infrared': '#406A8C'}


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df_yolo = df[(df['Padding'] == 'YOLO') & (df['Dice'] > 0)].copy()
    print(f"[Data] Full YOLO set: {len(df_yolo)} rows, "
          f"{df_yolo['Image_ID'].nunique()} unique images")
    return df_yolo


# ── 图 2：全量极简模态分组箱线图（2x2）────────────────────────────────────────
def plot_fig2_boxplot(df_yolo):
    metrics = [
        ('Dice',  'Dice Score (Higher is Better)',          (0.65, 0.98), False),
        ('HD95',  'HD95 Error (Pixels, Log Scale)',         None,         True),
        ('IoU',   'IoU Score (Higher is Better)',           (0.50, 0.90), False),
        ('ASD',   'Avg Surface Distance (Lower is Better)', (0, 15),      False),
    ]
    titles = [
        'Region Overlap (YOLO Prompts)',
        'Boundary Drift (YOLO Prompts)',
        'Intersection over Union (YOLO Prompts)',
        'Surface Distance (YOLO Prompts)',
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.ravel()

    box_props = dict(
        x='Model', hue='Modality', data=df_yolo,
        palette=MODALITY_COLORS, order=MODEL_ORDER,
        width=0.65, linewidth=1.5,
        whis=(5, 95), showfliers=False,
        boxprops={'alpha': 0.82, 'edgecolor': '#222222'},
        medianprops={'color': 'black', 'linewidth': 2.0},
        whiskerprops={'color': '#222222', 'linewidth': 1.5, 'linestyle': '-'},
        capprops={'color': '#222222', 'linewidth': 1.5},
    )

    for ax, (metric, ylabel, ylim, log_scale), title in zip(axes, metrics, titles):
        sns.boxplot(y=metric, ax=ax, **box_props)
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        if ylim:
            ax.set_ylim(*ylim)
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, axis='y', linestyle=':', alpha=0.4, color='gray')
        sns.despine(ax=ax)
        if ax.get_legend():
            ax.get_legend().remove()

    # 共享图例放右侧
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, title='Modality',
               loc='center right', bbox_to_anchor=(1.0, 0.5),
               frameon=True, edgecolor='black')

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    out = os.path.join(OUTPUT_DIR, 'Fig_2_Final_Boxplot.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Fig 2 saved: {out}")


# ── 图 3：2x2 散点 + KDE 等高线 + 回归线 ─────────────────────────────────────
def plot_fig3_kde(df_yolo):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, model in zip(axes, MODEL_ORDER):
        sub   = df_yolo[df_yolo['Model'] == model].copy()
        color = MODEL_COLORS[model]
        # clip to xlim range for KDE / regplot stability
        sub_clip = sub[sub['ASD'] < 20].copy()

        # 1) 底层散点（深灰背景）
        sns.scatterplot(data=sub_clip, x='ASD', y='Dice',
                        color='#555555', s=3, alpha=0.15,
                        edgecolor='none', ax=ax, zorder=1)

        # 2) 中层 KDE 等高线
        sns.kdeplot(data=sub_clip, x='ASD', y='Dice',
                    levels=5, linewidths=1.5,
                    color=color, ax=ax, zorder=2)

        # 3) 顶层回归线（纯黑，无置信区间）
        sns.regplot(data=sub_clip, x='ASD', y='Dice',
                    scatter=False, ci=None,
                    line_kws={'color': 'black', 'lw': 2, 'zorder': 3},
                    ax=ax)

        # 4) 统计文本框
        xy = sub_clip[['ASD', 'Dice']].dropna()
        n  = len(xy)
        r  = float(np.corrcoef(xy['ASD'], xy['Dice'])[0, 1])
        ax.text(0.97, 0.05,
                f"r = {r:.3f}\n$R^2$ = {r**2:.3f}\nN = {n}",
                transform=ax.transAxes, va='bottom', ha='right', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

        ax.set_xlim(0, 20)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(model.replace('_', ' '), fontweight='bold', pad=10,
                     color=color)
        ax.set_xlabel('ASD  (↓)')
        ax.set_ylabel('Dice Score  (↑)')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        sns.despine(ax=ax)

    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_3_Final_KDE.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Fig 3 saved: {os.path.join(OUTPUT_DIR, 'Fig_3_Final_KDE.png')}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_yolo = load_data('evaluation_results_5folds_full.csv')
    plot_fig2_boxplot(df_yolo)
    plot_fig3_kde(df_yolo)
    print(f"\n[Done] Final thesis figures saved to [{OUTPUT_DIR}/]")
