"""
plot_revised_figures.py  (v4 — 修复信息冗余与箭头交叉)

Fig_1_Comprehensive_Boxplot.pdf:
  Left  — Grouped bar: % cases with Dice > 0.90, by model × modality
  Right — Grouped bar: Modality Gap (Colour − Infrared Dice), by model

Fig_2_Scatter_Regression.pdf:
  Left  — ECDF of Dice (Annotating Median Shift)
  Right — Per-fold mean Dice with 95% CI error bars
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5,
    'legend.fontsize': 11,
    'axes.linewidth': 1.1,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
})

OUTPUT_DIR = "Ultimate_Thesis_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER  = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']
MODEL_LABELS = ['SAM2\nBaseline', 'SAM2\nLoRA', 'SAM2\nMSA', 'GAL-SAM\n(Ours)']
MODEL_COLORS = {
    'ST-SAM':        '#B30000',
    'MSA_SAM2':      '#1F77B4',
    'Baseline_SAM2': '#7F7F7F',
    'LoRA_SAM2':     '#D55E00',
}
MOD_COLORS = {'Colour': '#D45B42', 'Infrared': '#406A8C'}


# ════════════════════════════════════════════════════════════════════════════════
# Fig 1: Dice>0.9 比例 (left) + Modality Gap (right)
# ════════════════════════════════════════════════════════════════════════════════
def plot_fig1(df_yolo):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel A: % Dice > 0.90 ────────────────────────────────────────────────
    ax = axes[0]
    n = len(MODEL_ORDER)
    x = np.arange(n)
    w = 0.35

    pct_c, pct_ir = [], []
    for m in MODEL_ORDER:
        sub = df_yolo[df_yolo['Model'] == m]
        pct_c.append(100 * (sub[sub['Modality'] == 'Colour']['Dice'] > 0.9).mean())
        pct_ir.append(100 * (sub[sub['Modality'] == 'Infrared']['Dice'] > 0.9).mean())

    bars_c  = ax.bar(x - w/2, pct_c,  w, color=MOD_COLORS['Colour'],
                     edgecolor='#222', linewidth=1.0, label='Colour',   alpha=0.88)
    bars_ir = ax.bar(x + w/2, pct_ir, w, color=MOD_COLORS['Infrared'],
                     edgecolor='#222', linewidth=1.0, label='Infrared', alpha=0.88)

    st_idx = MODEL_ORDER.index('ST-SAM')
    for bar in [bars_c[st_idx], bars_ir[st_idx]]:
        bar.set_linewidth(2.5)
        bar.set_edgecolor('#B30000')

    for bars, vals in [(bars_c, pct_c), (bars_ir, pct_ir)]:
        for i, (bar, v) in enumerate(zip(bars, vals)):
            fw = 'bold' if MODEL_ORDER[i] == 'ST-SAM' else 'normal'
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.4,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight=fw)

    y_top = max(pct_c + pct_ir) * 1.28
    for offset, vals, mod, xshift in [
        (-w/2, pct_c,  'Colour',   +0.55),
        (+w/2, pct_ir, 'Infrared', -0.55),
    ]:
        delta = vals[st_idx] - vals[0]
        bar_h = vals[st_idx]
        xy_tip = (st_idx + offset, bar_h + 0.8)
        xt = st_idx + offset + xshift
        yt = bar_h + 10
        ax.annotate(
            f'+{delta:.1f}pp vs Baseline',
            xy=xy_tip, xytext=(xt, yt),
            fontsize=8.5, color='#222222', fontweight='bold', ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#B30000',
                      linewidth=1.2, alpha=0.95),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.set_ylabel('Cases with Dice > 0.90 (%)', fontsize=12)
    ax.set_title('A. High-Quality Segmentation Rate\n(Dice > 0.90 threshold)', fontweight='bold', pad=10)
    ax.set_ylim(0, max(pct_c + pct_ir) * 1.28)
    ax.legend(loc='upper left', frameon=True, edgecolor='#ccc')
    ax.grid(True, axis='y', linestyle=':', alpha=0.45, color='gray')
    sns.despine(ax=ax)

    # ── Panel B: Modality Gap ─────────────────────────────────────────────────
    ax = axes[1]
    gaps, gap_errs = [], []
    for m in MODEL_ORDER:
        sub = df_yolo[df_yolo['Model'] == m]
        c_vals  = sub[sub['Modality'] == 'Colour']['Dice'].values
        ir_vals = sub[sub['Modality'] == 'Infrared']['Dice'].values
        n_boot = 1000
        rng = np.random.default_rng(0)
        boot_gaps = [rng.choice(c_vals, len(c_vals), replace=True).mean() -
                     rng.choice(ir_vals, len(ir_vals), replace=True).mean()
                     for _ in range(n_boot)]
        gaps.append(np.mean(c_vals) - np.mean(ir_vals))
        gap_errs.append(np.std(boot_gaps) * 1.96)

    bar_colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(range(n), gaps, color=bar_colors, edgecolor='#222',
                  linewidth=1.0, alpha=0.88, width=0.55, zorder=3,
                  yerr=gap_errs, capsize=5,
                  error_kw=dict(elinewidth=1.3, ecolor='#333', capthick=1.3))
    bars[st_idx].set_linewidth(2.5)
    bars[st_idx].set_edgecolor('#B30000')

    for i, (bar, g, e) in enumerate(zip(bars, gaps, gap_errs)):
        fw = 'bold' if MODEL_ORDER[i] == 'ST-SAM' else 'normal'
        col = '#B30000' if MODEL_ORDER[i] == 'ST-SAM' else '#333'
        ax.text(bar.get_x() + bar.get_width()/2, g + e + 0.001,
                f'{g:.3f}', ha='center', va='bottom', fontsize=10, fontweight=fw, color=col)

    ax.axhline(gaps[st_idx], color='#B30000', ls='--', lw=1.4, alpha=0.6, zorder=2)
    ax.set_xticks(range(n))
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.set_ylabel('Colour − Infrared Dice  (↓ smaller = more robust)', fontsize=11)
    ax.set_title('B. Cross-Modality Robustness\n(Modality Gap in Dice Score)', fontweight='bold', pad=10)
    ax.set_ylim(0, max(gaps) * 1.45)
    ax.grid(True, axis='y', linestyle=':', alpha=0.45, color='gray')
    sns.despine(ax=ax)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig_1_Comprehensive_Boxplot.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Fig 1 -> {out}')


# ════════════════════════════════════════════════════════════════════════════════
# Fig 2: ECDF of Dice (left) + Per-fold mean Dice with CI (right)
# ════════════════════════════════════════════════════════════════════════════════
def plot_fig2(df_yolo):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel A: ECDF of Dice ─────────────────────────────────────────────────
    ax = axes[0]
    linestyles = {'Colour': '-', 'Infrared': '--'}
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        lw = 2.6 if model == 'ST-SAM' else 1.5
        for mod, ls in linestyles.items():
            sub = df_yolo[(df_yolo['Model'] == model) &
                          (df_yolo['Modality'] == mod)]['Dice'].dropna().sort_values().values
            ecdf = np.arange(1, len(sub) + 1) / len(sub)
            ax.plot(sub, ecdf, color=color, lw=lw, ls=ls,
                    alpha=1.0 if model == 'ST-SAM' else 0.55,
                    zorder=5 if model == 'ST-SAM' else 3)

    # 替换原来的0.9垂直线，改为标注中位数 (y=0.5)
    ax.axhline(0.5, color='#444', lw=1.0, ls=':', alpha=0.7)
    ax.text(0.555, 0.51, 'Median (50th Percentile)', fontsize=8.5, color='#444', va='bottom')


    # 【修复交叉与冗余】：改为标注中位数的绝对提升 (避开左上角图例)
    # 将 xtext 统一右移至 0.68，彻底让开左侧图例的区域

    
    # 【终极防遮挡】：完美避开左上角图例，利用左侧中部的纯净真空区
    annotations = [
        # Infrared：往下挪一点点 (从 0.66 降到 0.62)，留出呼吸空间，且与下方完美对称
        ('Infrared', 0.56, 0.62, 'left', 0.15),   
        # Colour：保持在 y=0.38
        ('Colour',   0.56, 0.38, 'left', -0.15)   
    ]
    
    for mod, xt, yt, align, rad in annotations:
        sub_st   = df_yolo[(df_yolo['Model'] == 'ST-SAM') & (df_yolo['Modality'] == mod)]['Dice'].dropna().values
        sub_base = df_yolo[(df_yolo['Model'] == 'Baseline_SAM2') & (df_yolo['Modality'] == mod)]['Dice'].dropna().values
        med_st   = np.median(sub_st)
        med_base = np.median(sub_base)
        
        ax.annotate(
            f'GAL-SAM {mod} Median: {med_st:.3f}\n(+{med_st - med_base:.3f} vs Baseline)',
            xy=(med_st, 0.5), xytext=(xt, yt),
            ha=align, va='center', 
            fontsize=8.5, color='#B30000', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#B30000', lw=1.1, connectionstyle=f'arc3,rad={rad}'),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#B30000', alpha=0.9))

    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Dice Score', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('A. Cumulative Distribution of Dice Score\n(YOLO Automated Mode)', fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.4, color='gray')
    sns.despine(ax=ax)

    model_patches = [mpatches.Patch(color=MODEL_COLORS[m], alpha=0.85,
                                    label=lbl.replace('\n', ' '))
                     for m, lbl in zip(MODEL_ORDER, MODEL_LABELS)]
    ls_lines = [plt.Line2D([0], [0], color='#555', lw=1.5, ls='-',  label='Colour'),
                plt.Line2D([0], [0], color='#555', lw=1.5, ls='--', label='Infrared')]
    ax.legend(handles=model_patches + ls_lines, loc='upper left',
              frameon=True, edgecolor='#ccc', fontsize=9.5)

    # ── Panel B: Per-Centre Dice Consistency ──────────────────────────────────
    ax = axes[1]
    fold_mod = (df_yolo.groupby(['Model', 'Fold'])
                .agg(Dice=('Dice', 'mean'), Modality=('Modality', 'first'))
                .reset_index())
    x = np.arange(len(MODEL_ORDER))
    dot_colors = {'Colour': '#D45B42', 'Infrared': '#406A8C'}

    for i, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model]
        sub = fold_mod[fold_mod['Model'] == model].sort_values('Fold')
        vals = sub['Dice'].values
        mods = sub['Modality'].values

        rng = np.random.default_rng(i)
        jit = rng.uniform(-0.10, 0.10, len(vals))
        for v, m, j in zip(vals, mods, jit):
            ax.scatter(x[i] + j, v, color=dot_colors[m], s=38, zorder=3,
                       alpha=0.85, edgecolors='white', linewidths=0.5,
                       marker='o' if m == 'Colour' else '^')

        mean_val = vals.mean()
        lw_mean = 2.8 if model == 'ST-SAM' else 2.0
        ax.plot([x[i] - 0.25, x[i] + 0.25], [mean_val, mean_val],
                color=color, lw=lw_mean, zorder=5, solid_capstyle='round')
        ax.text(x[i], mean_val + 0.0022,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=9.5,
                color=color, fontweight='bold' if model == 'ST-SAM' else 'normal')

    leg_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=dot_colors['Colour'],
                   markersize=7, label='Colour centre'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=dot_colors['Infrared'],
                   markersize=7, label='Infrared centre'),
        plt.Line2D([0], [0], color='#555', lw=2.2, label='Overall mean'),
    ]
    ax.legend(handles=leg_handles, loc='lower right', frameon=True,
              edgecolor='#ccc', fontsize=9.5)

    all_vals = fold_mod['Dice'].values
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.set_ylabel('Mean Dice Score per Centre (5-fold LOCO CV)', fontsize=11)
    ax.set_title('B. Per-Centre Dice Consistency\n'
                 '(Each dot = one held-out centre; bar = overall mean)',
                 fontweight='bold', pad=10)
    ax.set_ylim(all_vals.min() - 0.012, all_vals.max() + 0.022)
    ax.grid(True, axis='y', linestyle=':', alpha=0.45, color='gray')
    sns.despine(ax=ax)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig_2_Scatter_Regression.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Fig 2 -> {out}')


if __name__ == '__main__':
    df = pd.read_csv('evaluation_results_5folds_full.csv')
    df_yolo = df[(df['Padding'] == 'YOLO') & (df['Dice'] > 0)].copy()
    print(f'Loaded {len(df_yolo)} YOLO rows')
    plot_fig1(df_yolo)
    plot_fig2(df_yolo)
    print('Done.')