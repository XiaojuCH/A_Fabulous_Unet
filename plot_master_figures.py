import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 顶会极简学术样式 (Nature / CVPR 风格) =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 15,    
    'axes.titlesize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,   
    'figure.dpi': 300,
    'pdf.fonttype': 42
})

OUTPUT_DIR = "Final_Academic_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 顶级学术重色彩 (Deep Academic Palette) =================
MODEL_COLORS = {
    'ST-SAM': '#B30000',         # 深血红
    'MSA_SAM2': '#1F77B4',       # 经典深蓝
    'Baseline_SAM2': '#7F7F7F',  # 稳重中性灰
    'LoRA_SAM2': '#D55E00'       # 深焦糖/橘红色
}

MODALITY_COLORS = {
    'Colour': '#D45B42',         # 深铁锈红
    'Infrared': '#406A8C'        # 深钴蓝/灰蓝
}

# ================= 3. 图 1：SOTA 模态分组箱线图 =================
def plot_sota_glass_boxplot(df):
    print("1/2 正在绘制 SOTA 分组箱线图 (强化质感版)...")
    df_yolo = df[df['Padding'] == 'YOLO'].copy()
    model_order = ['Baseline_SAM2', 'LoRA_SAM2', 'MSA_SAM2', 'ST-SAM']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    box_props = {
        'x': 'Model', 'hue': 'Modality', 'data': df_yolo, 
        'palette': MODALITY_COLORS, 'order': model_order,
        'width': 0.65, 
        'linewidth': 1.5, 
        'boxprops': {'alpha': 0.8, 'edgecolor': '#333333'},
        'medianprops': {'color': 'black', 'linewidth': 1.8}, 
        'whiskerprops': {'color': '#333333', 'linewidth': 1.2},
        'capprops': {'color': '#333333', 'linewidth': 1.2},
        'flierprops': {'marker': '.', 'markerfacecolor': 'gray', 'markeredgecolor': 'none', 'alpha': 0.4, 'markersize': 5}
    }

    # 左侧：Dice 箱线图
    sns.boxplot(y='Dice', ax=axes[0], **box_props)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Region Overlap under Automated YOLO Prompts", fontweight='bold', pad=15)
    axes[0].set_ylabel("Dice Score (Higher is Better)")
    axes[0].set_xlabel("")

    # 右侧：HD95 箱线图 (对数轴)
    sns.boxplot(y='HD95', ax=axes[1], **box_props)
    axes[1].set_yscale('log') 
    axes[1].set_title("Boundary Drift under Automated YOLO Prompts", fontweight='bold', pad=15)
    axes[1].set_ylabel("HD95 Error (Pixels, Log Scale)", labelpad=10)
    axes[1].set_xlabel("")

    for ax in axes:
        ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray') 
        ax.set_xticks(ax.get_xticks())
        new_xticks = [label.get_text().replace('ST-SAM', 'GAL-SAM') for label in ax.get_xticklabels()]
        ax.set_xticklabels(new_xticks)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title='', loc='best', frameon=True, edgecolor='black')
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_1_SOTA_Boxplot.pdf'), bbox_inches='tight')
    plt.close()

# ================= 4. 图 2：连续鲁棒性折线图 =================
def plot_robustness_clean_lines(df):
    print("2/2 正在绘制连续鲁棒性折线图 (加深 CI 阴影版)...")
    df_num = df[df['Padding'] != 'YOLO'].copy()
    df_num['Padding'] = pd.to_numeric(df_num['Padding'])
    df_yolo = df[df['Padding'] == 'YOLO'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    xticks = [-5, 0, 5, 10, 20, 30, 40]
    metrics = [('Dice', 'Dice Score (Higher is Better)', axes[0]), 
               ('HD95', 'HD95 Error (Pixels, Log Scale)', axes[1])]
               
    for metric, ylabel, ax in metrics:
        sns.lineplot(
            data=df_num, x='Padding', y=metric, hue='Model', 
            palette=MODEL_COLORS, ax=ax, 
            errorbar=('ci', 95),      
            err_kws={'alpha': 0.2},  
            marker='o', markersize=7, markeredgecolor='white' 
        )
        
        for line in ax.lines:
            if line.get_color() == MODEL_COLORS['ST-SAM']:
                line.set_linewidth(3.0) 
                line.set_zorder(5) 
            else:
                line.set_linewidth(1.8) 

        models = df['Model'].unique()
        for model in models:
            mean_val = df_yolo[df_yolo['Model'] == model][metric].mean()
            color = MODEL_COLORS[model]
            lw = 2.0 if model == 'ST-SAM' else 1.2
            alpha = 0.8 if model == 'ST-SAM' else 0.6 
            
            ax.axhline(y=mean_val, color=color, linestyle='--', linewidth=lw, alpha=alpha, zorder=1)
            if model == 'ST-SAM':
                ax.text(41, mean_val, 'Auto (YOLO)', color=color, va='center', ha='left', fontsize=11, fontweight='bold')

        ax.set_xticks(xticks)
        ax.set_xlim(-7, 48) 
        ax.set_title(f'Robustness of {metric} to Box Expansion', fontweight='bold', pad=15)
        ax.set_xlabel('Box Expansion / Padding (Linear Pixels)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        if metric == 'HD95': ax.set_yscale('log')
        ax.grid(True, linestyle='-', alpha=0.3, color='gray')
        sns.despine(ax=ax)
        
        # 【修复 Error】：确保先获取原始图例，存在新变量里再替换
        handles, orig_labels = ax.get_legend_handles_labels()
        new_labels = [lbl.replace('ST-SAM', 'GAL-SAM') for lbl in orig_labels]
        
        ax.legend(handles=handles, labels=new_labels, loc='best', frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_2_Robustness_Lines.pdf'), bbox_inches='tight')
    plt.close()

# ================= 5. 主程序入口 =================
def main():
    csv_file = "evaluation_results_5folds_full.csv"
    if not os.path.exists(csv_file):
        print(f"❌ 找不到数据文件: {csv_file}")
        return

    print("正在加载海量评估数据...")
    df = pd.read_csv(csv_file).dropna(subset=['Dice', 'HD95'])
    
    plot_sota_glass_boxplot(df)
    plot_robustness_clean_lines(df)
    
    print(f"\n🎉 完美收工！最终版学术主图已生成至 [{OUTPUT_DIR}] 文件夹！")

if __name__ == "__main__":
    main()