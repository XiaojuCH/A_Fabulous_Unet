import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 路径配置 (确保包含所有对比模型)
# ==========================================
JSON_PATH = "data_splits/clean_full_list.json"
GT_ROOT_DIR = "../Unet/dataset"

DIRS = {
    "DeepLabV3+": "results/masks_deeplab",
    "Swin-UNETR": "results/masks_swinunet",
    # "U-Net": "results/masks_unet",
    "SAM2-LoRA": "results/masks_lora_yolo",
    "SAM2-MSA": "results/masks_msa_yolo",
    "ST-SAM (Ours)": "results/masks_stsam_yolo"
}

# ==========================================
# 2. 鲁棒测厚与长宽比提取
# ==========================================
def extract_mask_properties(mask_path):
    if not os.path.exists(mask_path): return None, None
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.max(mask) == 0: return 0.0, 0.0
        
    binary_mask = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels < 2: return 0.0, 0.0
    
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    width = stats[largest_idx, cv2.CC_STAT_WIDTH]
    height = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    
    if width < 10: return 0.0, 0.0
        
    tmh = float(area / width)
    aspect_ratio = float(width / max(height, 1))
    
    return tmh, aspect_ratio

# ==========================================
# 3. 数据遍历与双轨收集
# ==========================================
def main():
    print("🚀 启动全量 vs. 极端拓扑 双轨临床测量评估...")
    with open(JSON_PATH, 'r') as f:
        data_list = json.load(f)
        
    # 分别存储 全量 和 极端 样本的误差
    errors_all = {name: [] for name in DIRS.keys()}
    errors_extreme = {name: [] for name in DIRS.keys()}
    
    count_all = 0
    count_extreme = 0
    
    for item in data_list:
        sample_id = item.get("id")
        raw_gt_rel_path = item.get("label")
        gt_rel_path = raw_gt_rel_path.replace("Label", "Cleaned_Label")
        gt_path = os.path.normpath(os.path.join(GT_ROOT_DIR, "..", gt_rel_path))
        
        gt_tmh, aspect_ratio = extract_mask_properties(gt_path)
        
        # 过滤解剖学异常值
        if gt_tmh is None or gt_tmh == 0 or gt_tmh > 200: continue
            
        pred_tmhs = {}
        missing_mask = False
        for name, dir_path in DIRS.items():
            p_path = os.path.join(dir_path, f"{sample_id}.png")
            # 如果某个模型还没跑出图，给出警告并跳过该模型（防止脚本崩溃）
            if not os.path.exists(p_path):
                missing_mask = True
                break
            p_tmh, _ = extract_mask_properties(p_path)
            pred_tmhs[name] = p_tmh if p_tmh is not None else 0.0
            
        if missing_mask: continue
            
        is_extreme = (aspect_ratio > 8.0)
        
        for name in DIRS.keys():
            diff = abs(pred_tmhs[name] - gt_tmh)
            # 全图假阳性惩罚
            if pred_tmhs[name] > 200 or pred_tmhs[name] == 0: 
                diff = 100.0 
                
            errors_all[name].append(diff)
            if is_extreme:
                errors_extreme[name].append(diff)
                
        count_all += 1
        if is_extreme:
            count_extreme += 1

    print(f"✅ 解析完成！全量有效样本: {count_all} | 极端挑战样本: {count_extreme}")

    # ==========================================
    # 4. 绘制 1x2 双子图 (JBHI 标准排版)
    # ==========================================
    os.makedirs("results/clinical_plots", exist_ok=True)
    
    # 设置 1x2 画布，横向拉长以适应双栏或通栏排版
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    labels = [k.replace(" ", "\n") for k in DIRS.keys()]
    colors = ['#95a5a6', '#e67e22', '#3498db', '#9b59b6', '#f1c40f', '#e74c3c'][:len(DIRS)]
    
    # 绘图函数抽象
    def draw_boxplot(ax, data_dict, title, y_max=None):
        data_to_plot = [data_dict[k] for k in DIRS.keys()]
        # showfliers=False 隐藏离群点，让箱体对比更明显
        box = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=labels, 
                         showfliers=False, widths=0.6, zorder=3)
        
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for median in box['medians']:
            median.set(color='black', linewidth=2)
            
        ax.set_ylabel("Absolute Measurement Error (Pixels)", fontsize=12)
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        ax.tick_params(axis='x', labelsize=10)
        if y_max:
            ax.set_ylim(-0.5, y_max)
            
    # 左图：全量样本
    draw_boxplot(axes[0], errors_all, f"(a) Overall Clinical Performance (N={count_all})")
    
    # 右图：困难样本
    # 动态调整 y 轴上限，让右图的剧烈波动清晰可见
    max_median_extreme = max([np.median(errors_extreme[k]) for k in DIRS.keys()])
    draw_boxplot(axes[1], errors_extreme, f"(b) Extreme Topology Challenge (AR > 8.0, N={count_extreme})", y_max=max_median_extreme * 4)

    plt.tight_layout()
    plt.savefig("results/clinical_plots/Clinical_Dual_Boxplot.pdf", dpi=300)
    plt.close()
    
    print("\n出图成功！请查看 results/clinical_plots/Clinical_Dual_Boxplot.pdf")

if __name__ == "__main__":
    main()