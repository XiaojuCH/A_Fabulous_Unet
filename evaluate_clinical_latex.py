import os
import json
import cv2
import numpy as np

# ==========================================
# 1. 路径与配置 (涵盖全家福模型)
# ==========================================
JSON_PATH = "data_splits/clean_full_list.json"
GT_ROOT_DIR = "../Unet/dataset"
CLINICAL_TOLERANCE_PX = 15.0

# 组别 A: 全自动模式 (端到端 CNN 或 YOLO Prompt)
DIRS_AUTO = {
    "U-Net (End-to-End)": "results/masks_unet",
    "DeepLabV3+ (End-to-End)": "results/masks_deeplab",
    "Swin-UNETR (End-to-End)": "results/masks_swinunet",
    "SAM2-LoRA": "results/masks_lora_yolo",
    "SAM2-MSA": "results/masks_msa_yolo",
    "ST-SAM (Ours)": "results/masks_stsam_yolo"
}

# 组别 B: 半自动专家模式 (Oracle GT Box)
DIRS_EXPERT = {
    "MedSAM": "results/masks_medsam_gt",
    "SAM2-LoRA ": "results/masks_lora_gt",
    "SAM2-MSA ": "results/masks_msa_gt",
    "ST-SAM (Ours) ": "results/masks_stsam_gt" # 加个空格防止字典键名冲突
}

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

def evaluate_models(model_dict, data_list):
    diffs = {name: [] for name in model_dict.keys()}
    valid_samples = 0
    
    for item in data_list:
        sample_id = item.get("id")
        raw_gt_rel_path = item.get("label")
        gt_rel_path = raw_gt_rel_path.replace("Label", "Cleaned_Label")
        gt_path = os.path.normpath(os.path.join(GT_ROOT_DIR, "..", gt_rel_path))
        
        gt_tmh, _ = extract_mask_properties(gt_path)
        if gt_tmh is None or gt_tmh == 0 or gt_tmh > 200: continue
            
        pred_tmhs = {}
        missing = False
        for name, dir_path in model_dict.items():
            p_path = os.path.join(dir_path, f"{sample_id}.png")
            if not os.path.exists(p_path):
                missing = True; break
            p_tmh, _ = extract_mask_properties(p_path)
            pred_tmhs[name] = p_tmh if p_tmh is not None else 0.0
            
        if missing: continue
            
        for name in model_dict.keys():
            error = pred_tmhs[name] - gt_tmh
            if pred_tmhs[name] > 200 or pred_tmhs[name] == 0:
                error = 100.0 if pred_tmhs[name] > 200 else -gt_tmh
            diffs[name].append(error)
            
        valid_samples += 1
        
    results = {}
    for name in model_dict.keys():
        arr = np.array(diffs[name])
        abs_arr = np.abs(arr)
        results[name] = {
            "median": np.median(abs_arr),
            "max": np.max(abs_arr),
            "loa": 1.96 * np.std(arr) * 2,
            "failure": np.mean(abs_arr > CLINICAL_TOLERANCE_PX) * 100
        }
    return results, valid_samples

def generate_latex_table(res_auto, res_expert):
    # 找出每一列的最优值（最小值）进行加粗加星号标识
    auto_medians = [v["median"] for v in res_auto.values()]
    auto_maxes = [v["max"] for v in res_auto.values()]
    auto_loas = [v["loa"] for v in res_auto.values()]
    auto_fails = [v["failure"] for v in res_auto.values()]
    
    exp_medians = [v["median"] for v in res_expert.values()]
    exp_maxes = [v["max"] for v in res_expert.values()]
    exp_loas = [v["loa"] for v in res_expert.values()]
    exp_fails = [v["failure"] for v in res_expert.values()]
    
    def format_val(val, val_list):
        # 简单判断是否是当前组别最优，最优加粗
        is_best = (val == min(val_list))
        formatted = f"{val:.2f}"
        if is_best:
            return f"\\textbf{{{formatted}}}"
        return formatted

    latex_str = """
\\begin{table*}[t]
\\centering
\\caption{Clinical Safety and Reliability Metrics Across Dual Deployment Modes}
\\label{tab:clinical_metrics}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{l c c c c}
\\toprule
\\textbf{Modality \\& Model} & \\textbf{Median Error (px) $\\downarrow$} & \\textbf{Max Error (px) $\\downarrow$} & \\textbf{95\\% LoA Width (px) $\\downarrow$} & \\textbf{Failure Rate (\\%) $\\downarrow$} \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{Panel A: Automated Screening Mode (YOLOv8n Prompt)}} \\\\
\\midrule
"""
    # 填充自动模式
    for name, metrics in res_auto.items():
        med_str = format_val(metrics["median"], auto_medians)
        max_str = format_val(metrics["max"], auto_maxes)
        loa_str = format_val(metrics["loa"], auto_loas)
        fail_str = format_val(metrics["failure"], auto_fails) + "\\%" if metrics["failure"] != min(auto_fails) else f"\\textbf{{{metrics['failure']:.2f}}}\\%"
        
        # 将 ST-SAM 突出显示
        model_name = f"\\textbf{{{name}}}" if "ST-SAM" in name else name
        latex_str += f"{model_name} & {med_str} & {max_str} & {loa_str} & {fail_str} \\\\\n"
        
    latex_str += """\\midrule
\\multicolumn{5}{l}{\\textit{Panel B: Expert-Guided Mode (Oracle BBox Prompt)}} \\\\
\\midrule
"""
    # 填充专家模式
    for name, metrics in res_expert.items():
        med_str = format_val(metrics["median"], exp_medians)
        max_str = format_val(metrics["max"], exp_maxes)
        loa_str = format_val(metrics["loa"], exp_loas)
        fail_str = format_val(metrics["failure"], exp_fails) + "\\%" if metrics["failure"] != min(exp_fails) else f"\\textbf{{{metrics['failure']:.2f}}}\\%"
        
        # 去除字典命名时防止冲突的空格，并高亮 ST-SAM
        clean_name = name.strip()
        model_name = f"\\textbf{{{clean_name}}}" if "ST-SAM" in clean_name else clean_name
        latex_str += f"{model_name} & {med_str} & {max_str} & {loa_str} & {fail_str} \\\\\n"

    latex_str += """\\bottomrule
\\end{tabular}%
}
\\begin{tablenotes}
\\small
\\item \\textit{Note:} Failure Rate defines the percentage of cases with an absolute TMH error $>$ 15 pixels. The 100.00 pixels in Max Error represents the penalty cap for catastrophic segmentation collapse, primarily triggered by YOLO bounding box spatial jitter in the automated mode.
\\end{tablenotes}
\\end{table*}
"""
    return latex_str

def main():
    print("🚀 开始双轨模式临床评估扫描...")
    with open(JSON_PATH, 'r') as f:
        data_list = json.load(f)
        
    print("⏳ 正在评估 组别 A: 全自动模式...")
    res_auto, valid_a = evaluate_models(DIRS_AUTO, data_list)
    
    print("⏳ 正在评估 组别 B: 半自动专家模式...")
    res_expert, valid_b = evaluate_models(DIRS_EXPERT, data_list)
    
    print(f"\n✅ 评估完成! 全自动有效样本: {valid_a}, 专家模式有效样本: {valid_b}")
    print("\n👇 请将下方生成的 LaTeX 源码直接复制到 Overleaf 中 👇\n")
    print("=" * 80)
    
    latex_code = generate_latex_table(res_auto, res_expert)
    print(latex_code)
    
    print("=" * 80)

if __name__ == "__main__":
    main()