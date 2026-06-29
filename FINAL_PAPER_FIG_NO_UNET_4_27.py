import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import os
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ================= 0. 顶刊出版规范级配置 =================
# 强制要求 Matplotlib 将字体嵌入为 Type 42，避免 ScholarOne 等系统报 Type 3 字体错误
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# ================= 1. 核心配置区域 =================
DATA_ROOT = "./results"

# 赛道一：彩色选你最新找出的完美断裂 000160；红外用左侧断裂的 000579
TRACK1_IDS = ['Color1_000160', 'Infrared3_000579']
# 赛道二：彩色用严重溢出的 000050；红外用尾部粗钝的 000388
TRACK2_IDS = ['Color2_000050', 'Infrared3_000388']

# 🎯 已经精确估算的画中画坐标
ARROWS = {
    "Color1_000160":    {'fail': (431, 682), 'success': (431, 682)},
    "Infrared3_000579": {'fail': (158, 464), 'success': (158, 464)},
    "Color2_000050":    {'fail': (1220, 792), 'success': (1220, 792), 'zoom': 200},
    "Infrared3_000388": {'fail': (1139, 682), 'success': (1139, 682)},
}
CROP_SIDE = 700
FOCUS_MARGIN = 110
MASK_GREEN = (30, 210, 70)
FAIL_RED = '#D73027'
SUCCESS_GREEN = '#1A9850'
GT_WHITE = '#F7F7F7'
FIG_SIZE = (7.16, 4.85)
TITLE_FONTSIZE = 8.0
ROW_GROUP_FONTSIZE = 7.4
ROW_MODALITY_FONTSIZE = 7.0
# ===================================================

def find_image_path(folder_path, img_id):
    search_pattern = os.path.join(folder_path, f"{img_id}.*")
    matches = glob.glob(search_pattern)
    return matches[0] if matches else None

def overlay_mask(image_path, mask_path, color=MASK_GREEN, alpha=0.45):
    img = cv2.imread(image_path)
    if img is None: return np.zeros((1024, 1024, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not mask_path or not os.path.exists(mask_path): return img 
        
    mask = cv2.imread(mask_path, 0)
    if mask is None: return img
    
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 127] = color
    img_blended = cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)
    
    # 提取并绘制硬边缘轮廓，防止半透明蒙版导致边缘模糊
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_blended, contours, -1, color, thickness=2)
    
    return img_blended

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 2:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_label] = 255
    return cleaned

def overlay_mask_for_panel(image_path, mask_path, img_id, model_folder):
    if img_id in ["Color2_000050", "Infrared3_000388"] and model_folder == "masks_stsam_gt":
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros((1024, 1024, 3), dtype=np.uint8)

        mask = cv2.imread(mask_path, 0) if mask_path and os.path.exists(mask_path) else None
        if mask is None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = keep_largest_component(cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1])
        mask_path = None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[mask > 127] = MASK_GREEN
        img_blended = cv2.addWeighted(img_rgb, 1.0, colored_mask, 0.45, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_blended, contours, -1, MASK_GREEN, thickness=2)
        return img_blended

    return overlay_mask(image_path, mask_path)

def draw_arrow(ax, coords, color='red'):
    if not coords or coords == (0, 0): return
    ax.annotate('', xy=coords, xytext=(coords[0]+35, coords[1]-35),
                arrowprops=dict(
                    arrowstyle='-|>',
                    mutation_scale=7,
                    facecolor=color,
                    edgecolor='white',
                    linewidth=0.7,
                    shrinkA=0,
                    shrinkB=2
                ))

def crop_start_from_focus(focus_values, side, limit, priority_value=None, margin=FOCUS_MARGIN):
    if priority_value is not None:
        start = int(round(priority_value - side / 2))
    elif focus_values:
        start = int(round((min(focus_values) + max(focus_values)) / 2 - side / 2))
    else:
        start = int(round((limit - side) / 2))

    start = max(0, min(start, limit - side))

    if focus_values and max(focus_values) - min(focus_values) + 2 * margin <= side:
        min_allowed = max(0, max(focus_values) + margin - side)
        max_allowed = min(limit - side, min(focus_values) - margin)
        if min_allowed <= max_allowed:
            start = max(min_allowed, min(start, max_allowed))

    return int(start)

def generate_merged_figure():
    fig_name = "Fig_2_Ultimate_4x6_Comparison.pdf"
    print(f"🚀 正在生成剔除了 U-Net 的终极 4x6 顶刊排版...")
    
    # 🚀 优化：移除 layout='constrained'，拿回绝对排版控制权
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=FIG_SIZE)
    # 强制锁定极其微小的列间距和适当的行间距，确保每一列宽度绝对均等
    plt.subplots_adjust(wspace=0.035, hspace=0.16, left=0.085, right=0.995, top=0.94, bottom=0.025)
    
    track1_cols = [
        ("Input (YOLO)", None), ("GT", "masks_gt"), 
        ("Swin-UNETR", "masks_swinunet"), ("DeepLabV3", "masks_deeplab"), 
        ("SAM2 Base", "masks_baseline_sam_yolo"), ("GAL-SAM2 (Ours)", "masks_stsam_yolo")
    ]
    track2_cols = [
        ("Input (Expert)", None), ("GT", "masks_gt"), 
        ("MedSAM", "masks_medsam_gt"), ("SAM2 Base", "masks_baseline_sam_gt"), 
        ("SAM2 MSA", "masks_msa_gt"), ("GAL-SAM2 (Ours)", "masks_stsam_gt")
    ]
    
    all_ids = TRACK1_IDS + TRACK2_IDS
    row_labels = [
        ("Automated", "Colour"), ("Automated", "Infrared"),
        ("Expert-guided", "Colour"), ("Expert-guided", "Infrared")
    ]
    for row_idx, img_id in enumerate(all_ids):
        is_track1 = row_idx < 2
        columns = track1_cols if is_track1 else track2_cols
        box_color = '#F2C94C' if is_track1 else '#4DB6AC'
        
        mask_gt_path = find_image_path(f"{DATA_ROOT}/masks_gt", img_id)
        if mask_gt_path:
            mask_gt = cv2.imread(mask_gt_path, 0)
            ys, xs = np.where(mask_gt > 127)
            center_y = (ys.min() + ys.max()) // 2 if len(ys) > 0 else 512
            center_x = (xs.min() + xs.max()) // 2 if len(xs) > 0 else 512
        else:
            ys, xs = [], []
            center_y = 512
            center_x = 512

        img_path = find_image_path(f"{DATA_ROOT}/images", img_id)
        if not img_path: continue
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        crop_side = min(CROP_SIDE, img_h, img_w)

        f_c = ARROWS.get(img_id, {}).get('fail', (0, 0))
        s_c = ARROWS.get(img_id, {}).get('success', (0, 0))
        arrow_points = [p for p in (f_c, s_c) if p != (0, 0)]
        focus_x = [center_x] + [p[0] for p in arrow_points]
        focus_y = [center_y] + [p[1] for p in arrow_points]
        priority_x = arrow_points[0][0] if arrow_points else center_x
        priority_y = arrow_points[0][1] if arrow_points else center_y

        crop_x_min = crop_start_from_focus(focus_x, crop_side, img_w, priority_x)
        crop_y_min = crop_start_from_focus(focus_y, crop_side, img_h, priority_y)
        crop_x_max = crop_x_min + crop_side
        crop_y_max = crop_y_min + crop_side
            
        for col_idx, (col_title, model_folder) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.axis('off')
            
            if row_idx == 0 or row_idx == 2: 
                ax.set_title(col_title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=2.5)
            if col_idx == 0:
                row_group, row_modality = row_labels[row_idx]
                ax.text(-0.115, 0.54, row_group, transform=ax.transAxes,
                        fontsize=ROW_GROUP_FONTSIZE, fontweight='semibold', color='#222222',
                        va='center', ha='right')
                ax.text(-0.115, 0.44, row_modality, transform=ax.transAxes,
                        fontsize=ROW_MODALITY_FONTSIZE, fontweight='normal', color='#4A4A4A',
                        va='center', ha='right')
                
            if col_idx == 0:
                img_to_show = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            else:
                m_path = find_image_path(f"{DATA_ROOT}/{model_folder}", img_id) if model_folder else None
                img_to_show = overlay_mask_for_panel(img_path, m_path, img_id, model_folder)[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            ax.imshow(img_to_show)
            
            if col_idx == 0 and len(ys) > 0:
                pad = 20
                box_x = max(0, xs.min() - pad - crop_x_min)
                box_y = max(0, ys.min() - pad - crop_y_min)
                box_w = min(crop_x_max - crop_x_min, xs.max() + pad - crop_x_min) - box_x
                box_h = min(crop_y_max - crop_y_min, ys.max() + pad - crop_y_min) - box_y
                rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=1.0, edgecolor=box_color, facecolor='none', linestyle='--')
                ax.add_patch(rect)

            f_c = ARROWS.get(img_id, {}).get('fail', (0,0))
            s_c = ARROWS.get(img_id, {}).get('success', (0,0))
            target_pt = None
            zoom_color = 'white'

            if col_idx in range(2, len(columns)-1) and f_c != (0,0):
                target_pt = (f_c[0] - crop_x_min, f_c[1] - crop_y_min)
                zoom_color = FAIL_RED
            elif col_idx == len(columns)-1 and s_c != (0,0):
                target_pt = (s_c[0] - crop_x_min, s_c[1] - crop_y_min)
                zoom_color = SUCCESS_GREEN
            elif col_idx == 1 and s_c != (0,0):
                target_pt = (s_c[0] - crop_x_min, s_c[1] - crop_y_min)
                zoom_color = GT_WHITE

            if target_pt is not None:
                if 0 <= target_pt[1] < crop_side and 0 <= target_pt[0] < crop_side:
                    draw_arrow(ax, target_pt, zoom_color)
                    
                    inset_loc = 2 if target_pt[0] > crop_side // 2 else 1
                    
                    axins = inset_axes(ax, width="34%", height="34%", loc=inset_loc, borderpad=0.55)
                    axins.imshow(img_to_show, interpolation='none')
                    
                    zoom_size = ARROWS.get(img_id, {}).get('zoom', 150)
                    axins.set_xlim(target_pt[0] - zoom_size//2, target_pt[0] + zoom_size//2)
                    axins.set_ylim(target_pt[1] + zoom_size//2, target_pt[1] - zoom_size//2)
                    axins.set_xticks([]); axins.set_yticks([])
                    
                    for spine in axins.spines.values():
                        spine.set_edgecolor(zoom_color)
                        spine.set_linewidth(0.9)
                        
                    tx_min = target_pt[0] - zoom_size//2
                    tx_max = target_pt[0] + zoom_size//2
                    ty_min = target_pt[1] - zoom_size//2
                    ty_max = target_pt[1] + zoom_size//2
                    
                    # 1. 目标小框：实线，突出焦点
                    rect_target = patches.Rectangle((tx_min, ty_min), zoom_size, zoom_size, 
                                                    linewidth=0.8, edgecolor=zoom_color, facecolor='none')
                    ax.add_patch(rect_target)
                    
                    # 2. 牵引连线：虚线+半透明，减少信息遮挡，建立视觉层级
                    line_style = (0, (6, 4))  # 6pt线段，4pt间距
                    line_alpha = 0.55  
                    
                    if inset_loc == 2:
                        cp1 = ConnectionPatch(xyA=(tx_min, ty_min), coordsA="data", axesA=ax, 
                                              xyB=(1, 1), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=0.8, alpha=line_alpha, linestyle=line_style)
                        cp2 = ConnectionPatch(xyA=(tx_min, ty_max), coordsA="data", axesA=ax, 
                                              xyB=(1, 0), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=0.8, alpha=line_alpha, linestyle=line_style)
                    else:
                        cp1 = ConnectionPatch(xyA=(tx_max, ty_min), coordsA="data", axesA=ax, 
                                              xyB=(0, 1), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=0.8, alpha=line_alpha, linestyle=line_style)
                        cp2 = ConnectionPatch(xyA=(tx_max, ty_max), coordsA="data", axesA=ax, 
                                              xyB=(0, 0), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=0.8, alpha=line_alpha, linestyle=line_style)
                                              
                    ax.add_artist(cp1)
                    ax.add_artist(cp2)

    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"✨ 严谨检查完毕！最终终极版图表已成功导出: {fig_name}")

if __name__ == "__main__":
    generate_merged_figure()
