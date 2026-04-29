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
# ===================================================

def find_image_path(folder_path, img_id):
    search_pattern = os.path.join(folder_path, f"{img_id}.*")
    matches = glob.glob(search_pattern)
    return matches[0] if matches else None

def overlay_mask(image_path, mask_path, color=(0, 255, 0), alpha=0.55):
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

def draw_arrow(ax, coords, color='red'):
    if not coords or coords == (0, 0): return
    ax.annotate('', xy=coords, xytext=(coords[0]+35, coords[1]-35),
                arrowprops=dict(facecolor=color, edgecolor='white', shrink=0.05, width=2.5, headwidth=10))

def generate_merged_figure():
    fig_name = "Fig_2_Ultimate_4x6_Comparison.pdf"
    print(f"🚀 正在生成剔除了 U-Net 的终极 4x6 顶刊排版...")
    
    # 🚀 优化：移除 layout='constrained'，拿回绝对排版控制权
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 8.5))
    # 强制锁定极其微小的列间距和适当的行间距，确保每一列宽度绝对均等
    plt.subplots_adjust(wspace=0.02, hspace=0.05, left=0.05, right=0.98, top=0.92, bottom=0.05)
    
    track1_cols = [
        ("Input (YOLO)", None), ("GT", "masks_gt"), 
        ("Swin-UNETR", "masks_swinunet"), ("DeepLabV3", "masks_deeplab"), 
        ("SAM2 Base", "masks_baseline_sam_yolo"), ("ST-SAM (Ours)", "masks_stsam_yolo")
    ]
    track2_cols = [
        ("Input (Expert)", None), ("GT", "masks_gt"), 
        ("MedSAM", "masks_medsam_gt"), ("SAM2 Base", "masks_baseline_sam_gt"), 
        ("SAM2 MSA", "masks_msa_gt"), ("ST-SAM (Ours)", "masks_stsam_gt")
    ]
    
    all_ids = TRACK1_IDS + TRACK2_IDS
    row_labels = [
        "[Automated]\nColour", "[Automated]\nInfrared", 
        "[Expert-Guided]\nColour", "[Expert-Guided]\nInfrared"
    ]
    CROP_H = 350 

    for row_idx, img_id in enumerate(all_ids):
        is_track1 = row_idx < 2
        columns = track1_cols if is_track1 else track2_cols
        box_color = 'yellow' if is_track1 else 'cyan'
        
        mask_gt_path = find_image_path(f"{DATA_ROOT}/masks_gt", img_id)
        if mask_gt_path:
            mask_gt = cv2.imread(mask_gt_path, 0)
            ys, xs = np.where(mask_gt > 127)
            center_y = (ys.min() + ys.max()) // 2 if len(ys) > 0 else 512
        else:
            ys, xs = [], []
            center_y = 512

        crop_y_min = max(0, center_y - CROP_H // 2)
        crop_y_max = crop_y_min + CROP_H
        if crop_y_max > 1024:
            crop_y_max = 1024
            crop_y_min = 1024 - CROP_H

        img_path = find_image_path(f"{DATA_ROOT}/images", img_id)
        if not img_path: continue
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
        for col_idx, (col_title, model_folder) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.axis('off')
            
            if row_idx == 0 or row_idx == 2: 
                ax.set_title(col_title, fontsize=16, fontweight='bold', pad=10)
            if col_idx == 0:
                ax.text(-0.15, 0.5, row_labels[row_idx], transform=ax.transAxes, fontsize=15, fontweight='bold', va='center', ha='right')
                
            if col_idx == 0:
                img_to_show = img[crop_y_min:crop_y_max, :]
            else:
                m_path = find_image_path(f"{DATA_ROOT}/{model_folder}", img_id) if model_folder else None
                img_to_show = overlay_mask(img_path, m_path)[crop_y_min:crop_y_max, :]
            
            ax.imshow(img_to_show)
            
            if col_idx == 0 and len(ys) > 0:
                pad = 20
                box_x = max(0, xs.min() - pad)
                box_y = max(0, ys.min() - pad) - crop_y_min
                rect = patches.Rectangle((box_x, box_y), (xs.max() - xs.min()) + pad * 2, (ys.max() - ys.min()) + pad * 2, linewidth=3, edgecolor=box_color, facecolor='none', linestyle='--')
                ax.add_patch(rect)

            f_c = ARROWS.get(img_id, {}).get('fail', (0,0))
            s_c = ARROWS.get(img_id, {}).get('success', (0,0))
            target_pt = None
            zoom_color = 'white'

            if col_idx in range(2, len(columns)-1) and f_c != (0,0):
                target_pt = (f_c[0], f_c[1] - crop_y_min)
                zoom_color = '#FF0000' # 纯红，极致醒目
            elif col_idx == len(columns)-1 and s_c != (0,0):
                target_pt = (s_c[0], s_c[1] - crop_y_min)
                zoom_color = '#00FF00' # 纯绿，极致醒目
            elif col_idx == 1 and s_c != (0,0):
                target_pt = (s_c[0], s_c[1] - crop_y_min)
                zoom_color = 'white'

            if target_pt is not None:
                if 0 <= target_pt[1] < CROP_H and 0 <= target_pt[0] < img.shape[1]:
                    draw_arrow(ax, target_pt, zoom_color)
                    
                    inset_loc = 2 if target_pt[0] > img.shape[1] // 2 else 1
                    
                    axins = inset_axes(ax, width="25%", height="75%", loc=inset_loc, borderpad=0.5)
                    axins.imshow(img_to_show, interpolation='none')
                    
                    zoom_size = ARROWS.get(img_id, {}).get('zoom', 150)
                    axins.set_xlim(target_pt[0] - zoom_size//2, target_pt[0] + zoom_size//2)
                    axins.set_ylim(target_pt[1] + zoom_size//2, target_pt[1] - zoom_size//2)
                    axins.set_xticks([]); axins.set_yticks([])
                    
                    for spine in axins.spines.values():
                        spine.set_edgecolor(zoom_color)
                        spine.set_linewidth(2.5)
                        
                    tx_min = target_pt[0] - zoom_size//2
                    tx_max = target_pt[0] + zoom_size//2
                    ty_min = target_pt[1] - zoom_size//2
                    ty_max = target_pt[1] + zoom_size//2
                    
                    # 1. 目标小框：实线，突出焦点
                    rect_target = patches.Rectangle((tx_min, ty_min), zoom_size, zoom_size, 
                                                    linewidth=1.5, edgecolor=zoom_color, facecolor='none')
                    ax.add_patch(rect_target)
                    
                    # 2. 牵引连线：虚线+半透明，减少信息遮挡，建立视觉层级
                    line_style = (0, (6, 4))  # 6pt线段，4pt间距
                    line_alpha = 0.5  
                    
                    if inset_loc == 2:
                        cp1 = ConnectionPatch(xyA=(tx_min, ty_min), coordsA="data", axesA=ax, 
                                              xyB=(1, 1), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=1.5, alpha=line_alpha, linestyle=line_style)
                        cp2 = ConnectionPatch(xyA=(tx_min, ty_max), coordsA="data", axesA=ax, 
                                              xyB=(1, 0), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=1.5, alpha=line_alpha, linestyle=line_style)
                    else:
                        cp1 = ConnectionPatch(xyA=(tx_max, ty_min), coordsA="data", axesA=ax, 
                                              xyB=(0, 1), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=1.5, alpha=line_alpha, linestyle=line_style)
                        cp2 = ConnectionPatch(xyA=(tx_max, ty_max), coordsA="data", axesA=ax, 
                                              xyB=(0, 0), coordsB="axes fraction", axesB=axins, 
                                              color=zoom_color, lw=1.5, alpha=line_alpha, linestyle=line_style)
                                              
                    ax.add_artist(cp1)
                    ax.add_artist(cp2)

    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    print(f"✨ 严谨检查完毕！最终终极版图表已成功导出: {fig_name}")

if __name__ == "__main__":
    generate_merged_figure()