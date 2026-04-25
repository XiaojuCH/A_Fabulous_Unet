import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ================= 1. 核心配置区域 =================
DATA_ROOT = "./results"

TRACK1_IDS = ['Color2_000668', 'Infrared2_000645']
TRACK2_IDS = ['Color2_000098', 'Infrared2_000015']

# 你的坐标
ARROWS = {
    "Color2_000668":   {'fail': (250, 480), 'success': (250, 480)},
    "Infrared2_000645": {'fail': (400, 450), 'success': (400, 450)},
    "Color2_000098":   {'fail': (200, 460), 'success': (200, 460)},
    "Infrared2_000015": {'fail': (280, 510), 'success': (280, 510)},
}
# ===================================================

def find_image_path(folder_path, img_id):
    search_pattern = os.path.join(folder_path, f"{img_id}.*")
    matches = glob.glob(search_pattern)
    for m in matches:
        if m.lower().endswith(('.png', '.jpg', '.jpeg')):
            return m
    return None

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
    return cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)

def draw_arrow(ax, coords, color='red'):
    if not coords or coords == (0, 0): return
    # 箭头稍微靠近目标点，防止飞出去
    ax.annotate('', xy=coords, xytext=(coords[0]+35, coords[1]-35),
                arrowprops=dict(facecolor=color, edgecolor='white', shrink=0.05, width=2.5, headwidth=10))

def generate_merged_figure():
    fig_name = "Fig_2_Dual_Mode_Comparison.pdf"
    print(f"🚀 正在生成终极排版图表: {fig_name}")
    
    # 【修复1】调整 figsize，契合 3:1 的宽屏图片，防止被垂直挤压
    fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(28, 8.5), gridspec_kw={'wspace': 0.02, 'hspace': 0.2})
    
    track1_cols = [("Input (YOLO Box)", None), ("GT", "masks_gt"), ("U-Net", "masks_unet"), ("Swin-UNETR", "masks_swinunet"), ("DeepLabV3+", "masks_deeplab"), ("SAM2 Base", "masks_baseline_sam"), ("ST-SAM (Ours)", "masks_stsam")]
    track2_cols = [("Input (Expert BBox)", None), ("GT", "masks_gt"), ("MedSAM", "masks_medsam"), ("SAM2 Base", "masks_baseline_sam"), ("SAM2 LoRA", "masks_lora"), ("SAM2 MSA", "masks_msa"), ("ST-SAM (Ours)", "masks_stsam")]
    
    all_ids = TRACK1_IDS + TRACK2_IDS
    row_labels = [
        "[Automated]\nColour", "[Automated]\nInfrared", 
        "[Expert-Guided]\nColour", "[Expert-Guided]\nInfrared"
    ]
    
    # 【修复2】强制所有图片裁剪高度一致 (350像素)
    CROP_H = 350 

    for row_idx, img_id in enumerate(all_ids):
        is_track1 = row_idx < 2
        columns = track1_cols if is_track1 else track2_cols
        box_color = 'yellow' if is_track1 else 'cyan'
        
        mask_gt_path = find_image_path(f"{DATA_ROOT}/masks_gt", img_id)
        if mask_gt_path:
            mask_gt = cv2.imread(mask_gt_path, 0)
            ys, xs = np.where(mask_gt > 127)
            if len(ys) > 0:
                center_y = (ys.min() + ys.max()) // 2
            else:
                center_y = 512
        else:
            ys, xs = [], []
            center_y = 512

        # 强制上下裁剪，保证高度恒定为 CROP_H
        crop_y_min = max(0, center_y - CROP_H // 2)
        crop_y_max = crop_y_min + CROP_H
        if crop_y_max > 1024: # 假设原图 1024 高
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
                
            # X 轴不裁剪，Y 轴强制裁剪到 CROP_H
            if col_idx == 0:
                img_to_show = img[crop_y_min:crop_y_max, :]
            else:
                mask_path = find_image_path(f"{DATA_ROOT}/{model_folder}", img_id) if model_folder else None
                img_to_show = overlay_mask(img_path, mask_path)[crop_y_min:crop_y_max, :]
            
            # 【修复3】强制图片按照一致比例渲染
            ax.imshow(img_to_show, aspect='auto')
            
            # 绘制提示框
            if col_idx in [0, 1] and len(ys) > 0:
                pad = 20
                box_x = max(0, xs.min() - pad)
                box_y = max(0, ys.min() - pad) - crop_y_min 
                box_w = (xs.max() - xs.min()) + pad * 2
                box_h = (ys.max() - ys.min()) + pad * 2
                rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=3, edgecolor=box_color, facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
            # ================= 画中画 (Inset Zoom) 渲染 =================
            f_c = ARROWS.get(img_id, {}).get('fail', (0,0))
            s_c = ARROWS.get(img_id, {}).get('success', (0,0))
            target_pt = None
            zoom_color = 'white'
            
            # 因为 X 没有裁剪，所以 X 坐标直接用；Y 坐标需要减去 crop_y_min
            if col_idx in range(2, len(columns)-1) and f_c != (0,0):
                target_pt = (f_c[0], f_c[1] - crop_y_min)
                zoom_color = '#FF3333'
            elif col_idx == len(columns)-1 and s_c != (0,0):
                target_pt = (s_c[0], s_c[1] - crop_y_min)
                zoom_color = '#00FF00'
            elif col_idx in [0, 1]: 
                if s_c != (0,0): target_pt = (s_c[0], s_c[1] - crop_y_min)
                elif f_c != (0,0): target_pt = (f_c[0], f_c[1] - crop_y_min)

            if target_pt is not None:
                if zoom_color in ['#FF3333', '#00FF00']: 
                    draw_arrow(ax, target_pt, zoom_color)

                # 【修复4】调整放大镜的长宽比例，使其在宽屏下依然呈现正方形
                axins = inset_axes(ax, width="25%", height="75%", loc=1, borderpad=0.5)
                axins.imshow(img_to_show, aspect='auto')
                
                zoom_size = 150
                axins.set_xlim(target_pt[0] - zoom_size//2, target_pt[0] + zoom_size//2)
                axins.set_ylim(target_pt[1] + zoom_size//2, target_pt[1] - zoom_size//2)
                axins.set_xticks([]); axins.set_yticks([])
                
                for spine in axins.spines.values():
                    spine.set_edgecolor(zoom_color)
                    spine.set_linewidth(2.5)
                mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec=zoom_color, lw=1.5, alpha=0.8)

    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✨ 顶级二合一排版生成完毕: {fig_name}")

if __name__ == "__main__":
    generate_merged_figure()