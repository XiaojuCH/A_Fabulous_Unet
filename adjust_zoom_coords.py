import cv2
import numpy as np
import os
import glob

DATA_ROOT = "./results"
CROP_H = 350
ZOOM_SIZE = 150

IMAGES = {
    "Color1_000160":    {'fail': (450, 750), 'success': (450, 750)},
    "Infrared3_000579": {'fail': (400, 760), 'success': (400, 760)},
    "Color2_000050":    {'fail': (850, 760), 'success': (850, 760)},
    "Infrared3_000388": {'fail': (750, 730), 'success': (750, 730)},
}

TRACK1_COLS = [
    ("Input",       None),
    ("GT",          "masks_gt"),
    ("Swin-UNETR",  "masks_swinunet"),
    ("DeepLabV3+",  "masks_deeplab"),
    ("SAM2 Base",   "masks_baseline_sam_yolo"),
    ("ST-SAM",      "masks_stsam_yolo"),
]
TRACK2_COLS = [
    ("Input",       None),
    ("GT",          "masks_gt"),
    ("MedSAM",      "masks_medsam_gt"),
    ("SAM2 Base",   "masks_baseline_sam_gt"),
    ("SAM2 MSA",    "masks_msa_gt"),
    ("ST-SAM",      "masks_stsam_gt"),
]

def find_image_path(folder, img_id):
    matches = glob.glob(os.path.join(folder, f"{img_id}.*"))
    return matches[0] if matches else None

def get_crop_y(img_id):
    mask_path = find_image_path(f"{DATA_ROOT}/masks_gt", img_id)
    if mask_path:
        mask = cv2.imread(mask_path, 0)
        ys, _ = np.where(mask > 127)
        center_y = int((ys.min() + ys.max()) // 2) if len(ys) > 0 else 512
    else:
        center_y = 512
    crop_y_min = max(0, center_y - CROP_H // 2)
    if crop_y_min + CROP_H > 1024:
        crop_y_min = 1024 - CROP_H
    return crop_y_min

def overlay_mask(img_bgr, mask_path, color=(0, 255, 0), alpha=0.55):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
    if img is None:
        return np.zeros((CROP_H, 1024, 3), dtype=np.uint8)
    if not mask_path or not os.path.exists(mask_path):
        return img
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return img
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored = np.zeros_like(img)
    colored[mask > 127] = color
    return cv2.addWeighted(img, 1.0, colored, alpha, 0)

# shared state
pt = [0, 0]          # current point in cropped coords
dragging = False
panels = []          # list of cropped RGB images (numpy)
col_titles = []
canvas_w = 0
PANEL_W = 0

COLS_PER_ROW = 2

def make_canvas():
    rows = []
    for row_start in range(0, len(panels), COLS_PER_ROW):
        row_panels = panels[row_start:row_start + COLS_PER_ROW]
        row_titles = col_titles[row_start:row_start + COLS_PER_ROW]
        strips = []
        for i, img in enumerate(row_panels):
            strip = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            x1 = max(0, pt[0] - ZOOM_SIZE // 2)
            y1 = max(0, pt[1] - ZOOM_SIZE // 2)
            x2 = min(PANEL_W - 1, pt[0] + ZOOM_SIZE // 2)
            y2 = min(CROP_H - 1, pt[1] + ZOOM_SIZE // 2)
            cv2.rectangle(strip, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(strip, (pt[0], pt[1]), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
            cv2.putText(strip, row_titles[i], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            strips.append(strip)
        # pad last row if odd number of panels
        while len(strips) < COLS_PER_ROW:
            strips.append(np.zeros_like(strips[0]))
        rows.append(np.concatenate(strips, axis=1))
    return np.concatenate(rows, axis=0)

def mouse_cb(event, x, y, _flags, _param):
    global dragging
    if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and (event == cv2.EVENT_LBUTTONDOWN or dragging):
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
        col = x // PANEL_W
        row = y // CROP_H
        if col >= COLS_PER_ROW or row >= (len(panels) + COLS_PER_ROW - 1) // COLS_PER_ROW:
            return
        pt[0] = x - col * PANEL_W
        pt[1] = y - row * CROP_H
        cv2.imshow("Adjust", make_canvas())
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def run():
    global panels, col_titles, canvas_w, PANEL_W, pt

    img_ids = list(IMAGES.keys())
    results = {}

    for img_id in img_ids:
        img_path = find_image_path(f"{DATA_ROOT}/images", img_id)
        if not img_path:
            print(f"[skip] {img_id}: image not found")
            continue

        is_track1 = img_id.startswith("Color1") or img_id.startswith("Infrared3_000579")
        columns = TRACK1_COLS if is_track1 else TRACK2_COLS

        crop_y_min = get_crop_y(img_id)
        img_bgr_full = cv2.imread(img_path)

        panels = []
        col_titles = []
        for title, folder in columns:
            if folder is None:
                cropped_rgb = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)[crop_y_min:crop_y_min+CROP_H, :]
            else:
                m_path = find_image_path(f"{DATA_ROOT}/{folder}", img_id)
                full_rgb = overlay_mask(img_bgr_full, m_path)
                cropped_rgb = full_rgb[crop_y_min:crop_y_min+CROP_H, :]
            panels.append(cropped_rgb)
            col_titles.append(title)

        PANEL_W = panels[0].shape[1]
        canvas_w = PANEL_W * len(panels)

        init = IMAGES[img_id]['fail']
        pt[0] = max(0, min(init[0], PANEL_W - 1))
        pt[1] = max(0, min(init[1] - crop_y_min, CROP_H - 1))

        win = "Adjust"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, mouse_cb)
        cv2.imshow(win, make_canvas())

        print(f"\n[{img_id}]  drag to move box | S=save | N=next(skip)")
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s'):
                real = (pt[0], pt[1] + crop_y_min)
                results[img_id] = real
                print(f"  Saved: {real}")
                break
            elif key in (ord('n'), ord('q')):
                print(f"  Skipped")
                break
            # refresh coords display in title
            cv2.setWindowTitle(win, f"Adjust: {img_id}  |  x={pt[0]}, y={pt[1]+crop_y_min}  (S=save  N=skip)")

    cv2.destroyAllWindows()

    print("\n\n========== Paste into FINAL_PAPER_FIG_NO_UNET_4_27.py ==========")
    print("ARROWS = {")
    for img_id in img_ids:
        coord = results.get(img_id, IMAGES[img_id]['fail'])
        print(f'    "{img_id}":    {{\'fail\': {coord}, \'success\': {coord}}},')
    print("}")

if __name__ == "__main__":
    run()
