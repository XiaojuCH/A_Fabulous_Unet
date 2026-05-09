import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
import sys

# 确保能找到你的 dataset 和 model
sys.path.append("src") 
from model import ST_SAM
from monai.metrics import compute_dice

# ======= 配置 =======
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "./checkpoints_run1"  # 你的 ST-SAM 权重路径

def load_st_sam(fold):
    ckpt_path = f"{CKPT_DIR}/fold_{fold}/best_model.pth"
    if not os.path.exists(ckpt_path):
        return None
    model = ST_SAM().to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.eval()
    return model

def main():
    print("🚀 启动 [YOLOv8n Fallback 机制] 定量评估")
    
    total_images = 0
    missed_images = 0
    fallback_dices = []
    
    for fold in range(5):
        split_json = f"./data_splits/fold_{fold}.json"
        yolo_json = f"./data_splits/yolo_boxes_fold{fold}.json"
        
        if not os.path.exists(split_json) or not os.path.exists(yolo_json):
            continue
            
        with open(split_json, 'r') as f: split_data = json.load(f)
        with open(yolo_json, 'r') as f: yolo_preds = json.load(f)
        
        model = load_st_sam(fold)
        if model is None:
            print(f"⚠️ 找不到 Fold {fold} 的权重，跳过...")
            continue
            
        val_list = split_data['val']
        
        for item in tqdm(val_list, desc=f"Scanning Fold {fold}", leave=False):
            total_images += 1
            img_id = item['id']
            
            is_missed = False
            
            # 1. 兼容性获取框数据
            box_norm = yolo_preds.get(img_id) or yolo_preds.get(str(img_id))
            
            # 2. 如果彻底没这个键，或者输出的是空列表 []
            if box_norm is None or len(box_norm) < 4:
                is_missed = True
            else:
                # 3. 如果输出的是极其离谱的框或 [0,0,0,0]
                x1, y1, x2, y2 = box_norm
                w = (x2 - x1) * IMG_SIZE
                h = (y2 - y1) * IMG_SIZE
                if x2 <= x1 or y2 <= y1 or w < 10 or h < 10:
                    is_missed = True

            if is_missed:
                missed_images += 1
                # 💥 触发兜底机制 (Fallback)：生成覆盖全图的全局边界框
                fallback_box = torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=torch.float32).to(DEVICE)
                
                # 读取图像和标签
                img_path = item['image']
                label_path = item['label'].replace("/Label/", "/Cleaned_Label/")
                
                image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
                label = Image.open(label_path).convert("L").resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
                
                img_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)
                lbl_np = (np.array(label) > 127).astype(np.float32)
                lbl_tensor = torch.tensor(lbl_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                # 使用全局框进行 ST-SAM 推理
                with torch.no_grad():
                    logits = model(img_tensor, fallback_box)
                    pred = (torch.sigmoid(logits) > 0.5).float()
                
                # 计算恢复后的 Dice
                if lbl_tensor.sum() == 0 and pred.sum() == 0:
                    dice = 1.0
                else:
                    dice = compute_dice(pred, lbl_tensor, include_background=False).item()
                    if np.isnan(dice): dice = 0.0
                    
                fallback_dices.append(dice)

    # 打印最终报告
    print("\n" + "="*50)
    print("📊 Fallback 机制鲁棒性定量报告")
    print("="*50)
    print(f"总测试样本数     : {total_images}")
    print(f"YOLOv8n 漏检数   : {missed_images}")
    
    if total_images > 0:
        failure_rate = (missed_images / total_images) * 100
        print(f"检测失败率 (FNR) : {failure_rate:.2f}%")
        
    if missed_images > 0:
        avg_recovered_dice = np.mean(fallback_dices)
        print(f"触发兜底后恢复 Dice: {avg_recovered_dice:.4f}")
    else:
        print("🎉 太强了，YOLO 居然一张都没漏检！(那就不用提 Fallback 恢复了)")
    print("="*50)

if __name__ == "__main__":
    main()