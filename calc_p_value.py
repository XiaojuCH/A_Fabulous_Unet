import os
import json
import torch
import numpy as np
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append("src")
from model import ST_SAM, MSA_Baseline_SAM2
from dataset import TearDataset
from monai.metrics import compute_dice

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024

def load_weights(model, path):
    if not os.path.exists(path): return None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
    model.eval()
    return model

def main():
    print("🚀 启动 ST-SAM vs MSA 统计显著性 (p-value) 检验...")
    
    st_sam_dices = []
    msa_dices = []
    
    for fold in range(5):
        json_path = f"./data_splits/fold_{fold}.json"
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: data = json.load(f)
        
        dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE, yolo_pred_json=f"./data_splits/yolo_boxes_fold{fold}.json")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        st_sam = load_weights(ST_SAM().to(DEVICE), f"./checkpoints_run1/fold_{fold}/best_model.pth")
        msa = load_weights(MSA_Baseline_SAM2().to(DEVICE), f"./checkpoints_msa/fold_{fold}/best_model.pth")
        if st_sam is None or msa is None: continue

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Fold {fold}", leave=False):
                img, lbl, box = batch['image'].to(DEVICE), batch['label'].to(DEVICE), batch['box'].to(DEVICE)
                
                pred_st = (torch.sigmoid(st_sam(img, box)) > 0.5).float()
                pred_msa = (torch.sigmoid(msa(img, box)) > 0.5).float()
                
                # 计算 Dice
                dice_st = compute_dice(pred_st, lbl).item() if lbl.sum()>0 else 1.0
                dice_msa = compute_dice(pred_msa, lbl).item() if lbl.sum()>0 else 1.0
                
                st_sam_dices.append(dice_st)
                msa_dices.append(dice_msa)

    # 配对 Wilcoxon 符号秩检验
    stat, p_value = wilcoxon(st_sam_dices, msa_dices)
    
    print("\n" + "="*50)
    print(f"✅ ST-SAM 均值: {np.mean(st_sam_dices):.4f}")
    print(f"✅ MSA 均值:    {np.mean(msa_dices):.4f}")
    print(f"🔥 Wilcoxon P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("🎉 结论: 差异在统计学上【显著】(p < 0.05)！可以直接标在表1上！")
    else:
        print("⚠️ 结论: 差异不显著 (p >= 0.05)。需要在正文中改变论述侧重点。")
    print("="*50)

if __name__ == "__main__":
    main()