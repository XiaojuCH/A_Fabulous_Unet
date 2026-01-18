import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np

# 引入你的项目模块
from models.unet import UNet
from utils.dataset import TearMeniscusDataset
from utils.losses import DiceBCELoss
from utils.transforms import JointCompose, JointResize
from utils.metrics import (
    dice_coefficient, iou_score, accuracy_score,
    precision_score, recall_score, specificity_score,
    compute_hd95
)

def evaluate(model, dataloader, device):
    model.eval()
    
    metrics = {
        'dice': [], 'iou': [], 'acc': [], 
        'prec': [], 'recall': [], 'spec': [], 'hd95': []
    }
    
    print("正在计算各项指标 (HD95 计算较慢，请耐心等待)...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Testing'):
            images = images.to(device)
            masks = masks.to(device)

            with autocast():
                outputs = model(images)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # 转为 numpy 用于计算 HD95
            pred_np = preds.cpu().numpy().squeeze(1)
            mask_np = masks.cpu().numpy().squeeze(1)

            # 逐张计算指标
            for i in range(images.size(0)):
                # 基础指标 (Tensor计算快)
                p = preds[i]
                m = masks[i]
                metrics['dice'].append(dice_coefficient(p, m).item())
                metrics['iou'].append(iou_score(p, m).item())
                metrics['acc'].append(accuracy_score(p, m).item())
                metrics['prec'].append(precision_score(p, m).item())
                metrics['recall'].append(recall_score(p, m).item())
                metrics['spec'].append(specificity_score(p, m).item())
                
                # HD95 (Numpy计算慢)
                # 只有当GT和预测都有前景时才计算，否则跳过或给惩罚
                if np.sum(pred_np[i]) > 0 and np.sum(mask_np[i]) > 0:
                    metrics['hd95'].append(compute_hd95(pred_np[i], mask_np[i]))

    # 计算平均值
    final_metrics = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}
    return final_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🚀 Testing on device: {device}')

    # 1. 准备测试集数据
    print('📦 Loading Test Dataset...')
    test_joint = JointCompose([JointResize((args.img_size, args.img_size))])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([transforms.ToTensor()])

    # 实例化数据集
    # 注意：这里我们重新加载一遍数据来划分，确保和训练时一致（依赖随机种子）
    # 如果你想快一点，可以直接只加载一部分，但为了严谨，我们复现训练时的划分逻辑
    ds_obj = TearMeniscusDataset(args.data_root, joint_transform=test_joint, transform=img_transform, target_transform=mask_transform)
    
    total_len = len(ds_obj)
    indices = torch.randperm(total_len, generator=torch.Generator().manual_seed(42)).tolist()
    
    train_len = int(0.7 * total_len)
    val_len = int(0.1 * total_len)
    test_indices = indices[train_len + val_len :]
    
    test_set = Subset(ds_obj, test_indices)
    print(f'   Test Set Size: {len(test_set)}')

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 2. 初始化模型
    model = UNet(n_channels=3, n_classes=1).to(device)

    # 3. 万能加载逻辑 (核心修复)
    ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    print(f'🔓 Loading model from {ckpt_path}...')
    
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 情况A: 这是一个包含 'epoch', 'model_state_dict' 的完整字典
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("   -> Detected full checkpoint dict.")
        
        # 情况B: 这直接就是权重字典 (KeyError的原因通常是这个)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            print("   -> Detected raw state_dict.")
        else:
            raise ValueError("Unknown checkpoint format")

        # 处理 'module.' 前缀 (防止 DataParallel 带来的 key 不匹配)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # 去掉 'module.'
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 开启多卡加速测试 (可选)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 4. 开始测试
    metrics = evaluate(model, test_loader, device)

    # 5. 输出最终报表
    print('\n' + '='*40)
    print('📄 FINAL SCI REPORT (Test Set Results)')
    print('='*40)
    print(f"Dice (DSC):    {metrics['dice']:.4f}")
    print(f"IoU (Jaccard): {metrics['iou']:.4f}")
    print(f"Accuracy:      {metrics['acc']:.4f}")
    print(f"Precision:     {metrics['prec']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"Specificity:   {metrics['spec']:.4f}")
    print(f"HD95 (px):     {metrics['hd95']:.4f}")
    print('='*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=768) # 保持和你训练时一致
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    main(args)