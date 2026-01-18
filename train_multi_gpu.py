import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from models.unet import UNet
from utils.dataset import TearMeniscusDataset
from utils.losses import DiceBCELoss
from utils.transforms import JointCompose, JointResize, JointRandomHorizontalFlip, JointRandomVerticalFlip
from utils.metrics import (
    dice_coefficient, 
    iou_score, 
    accuracy_score,
    precision_score, 
    recall_score, 
    specificity_score,
    compute_hd95  # 新增
)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice = dice_coefficient(preds, masks)

        running_loss += loss.item()
        running_dice += dice.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})

    return running_loss / len(dataloader), running_dice / len(dataloader)

def evaluate(model, dataloader, criterion, device, mode='Val', calc_hd95=False):
    """
    calc_hd95: 是否计算 HD95 (非常耗时，建议只在 Test 阶段开启)
    """
    model.eval()
    
    metrics = {
        'loss': 0.0, 'dice': 0.0, 'iou': 0.0, 
        'acc': 0.0, 'prec': 0.0, 'recall': 0.0, 'spec': 0.0,
        'hd95': 0.0
    }
    
    # HD95 需要累加后取平均，因为它不能像 dice 那样在 tensor 上做 batch 计算
    hd95_list = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=mode):
            images = images.to(device)
            masks = masks.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            preds_prob = torch.sigmoid(outputs)
            preds = (preds_prob > 0.5).float()
            
            # 基础指标计算
            metrics['loss'] += loss.item()
            metrics['dice'] += dice_coefficient(preds, masks).item()
            metrics['iou'] += iou_score(preds, masks).item()
            metrics['acc'] += accuracy_score(preds, masks).item()
            metrics['prec'] += precision_score(preds, masks).item()
            metrics['recall'] += recall_score(preds, masks).item()
            metrics['spec'] += specificity_score(preds, masks).item()

            # --- HD95 计算 (如果在 Test 模式) ---
            if calc_hd95:
                # 需要把 batch 里的每一张图单独拿出来转 numpy 算
                # 这是一个 cpu 密集型操作
                pred_np = preds.cpu().numpy().squeeze(1) # (B, H, W)
                mask_np = masks.cpu().numpy().squeeze(1) # (B, H, W)
                
                for i in range(pred_np.shape[0]):
                    val = compute_hd95(pred_np[i], mask_np[i])
                    hd95_list.append(val)

    # 基础指标平均
    length = len(dataloader)
    final_metrics = {k: v / length for k, v in metrics.items()}
    
    # HD95 平均
    if calc_hd95 and len(hd95_list) > 0:
        final_metrics['hd95'] = np.mean(hd95_list)
    else:
        final_metrics['hd95'] = 0.0 # 占位

    return final_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    print(f'🚀 Device: {device} | GPUs: {gpu_count} | Resolution: {args.img_size}x{args.img_size}')

    # Transforms
    train_joint = JointCompose([
        JointResize((args.img_size, args.img_size)),
        JointRandomHorizontalFlip(),
        JointRandomVerticalFlip()
    ])
    eval_joint = JointCompose([JointResize((args.img_size, args.img_size))])
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([transforms.ToTensor()])

    print('📦 Loading and splitting dataset...')
    ds_train_obj = TearMeniscusDataset(args.data_root, joint_transform=train_joint, transform=img_transform, target_transform=mask_transform)
    ds_eval_obj = TearMeniscusDataset(args.data_root, joint_transform=eval_joint, transform=img_transform, target_transform=mask_transform)
    
    total_len = len(ds_train_obj)
    indices = torch.randperm(total_len, generator=torch.Generator().manual_seed(42)).tolist()
    
    train_len = int(0.7 * total_len)
    val_len = int(0.1 * total_len)
    
    train_set = Subset(ds_train_obj, indices[:train_len])
    val_set = Subset(ds_eval_obj, indices[train_len : train_len + val_len])
    test_set = Subset(ds_eval_obj, indices[train_len + val_len :])
    
    print(f'   Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}')

    num_workers = 16
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = UNet(n_channels=3, n_classes=1).to(device)
    if gpu_count > 1:
        model = nn.DataParallel(model)

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    scaler = GradScaler()

    best_dice = 0.0
    print('🏁 Starting training...')

    for epoch in range(1, args.epochs + 1):
        t_loss, t_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler)
        
        # 验证阶段：关闭 HD95 (calc_hd95=False)
        metrics = evaluate(model, val_loader, criterion, device, mode='Val', calc_hd95=False)
        v_dice = metrics['dice']
        
        scheduler.step(v_dice)
        
        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'   [Train] Loss: {t_loss:.4f} | Dice: {t_dice:.4f}')
        print(f'   [Val  ] Loss: {metrics["loss"]:.4f} | Dice: {v_dice:.4f} | IoU: {metrics["iou"]:.4f} | Acc: {metrics["acc"]:.4f}')
        
        if v_dice > best_dice:
            print(f'   🎉 New Best Val Dice: {v_dice:.4f} -> Saving model...')
            best_dice = v_dice
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(args.checkpoint_dir, 'best_model.pth'))

    # --- 最终测试 ---
    print('\n' + '='*50)
    print('🧪 Running Final Test (Calculating HD95, this may take a while)...')
    print('='*50)
    
    # 1. 先初始化纯净的模型
    final_model = UNet(n_channels=3, n_classes=1).to(device)
    
    # 2. 加载权重 (这时候模型还没被 DataParallel 包裹，key 是完全匹配的)
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    print(f'✅ Loaded best model from {best_model_path}')
    
    # 3. 加载完权重后，再开启多卡并行进行测试
    if gpu_count > 1:
        final_model = nn.DataParallel(final_model)
    
    # 4. 开始测试
    test_metrics = evaluate(final_model, test_loader, criterion, device, mode='Test', calc_hd95=True)
    
    print('\n📄 FINAL SCI REPORT (Test Set):')
    print('-'*30)
    print(f"Dice (DSC):    {test_metrics['dice']:.4f}")
    print(f"IoU (Jaccard): {test_metrics['iou']:.4f}")
    print(f"Accuracy:      {test_metrics['acc']:.4f}")
    print(f"Precision:     {test_metrics['prec']:.4f}")
    print(f"Recall:        {test_metrics['recall']:.4f}")
    print(f"Specificity:   {test_metrics['spec']:.4f}")
    print(f"HD95 (px):     {test_metrics['hd95']:.4f}")
    print('-'*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)