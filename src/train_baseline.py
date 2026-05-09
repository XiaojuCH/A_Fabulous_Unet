import os
import argparse
import json
import time
import datetime
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 引入 MONAI 模型库 ===
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, SegResNet, BasicUNetPlusPlus
# === 引入 Torchvision 模型库 (用于 DeepLab) ===
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, fcn_resnet50, FCN_ResNet50_Weights
import segmentation_models_pytorch as smp
from dataset import TearDataset

# ================= 损失函数 (与 train.py 完全一致) =================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)
        return 0.5 * bce_loss + 0.5 * dice_loss

# ================= 配置 =================
IMG_SIZE = 1024  # 必须保持一致

def setup_ddp():
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup():
    if "RANK" in os.environ: destroy_process_group()

def get_model(name):
    """加载 Baseline 模型"""
    if name == "unet":
        return UNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2,
        )
    elif name == "swinunet":
        return SwinUNETR(
            in_channels=3, out_channels=1,
            feature_size=48, spatial_dims=2, use_v2=True, window_size=8
        )
    elif name == "attentionunet":
        return AttentionUnet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
        )
    elif name == "segresnet":
        return SegResNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1]
        )
    # === 新增 Q1 强力 Baseline: UNet++ ===
    elif name == "unetplusplus":
        return BasicUNetPlusPlus(
            spatial_dims=2, in_channels=3, out_channels=1,
            features=(16, 32, 64, 128, 256, 256),  # 减小通道数，降低显存占用
            deep_supervision=False
        )
    # === 新增 Q1 强力 Baseline: DeepLabV3+ ===
    elif name == "deeplab":
        # num_classes=1 会自动重置最后的全连接层
        return deeplabv3_resnet50(weights=None, num_classes=1)
    
    elif name == "deeplab_p":
        # ====================================================
        # 🔥 修改这里：加载 ImageNet 预训练权重
        # ====================================================
        # 1. 加载带预训练权重的模型 (默认是 COCO/ImageNet 预训练)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        m = deeplabv3_resnet50(weights=weights)
        
        # 2. 修改分类头 (Cls Head)
        # 预训练模型输出是 21 类 (VOC) 或 91 类 (COCO)，我们要改成 1 类 (二分类)
        # DeepLabV3 的分类头在 classifier[4] 和 aux_classifier[4]
        m.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        m.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        
        return m
    
    elif name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            encoder_output_stride=8,
        )
    elif name == "fcn":
        weights = FCN_ResNet50_Weights.DEFAULT
        m = fcn_resnet50(weights=weights)
        m.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        m.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        return m

    else:
        raise ValueError(f"Unknown model: {name}")

def log_to_csv(stats, filename="baseline_summary.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(stats)

def plot_training_curves(train_losses, val_dices, save_path):
    """绘制训练曲线：loss和dice"""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    ax1.plot(train_losses, label='Train Loss', color='#2E86AB', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Dice曲线
    epochs_with_val = [i for i, d in enumerate(val_dices) if d is not None]
    dice_values = [d for d in val_dices if d is not None]
    ax2.plot(epochs_with_val, dice_values, label='Val Dice', color='#A23B72', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main(args):
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)
    
    # 1. 数据集
    split_path = f"./data_splits/fold_{args.fold}.json"
    with open(split_path, 'r') as f: split_data = json.load(f)
    
    # 这里直接调用新的 dataset.py，它会自动去掉瞳孔
    train_ds = TearDataset(split_data['train'], mode='train', img_size=IMG_SIZE)
    val_ds = TearDataset(split_data['val'], mode='val', img_size=IMG_SIZE)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # 2. 模型
    model = get_model(args.model).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # UNet++返回list，只用第一个输出，需要find_unused_parameters=True
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 3. 优化器 (统一用 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = DiceBCELoss().to(local_rank)
    scaler = GradScaler('cuda')

    best_dice = 0.0
    save_dir = f"./checkpoints_New_baseline/{args.model}/fold_{args.fold}"
    if is_master: os.makedirs(save_dir, exist_ok=True)

    # 记录训练历史
    train_losses = []
    val_dices = []

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, disable=not is_master, desc=f"[{args.model.upper()}] Fold {args.fold} Ep {epoch}")

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(pbar):
            img, lbl = batch['image'].to(local_rank), batch['label'].to(local_rank)

            with autocast('cuda'):
                output = model(img)

                # 🔥【关键修复】处理不同模型的输出格式
                if isinstance(output, dict) and 'out' in output:
                    # DeepLab 返回字典 {'out': tensor, 'aux': tensor}
                    logits = output['out']
                    # 训练时计算辅助loss（权重0.4）
                    loss = loss_fn(logits, lbl)
                    if 'aux' in output:
                        aux_loss = loss_fn(output['aux'], lbl)
                        loss = loss + 0.4 * aux_loss
                elif isinstance(output, list):
                    # UNet++ 可能返回list，取第一个（主输出）
                    logits = output[0]
                    loss = loss_fn(logits, lbl)
                else:
                    logits = output
                    loss = loss_fn(logits, lbl)
                # 梯度累积：loss需要除以累积步数
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            # 每accumulation_steps步才更新一次权重
            if (batch_idx + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.accumulation_steps  # 记录真实loss
            num_batches += 1
            pbar.set_postfix(loss=loss.item() * args.accumulation_steps)

        # 记录平均loss
        avg_loss = epoch_loss / num_batches
        if is_master:
            train_losses.append(avg_loss)

        # Validation - 每个epoch都做validation（与train.py保持一致）
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for batch in val_loader:
                img, lbl = batch['image'].to(local_rank), batch['label'].to(local_rank)
                with autocast('cuda'):
                    output = model(img)

                    # 🔥【关键修复】处理不同模型的输出格式
                    if isinstance(output, dict) and 'out' in output:
                        logits = output['out']
                    elif isinstance(output, list):
                        logits = output[0]
                    else:
                        logits = output

                    pred = (torch.sigmoid(logits) > 0.5).float()

                # 简单 Dice 计算
                inter = (pred * lbl).sum()
                dice = (2 * inter) / (pred.sum() + lbl.sum() + 1e-6)
                val_dice += dice.item()

        val_dice_t = torch.tensor(val_dice).to(local_rank)
        all_reduce(val_dice_t, op=ReduceOp.SUM)
        avg_dice = val_dice_t.item() / (len(val_loader) * world_size)

        if is_master:
            val_dices.append(avg_dice)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Val Dice: {avg_dice:.4f}")

            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(model.module.state_dict(), f"{save_dir}/best_model.pth")
                print(f"🔥 New Best Dice ({args.model}): {best_dice:.4f}")
    
    if is_master:
        duration = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        log_to_csv({
            "model": args.model,
            "fold": args.fold,
            "best_dice": best_dice,
            "duration": duration,
            "lr": args.lr
        })

        # 生成训练曲线图
        plot_path = f"{save_dir}/fold{args.fold}_training_curves.png"
        plot_training_curves(train_losses, val_dices, plot_path)
        print(f"📊 Training curves saved to {plot_path}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    # 🔥 加入了新模型
    parser.add_argument("--model", type=str, required=True, choices=["unet", "swinunet","attentionunet", "segresnet", "deeplab", "unetplusplus", "deeplab_p","fcn","deeplabv3plus"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) # 统一用 1e-4 保证公平
    parser.add_argument("--epochs", type=int, default=100) # Baseline 跑满 100 轮
    parser.add_argument("--accumulation_steps", type=int, default=1) # 梯度累积步数，默认1（不累积）
    args = parser.parse_args()
    main(args)