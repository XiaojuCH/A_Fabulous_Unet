import sys
import os
import time
import torch
sys.path.append("src")

from model import ST_SAM, LoRA_SAM2, MSA_Baseline_SAM2, MedSAM_SAM2, Baseline_SAM2
from medsam_model import True_MedSAM
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, SegResNet, BasicUNetPlusPlus
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, FCN_ResNet50_Weights, DeepLabV3_ResNet50_Weights
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
WARMUP = 20
RUNS = 100


def measure_fps(model, has_box=True):
    model.eval()
    img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    box = torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]]).float().to(DEVICE)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(img, box) if has_box else model(img)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(RUNS):
            _ = model(img, box) if has_box else model(img)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0
    return RUNS / elapsed, vram_mb


SAM2_MODELS = {
    "ST_SAM":           lambda: ST_SAM(),
    "Baseline_SAM2":    lambda: Baseline_SAM2(),
    "LoRA_SAM2":        lambda: LoRA_SAM2(),
    "MSA_Baseline_SAM2":lambda: MSA_Baseline_SAM2(),
    "MedSAM_SAM2":      lambda: MedSAM_SAM2(),
    "True_MedSAM":      lambda: True_MedSAM(checkpoint_path="./checkpoints/medsam_vit_b.pth"),
}

BASELINE_MODELS = {
    "UNet":             lambda: UNet(spatial_dims=2, in_channels=3, out_channels=1, channels=(32,64,128,256,512), strides=(2,2,2,2), num_res_units=2),
    "AttentionUnet":    lambda: AttentionUnet(spatial_dims=2, in_channels=3, out_channels=1, channels=(32,64,128,256,512), strides=(2,2,2,2)),
    "SwinUNet":         lambda: SwinUNETR(in_channels=3, out_channels=1, feature_size=48, spatial_dims=2, use_v2=True, window_size=8),
    "SegResNet":        lambda: SegResNet(spatial_dims=2, in_channels=3, out_channels=1, init_filters=32, blocks_down=[1,2,2,4], blocks_up=[1,1,1]),
    "UNet++":           lambda: BasicUNetPlusPlus(spatial_dims=2, in_channels=3, out_channels=1, features=(16,32,64,128,256,256), deep_supervision=False),
    "DeepLabV3":        lambda: deeplabv3_resnet50(weights=None, num_classes=1),
    "FCN":              lambda: fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT),
    "DeepLabV3+":       lambda: smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1),
}


if __name__ == "__main__":
    print(f"Device: {DEVICE} | Warmup: {WARMUP} | Runs: {RUNS}\n")
    print(f"{'Model':<22} {'FPS':>8} {'VRAM(MB)':>10}")
    print("-" * 44)

    for name, builder in SAM2_MODELS.items():
        try:
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model = builder().to(DEVICE)
            fps, vram = measure_fps(model, has_box=True)
            print(f"{name:<22} {fps:>8.2f} {vram:>10.0f}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{name:<22} {'ERROR':>8}  ({e})")

    print()

    for name, builder in BASELINE_MODELS.items():
        try:
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model = builder().to(DEVICE)
            fps, vram = measure_fps(model, has_box=False)
            print(f"{name:<22} {fps:>8.2f} {vram:>10.0f}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{name:<22} {'ERROR':>8}  ({e})")
