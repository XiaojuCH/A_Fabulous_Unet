import os
import torch
import sys

sys.path.append("src")
from model import ST_SAM
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def measure_vram(model, model_name, dummy_inputs, is_sam=False):
    """
    严谨测算模型在纯推理阶段的峰值显存占用
    """
    device = "cuda"
    model = model.to(device)
    model.eval()
    
    # 1. 预热 (Warm-up) - 消除 CUDA 初始化带来的统计误差
    with torch.no_grad():
        for _ in range(3):
            if is_sam:
                _ = model(dummy_inputs[0], dummy_inputs[1])
            else:
                _ = model(dummy_inputs[0])
                
    # 2. 清空缓存，重置显存峰值计数器
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 3. 记录真实推理的前向传播显存
    with torch.no_grad():
        if is_sam:
            _ = model(dummy_inputs[0], dummy_inputs[1])
        else:
            _ = model(dummy_inputs[0])
            
    # 4. 获取峰值显存并转化为 MB
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024 ** 2)
    
    # 测算完毕后将模型移出显存，防止影响下一次测试
    model.cpu()
    torch.cuda.empty_cache()
    
    return peak_vram_mb

def main():
    if not torch.cuda.is_available():
        print("❌ 必须在 GPU 环境下才能测算 VRAM。")
        return
        
    print("🚀 启动前向推理峰值显存 (Inference VRAM) 严谨测算...")
    print("📌 测试条件: 单张 1024x1024 图像 (Batch Size = 1)\n")
    
    device = "cuda"
    B, C, H, W = 1, 3, 1024, 1024
    
    # 准备共享的 Dummy 数据
    dummy_img = torch.randn(B, C, H, W, device=device)
    dummy_box = torch.tensor([[100.0, 100.0, 900.0, 900.0]], device=device)
    
    # ==========================================
    # 1. 测算 DeepLabV3 (端到端 CNN 代表)
    # ==========================================
    print("⏳ 正在测算 DeepLabV3...")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    deeplab = deeplabv3_resnet50(weights=weights)
    deeplab.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    deeplab.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    deeplab_vram = measure_vram(deeplab, "DeepLabV3", [dummy_img])
    
    # ==========================================
    # 2. 测算 ST-SAM (我们的模型)
    # ==========================================
    print("⏳ 正在测算 ST-SAM (Ours)...")
    st_sam = ST_SAM()
    st_sam_vram = measure_vram(st_sam, "ST-SAM", [dummy_img, dummy_box], is_sam=True)
    
    # ==========================================
    # 打印最终对比报告
    # ==========================================
    print("\n" + "="*50)
    print(f"📊 【单张图像推理显存消耗对比 (MB)】")
    print("-" * 50)
    print(f"DeepLabV3+    : {deeplab_vram:>8.2f} MB")
    print(f"ST-SAM (Ours) : {st_sam_vram:>8.2f} MB")
    print("="*50)
    
    if st_sam_vram < 4000:
        print("\n🎉 结论: ST-SAM 的推理显存甚至不到 4GB！完全可以部署在轻薄本、一体机等低端医疗终端设备上，彻底粉碎了审稿人关于硬件开销的质疑！")

if __name__ == "__main__":
    main()