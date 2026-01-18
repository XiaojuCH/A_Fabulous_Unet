import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    """
    自适应边缘提取层 (不可学习的固定卷积，提取梯度)
    """
    def __init__(self):
        super(SobelFilter, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # x: [B, 3, H, W], 取平均变灰度
        gray = torch.mean(x, dim=1, keepdim=True) 
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return magnitude # [B, 1, H, W] Edge Map

class ChannelAttention(nn.Module):
    """
    注意力融合模块的核心组件
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class FusionBlock(nn.Module):
    """
    双流特征融合模块: 将 Texture特征 和 Edge特征 融合
    """
    def __init__(self, in_channels):
        super(FusionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(in_channels)

    def forward(self, x_texture, x_edge):
        # 简单的拼接
        cat = torch.cat([x_texture, x_edge], dim=1)
        feat = self.conv(cat)
        # 注意力加权
        att = self.ca(feat)
        return feat * att

# --- 基础 U-Net 组件 ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels) # simplified for concat
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# --- 主模型: Dual-Stream Edge-Aware U-Net ---
class DualUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DualUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 1. 边缘提取器
        self.sobel = SobelFilter()

        # 2. Texture Encoder (处理原图)
        self.inc_t = DoubleConv(n_channels, 64)
        self.down1_t = Down(64, 128)
        self.down2_t = Down(128, 256)
        self.down3_t = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4_t = Down(512, 1024 // factor)

        # 3. Edge Encoder (处理边缘图)
        self.inc_e = DoubleConv(1, 64) # 输入是单通道边缘
        self.down1_e = Down(64, 128)
        self.down2_e = Down(128, 256)
        self.down3_e = Down(256, 512)
        self.down4_e = Down(512, 1024 // factor)

        # 4. 融合模块 (在每一层都融合)
        self.fuse1 = FusionBlock(64)
        self.fuse2 = FusionBlock(128)
        self.fuse3 = FusionBlock(256)
        self.fuse4 = FusionBlock(512)
        self.fuse5 = FusionBlock(1024 // factor)

        # 5. Decoder (共享解码器)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # --- Stream 1: Edge Extraction ---
        edge = self.sobel(x) # [B, 1, H, W]

        # --- Encoders ---
        # Layer 1
        x1_t = self.inc_t(x)
        x1_e = self.inc_e(edge)
        x1_f = self.fuse1(x1_t, x1_e) # 融合特征

        # Layer 2
        x2_t = self.down1_t(x1_t)
        x2_e = self.down1_e(x1_e)
        x2_f = self.fuse2(x2_t, x2_e)

        # Layer 3
        x3_t = self.down2_t(x2_t)
        x3_e = self.down2_e(x2_e)
        x3_f = self.fuse3(x3_t, x3_e)

        # Layer 4
        x4_t = self.down3_t(x3_t)
        x4_e = self.down3_e(x3_e)
        x4_f = self.fuse4(x4_t, x4_e)

        # Layer 5 (Bottom)
        x5_t = self.down4_t(x4_t)
        x5_e = self.down4_e(x4_e)
        x5_f = self.fuse5(x5_t, x5_e)

        # --- Decoder (使用融合后的特征进行跳跃连接) ---
        x = self.up1(x5_f, x4_f) # Skip connection uses Fused feature
        x = self.up2(x, x3_f)
        x = self.up3(x, x2_f)
        x = self.up4(x, x1_f)
        logits = self.outc(x)
        
        return logits