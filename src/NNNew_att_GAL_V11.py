import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False
        )
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7, reduction=4):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        pad_l = (kernel_size_large - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        pad_m = (15 - 1) // 2  # 新增：中等尺度 kernel_size=15

        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        # 新增：中等尺度条形卷积
        self.strip_h_mid = DWConv(in_channels, (15, 1), (pad_m, 0))
        self.strip_w_mid = DWConv(in_channels, (1, 15), (0, pad_m))

        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # 更新：从6分支扩展到8分支
        mid_channels = max(in_channels * 8 // reduction, 16)
        self.branch_weight = nn.Sequential(
            nn.Conv2d(in_channels * 8, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels * 8, kernel_size=1, bias=False)
        )

        # 更新：从2统计量扩展到3统计量（mean, std, max）
        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels * 2, 1)
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)

        x_oriented = self.pre_orient(x)

        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)

        # 新增：中等尺度分支
        mh = self.strip_h_mid(x_oriented)
        mw = self.strip_w_mid(x_oriented)

        loc3 = self.local_3x3(x)
        loc5 = self.local_5x5(x)

        # 更新：8个分支
        branches = [lh, lw, sh, sw, mh, mw, loc3, loc5]

        # 像素级空间门控 (Spatial Gating) - 8分支版本
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C8, H, W = weight.shape
        C = C8 // 8
        weight = weight.view(B, 8, C, H, W)
        weight = F.softmax(weight, dim=1)

        stacked = torch.stack(branches, dim=1)
        out = (weight * stacked).sum(dim=1)

        # 动态通道缩放 - 3统计量版本（mean, std, max）
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)

        # 提取3个统计特征
        feat_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1)  # [B, C, 1, 1]
        var = out_flat.var(dim=2, keepdim=True, unbiased=False)
        feat_std = torch.sqrt(var + 1e-5).unsqueeze(-1)  # [B, C, 1, 1]
        feat_max = out_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)  # 新增！[B, C, 1, 1]

        # 拼接3个统计量
        style_feat = torch.cat([feat_mean, feat_std, feat_max], dim=1)

        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 2.0 * torch.sigmoid(gamma)

        out = gamma * out + beta

        out = self.proj_out(out)

        return shortcut + out
