import torch
import numpy as np
# 引入 medpy 的 hd95 计算函数
from medpy.metric.binary import hd95

def dice_coefficient(preds, masks, smooth=1e-5):
    preds = preds.view(-1)
    masks = masks.view(-1)
    intersection = (preds * masks).sum()
    return (2. * intersection + smooth) / (preds.sum() + masks.sum() + smooth)

def iou_score(preds, masks, smooth=1e-5):
    preds = preds.view(-1)
    masks = masks.view(-1)
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def accuracy_score(preds, masks):
    preds = preds.view(-1)
    masks = masks.view(-1)
    correct = (preds == masks).sum()
    total = preds.numel()
    return correct.float() / total

def precision_score(preds, masks, smooth=1e-5):
    preds = preds.view(-1)
    masks = masks.view(-1)
    intersection = (preds * masks).sum()
    return (intersection + smooth) / (preds.sum() + smooth)

def recall_score(preds, masks, smooth=1e-5):
    preds = preds.view(-1)
    masks = masks.view(-1)
    intersection = (preds * masks).sum()
    return (intersection + smooth) / (masks.sum() + smooth)

def specificity_score(preds, masks, smooth=1e-5):
    preds = preds.view(-1)
    masks = masks.view(-1)
    preds_inv = 1 - preds
    masks_inv = 1 - masks
    intersection = (preds_inv * masks_inv).sum()
    return (intersection + smooth) / (masks_inv.sum() + smooth)

def compute_hd95(pred, gt, spacing=1.0):
    """
    使用 MedPy 计算 HD95 (标准版)
    pred, gt: numpy array, 0 or 1
    """
    # 1. 判空处理：MedPy 如果遇到全黑图像会报错，必须处理
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        # 如果预测或标签全黑，返回图像对角线长度作为最大惩罚
        return np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)
    
    try:
        # 2. 调用官方库计算
        # 注意：medpy 不需要我们手动提取边缘，它内部会自动做
        return hd95(pred, gt, voxelspacing=spacing)
    except Exception as e:
        print(f"HD95 Error: {e}")
        # 万一出错了（比如形状极度畸形），返回惩罚值
        return np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)