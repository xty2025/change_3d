# train_3d.py - 3D时序多模态模型训练
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import json
from models_3d import MultiModal3DModel
import clip
from simpledata import AugmentedMultiModalDataset

##matplot不兼容
SKIP_PLOTS = os.environ.get('SKIP_PLOTS', '0') == '1'
if not SKIP_PLOTS:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        SKIP_PLOTS = True

def train_3d_model():
    print("[STEP] 设置超参数...")
    # 支持快速测试模式：设置环境变量 QUICK_TRAIN=1 会将参数缩小用于快速 smoke-test
    QUICK = os.environ.get('QUICK_TRAIN', '0') == '1'
    time_steps = 5 if QUICK else 20
    batch_size = 1 if QUICK else 4
    num_epochs = 1 if QUICK else 50
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/mnt/d/TEMP/示例样本数据" 
    #data_dir="D:\TEMP\示例样本数据"
    dataset = AugmentedMultiModalDataset(
        root_dir=data_dir,
        time_steps=time_steps,
        target_size=(224, 224)
    )
    print(f"[INFO] 数据集样本数: {len(dataset)}")

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    model = MultiModal3DModel(time_steps=time_steps).to(device)
    criterion_smooth_l1 = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    checkpoint_path = "checkpoint_3d.pth"
    start_epoch = 0
    best_iou = 0.0
    print("[STEP] 检查是否有历史检查点...")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"[INFO] 加载检查点，从第 {start_epoch} 轮继续训练，最佳IoU: {best_iou:.4f}")
    print("[STEP] 开始训练循环...")
    stop_early = False
    for epoch in range(start_epoch, num_epochs):
        print(f"[EPOCH] 当前轮次: {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        total_iou = 0
        for batch_idx, batch in enumerate(dataloader):
            print(f"[BATCH] epoch={epoch+1}, batch={batch_idx+1}")
            rgb_sequence = batch['rgb_sequence'].to(device)
            depth_sequence = batch['depth_sequence'].to(device)
            text_tokens = batch['text_tokens']
            text_tokens['input_ids'] = text_tokens['input_ids'].to(device)
            bbox_sequence = batch['bbox_sequence'].to(device)
            # 可视化首个batch的RGB和深度序列（可选，SKIP_PLOTS=1 时跳过）
            if not SKIP_PLOTS and epoch == start_epoch and batch_idx == 0:
                try:
                    fig, axs = plt.subplots(2, time_steps, figsize=(2*time_steps, 4))
                    for t in range(time_steps):
                        axs[0, t].imshow(rgb_sequence[0, t].cpu().permute(1,2,0))
                        axs[0, t].set_title(f'RGB t={t}')
                        axs[0, t].axis('off')
                        axs[1, t].imshow(depth_sequence[0, t].cpu().permute(1,2,0)[:,:,0], cmap='gray')
                        axs[1, t].set_title(f'Depth t={t}')
                        axs[1, t].axis('off')
                    plt.suptitle('首个batch的时序图像')
                    plt.show()
                except Exception as e:
                    print(f"[WARN] 绘图失败，已跳过: {e}")
            print("[FORWARD] 正在前向推理...")
            pred_out = model(rgb_sequence, depth_sequence, text_tokens)
            # model 返回 (bbox_sequence, attention_weights)
            if isinstance(pred_out, tuple) or isinstance(pred_out, list):
                pred_bbox, attention = pred_out
            else:
                pred_bbox = pred_out
                attention = None
            print(f"[FORWARD] 预测bbox: {pred_bbox.shape}")
            print("[LOSS] 计算损失...")
            # 运行时检查：确保模型预测与目标在时间维度和形状上匹配，避免广播/维度错误
            if pred_bbox.shape != bbox_sequence.shape:
                raise ValueError(f"时间维度或形状不匹配: pred_bbox={pred_bbox.shape}, bbox_sequence={bbox_sequence.shape}. 请检查 dataset.time_steps 与模型 time_steps 是否一致。")
            smooth_l1_loss = criterion_smooth_l1(pred_bbox, bbox_sequence)
            # 使用图像尺寸归一化 IoU / GIoU
            img_h = rgb_sequence.size(-2)
            img_w = rgb_sequence.size(-1)
            giou_loss = calculate_giou_loss(pred_bbox, bbox_sequence, img_w=img_w, img_h=img_h)
            loss = smooth_l1_loss + giou_loss
            print(f"[LOSS] SmoothL1: {smooth_l1_loss.item():.4f}, GIoU: {giou_loss.item():.4f}, 总损失: {loss.item():.4f}")
            optimizer.zero_grad()
            print("[BACKWARD] 反向传播...")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print("[METRIC] 计算IoU...")
            iou = calculate_temporal_iou(pred_bbox, bbox_sequence, img_w=img_w, img_h=img_h)
            print(f"[METRIC] 当前batch IoU: {iou.item():.4f}")
            total_loss += loss.item()
            total_iou += iou.item()
            # 如果为快速测试模式，只跑第一个 batch 然后退出循环
            if QUICK:
                stop_early = True
                break
            if batch_idx % 10 == 0:
                print(f"[REPORT] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, SmoothL1: {smooth_l1_loss.item():.4f}, GIoU: {giou_loss.item():.4f}, IoU: {iou.item():.4f}")
        print("[STEP] 更新学习率...")
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        print(f"[EPOCH END] Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average IoU: {avg_iou:.4f}")
        print("[STEP] 保存检查点...")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_iou': best_iou,
        }, checkpoint_path)
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "best_3d_model.pth")
            print(f"[SAVE] 保存最佳模型，IoU: {best_iou:.4f}")
        if stop_early:
            print('[INFO] QUICK mode: stopping early after first batch')
            break

def _to_xyxy_norm(boxes, img_w, img_h):
    """将 boxes (b,4) 从 xywh（或已是xyxy）转换为归一化的 xyxy [0,1]"""
    boxes = boxes.clone()
    # 判断是否为绝对坐标（>1）或归一化（<=1）
    if boxes.max() > 1.0:
        # 假设为 xywh 绝对坐标
        x1 = boxes[:, 0] / img_w
        y1 = boxes[:, 1] / img_h
        x2 = (boxes[:, 0] + boxes[:, 2]) / img_w
        y2 = (boxes[:, 1] + boxes[:, 3]) / img_h
    else:
        # 假设为归一化 xywh
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
    x1 = x1.clamp(0.0, 1.0)
    y1 = y1.clamp(0.0, 1.0)
    x2 = x2.clamp(0.0, 1.0)
    y2 = y2.clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=1)


def calculate_temporal_iou(pred_bbox, target_bbox, img_w=224, img_h=224):
    """计算时序IoU，使用归一化的 xyxy 坐标"""
    batch_size, time_steps, _ = pred_bbox.shape
    total_iou = 0.0
    for t in range(time_steps):
        pred_t = pred_bbox[:, t, :]
        target_t = target_bbox[:, t, :]
        pred_xyxy = _to_xyxy_norm(pred_t, img_w, img_h)
        target_xyxy = _to_xyxy_norm(target_t, img_w, img_h)
        iou_t = calculate_iou(pred_xyxy, target_xyxy, inputs_are_xyxy=True)
        total_iou += iou_t.mean()
    return total_iou / time_steps

def calculate_iou(pred, target, inputs_are_xyxy=False):
    """计算 IoU。pred 和 target 为 (b,4)。
    如果 inputs_are_xyxy=True，则输入为 [x1,y1,x2,y2]（归一化或绝对均可），
    否则按[ x, y, w, h ] 处理并转换为xyxy。
    返回每个样本的 IoU 张量 (b,)
    """
    if not inputs_are_xyxy:
        pred_x1 = pred[:, 0]
        pred_y1 = pred[:, 1]
        pred_x2 = pred[:, 0] + pred[:, 2]
        pred_y2 = pred[:, 1] + pred[:, 3]
        target_x1 = target[:, 0]
        target_y1 = target[:, 1]
        target_x2 = target[:, 0] + target[:, 2]
        target_y2 = target[:, 1] + target[:, 3]
    else:
        pred_x1 = pred[:, 0]
        pred_y1 = pred[:, 1]
        pred_x2 = pred[:, 2]
        pred_y2 = pred[:, 3]
        target_x1 = target[:, 0]
        target_y1 = target[:, 1]
        target_x2 = target[:, 2]
        target_y2 = target[:, 3]

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = torch.clamp(pred_x2 - pred_x1, min=0) * torch.clamp(pred_y2 - pred_y1, min=0)
    target_area = torch.clamp(target_x2 - target_x1, min=0) * torch.clamp(target_y2 - target_y1, min=0)
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou

def calculate_giou_loss(pred_bbox, target_bbox, img_w=224, img_h=224):
    """计算GIoU损失（接受图像宽高用于将 xywh/绝对坐标 归一化为 xyxy [0,1] 并计算 GIoU）
    pred_bbox / target_bbox: (batch, time_steps, 4) 格式假定为 [x, y, w, h]（绝对或归一化）
    """
    batch_size, time_steps, _ = pred_bbox.shape

    total_giou_loss = 0
    for t in range(time_steps):
        pred_t = pred_bbox[:, t, :]
        target_t = target_bbox[:, t, :]

        # 先转为归一化 xyxy
        pred_xyxy = _to_xyxy_norm(pred_t, img_w, img_h)
        target_xyxy = _to_xyxy_norm(target_t, img_w, img_h)

        giou_loss_t = calculate_single_giou_loss_xyxy(pred_xyxy, target_xyxy)
        total_giou_loss += giou_loss_t.mean()

    return total_giou_loss / time_steps

def calculate_single_giou_loss_xyxy(pred_xyxy, target_xyxy):
    """计算单帧的 GIoU 损失，输入为归一化的 xyxy 格式 (b,4)。返回 (b,) 的 giou loss。"""
    pred_x1 = pred_xyxy[:, 0]
    pred_y1 = pred_xyxy[:, 1]
    pred_x2 = pred_xyxy[:, 2]
    pred_y2 = pred_xyxy[:, 3]

    target_x1 = target_xyxy[:, 0]
    target_y1 = target_xyxy[:, 1]
    target_x2 = target_xyxy[:, 2]
    target_y2 = target_xyxy[:, 3]

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = torch.clamp(pred_x2 - pred_x1, min=0) * torch.clamp(pred_y2 - pred_y1, min=0)
    target_area = torch.clamp(target_x2 - target_x1, min=0) * torch.clamp(target_y2 - target_y1, min=0)
    union_area = pred_area + target_area - inter_area

    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    enclose_area = torch.clamp(enclose_x2 - enclose_x1, min=0) * torch.clamp(enclose_y2 - enclose_y1, min=0)

    iou = inter_area / (union_area + 1e-6)
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    giou_loss = 1 - giou
    return giou_loss

def calculate_single_giou_loss(pred, target):
    """计算单个边界框的GIoU损失"""
    # 确保边界框格式为 [x, y, width, height]
    pred_x1 = pred[:, 0]
    pred_y1 = pred[:, 1]
    pred_x2 = pred[:, 0] + pred[:, 2]
    pred_y2 = pred[:, 1] + pred[:, 3]
    
    target_x1 = target[:, 0]
    target_y1 = target[:, 1]
    target_x2 = target[:, 0] + target[:, 2]
    target_y2 = target[:, 1] + target[:, 3]
    
    # 计算交集坐标
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    # 计算交集面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算并集面积
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # 计算最小包围框
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    
    # 计算最小包围框面积
    enclose_area = torch.clamp(enclose_x2 - enclose_x1, min=0) * torch.clamp(enclose_y2 - enclose_y1, min=0)
    
    # 计算IoU
    iou = inter_area / (union_area + 1e-6)
    
    # 计算GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    
    # GIoU损失 = 1 - GIoU
    giou_loss = 1 - giou
    
    return giou_loss

if __name__ == "__main__":
    train_3d_model()