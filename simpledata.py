import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import clip
import json

from transformers import BertTokenizer
from parameters import data_path_

# 时序多模态数据集类（支持帧序列图片）
class MultiModalDataset(Dataset):
    def __init__(self, root_dir, time_steps=5, transform=None, target_size=(224, 224), save_preds=False, pred_out_dir=None):

        self.root_dir = root_dir
        self.time_steps = time_steps
        self.transform = transform
        self.target_size = target_size
        # Whether to save predicted rects (xywh) and IoU to files
        self.save_preds = save_preds
        if pred_out_dir is None:
            self.pred_out_dir = os.path.join(root_dir, 'pred_rects')
        else:
            self.pred_out_dir = pred_out_dir
        os.makedirs(self.pred_out_dir, exist_ok=True)
        
        # 加载CLIP tokenizer
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device='cpu')
        
        # 收集所有类别文件夹
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.samples = []
        # 遍历每个类别文件夹
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            rgb_path = os.path.join(class_path, 'color')
            depth_path = os.path.join(class_path, 'depth')
            description_path = os.path.join(class_path, 'nlp.txt')
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(description_path)):
                print(f"Warning: Missing files in {class_name}, skipping...")
                continue
            rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            with open(description_path, 'r', encoding='utf-8') as f:
                description = f.read().strip()
            # 每个样本为连续time_steps帧
            min_len = min(len(rgb_files), len(depth_files))
            # 使用传入的 time_steps 参数（不要在此硬编码为 10）
            for start_idx in range(0, min_len - self.time_steps + 1):
                sample = {
                    'class_name': class_name,
                    'start_idx': start_idx,
                    'text_description': description,
                    'rgb_files': rgb_files[start_idx:start_idx + self.time_steps],
                    'depth_files': depth_files[start_idx:start_idx + self.time_steps],
                    'rgb_path': rgb_path,
                    'depth_path': depth_path
                }
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)

    # --- 辅助方法: 从深度张量生成简单 bbox (xywh) ---
    def _bbox_from_depth_tensor(self, d_tensor, threshold=0.02):
        """
        d_tensor: torch tensor (1, H, W) or (H, W), values 已归一化到 [0,1]
        返回 [x, y, w, h]，像素坐标（基于 self.target_size）
        如果未检测到目标，返回 [0,0,0,0]
        """
        try:
            d = d_tensor.squeeze().cpu().numpy()
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
        if d.ndim != 2:
            return [0.0, 0.0, 0.0, 0.0]
        h, w = d.shape
        # 使用中位数作为背景估计，检测比背景更小（更近）的像素
        bg = float(np.median(d))
        mask = d < (bg - threshold)
        if mask.sum() == 0:
            # 退化处理: 没有检测到，返回全零
            return [0.0, 0.0, 0.0, 0.0]
        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w_box = float(x_max - x_min + 1)
        h_box = float(y_max - y_min + 1)
        return [float(x_min), float(y_min), w_box, h_box]

    def _load_groundtruth_rects(self, gt_path):
        """读取 groundtruth_rect.txt，返回 list of [x,y,w,h]（floats）。支持逗号或空格分隔。"""
        rects = []
        if not os.path.exists(gt_path):
            return rects
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 有些文件以逗号分隔
                line2 = line.replace(',', ' ')
                parts = line2.split()
                if len(parts) < 4:
                    continue
                try:
                    x, y, w, h = map(float, parts[:4])
                    rects.append([x, y, w, h])
                except Exception:
                    continue
        return rects

    def _iou_xywh(self, boxA, boxB):
        """计算两个 xywh 格式框的 IoU，返回浮点数（保留 4 位）或 0.0。"""
        try:
            xA, yA, wA, hA = map(float, boxA)
            xB, yB, wB, hB = map(float, boxB)
        except Exception:
            return 0.0
        if wA <= 0 or hA <= 0 or wB <= 0 or hB <= 0:
            return 0.0
        xa1, ya1, xa2, ya2 = xA, yA, xA + wA, yA + hA
        xb1, yb1, xb2, yb2 = xB, yB, xB + wB, yB + hB
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        areaA = wA * hA
        areaB = wB * hB
        union = areaA + areaB - inter_area
        if union <= 0:
            return 0.0
        iou = inter_area / union
        return float(round(iou, 4))
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载连续time_steps帧的RGB和深度图
        rgb_sequence = []
        for rgb_file in sample['rgb_files']:
            rgb_path = os.path.join(sample['rgb_path'], rgb_file)
            rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_image = rgb_image.resize(self.target_size)
            if self.transform:
                rgb_image = self.transform(rgb_image)
            else:
                rgb_image = transforms.ToTensor()(rgb_image)
                rgb_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb_image)
            rgb_sequence.append(rgb_image)
        depth_sequence = []
        for depth_file in sample['depth_files']:
            depth_path = os.path.join(sample['depth_path'], depth_file)
            depth_image = Image.open(depth_path).convert('L')
            depth_image = depth_image.resize(self.target_size)
            if self.transform:
                depth_image = self.transform(depth_image)
            else:
                depth_image = transforms.ToTensor()(depth_image)
                depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min() + 1e-8)
            depth_sequence.append(depth_image)
        # 文本编码
        text_tokens = clip.tokenize([sample['text_description']], truncate=True)
        # 基于深度图检测简单边界框（xywh，像素坐标）
        # depth_sequence 是一个 list of tensors (time_steps, 1, H, W) 已经在上面构造
        bbox_sequence = torch.zeros((self.time_steps, 4), dtype=torch.float32)
        # 简易检测阈值，深度归一化后使用
        depth_thresh = 0.02
        # 将 depth_sequence 转为 numpy 来计算 bbox
        for t in range(self.time_steps):
            d = depth_sequence[t]  # (1, H, W)
            bbox = self._bbox_from_depth_tensor(d, threshold=depth_thresh)
            bbox_sequence[t] = torch.tensor(bbox, dtype=torch.float32)

        #保存预测 rects，与 groundtruth 计算 IoU 并写入文件
        if self.save_preds:
            try:
                gt_path = os.path.join(self.root_dir, sample['class_name'], 'groundtruth_rect.txt')
                gt_rects = self._load_groundtruth_rects(gt_path)
            except Exception:
                gt_rects = None
            out_fname = f"{sample['class_name']}_{sample['start_idx']}_pred.txt"
            out_path = os.path.join(self.pred_out_dir, out_fname)
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write('#frame_idx x y w h iou\n')
                for t in range(self.time_steps):
                    pred = bbox_sequence[t].tolist()
                    frame_idx = sample['start_idx'] + t
                    iou = ''
                    if gt_rects is not None:
                        # gt_rects assumed 0-based lines corresponding to frames
                        if 0 <= frame_idx < len(gt_rects):
                            gt = gt_rects[frame_idx]
                            iou = self._iou_xywh(pred, gt)
                        else:
                            iou = ''
                    out_f.write(f"{frame_idx} {pred[0]:.2f} {pred[1]:.2f} {pred[2]:.2f} {pred[3]:.2f} {iou}\n")
        # 转换为张量
        rgb_sequence = torch.stack(rgb_sequence)  # (time_steps, 3, H, W)
        depth_sequence = torch.stack(depth_sequence)  # (time_steps, 1, H, W)
        return {
            'rgb_sequence': rgb_sequence,
            'depth_sequence': depth_sequence,
            'text_tokens': {'input_ids': text_tokens},
            'bbox_sequence': bbox_sequence
        }

    # --- 可视化方法 ---
    def draw_bbox_on_image(self, pil_image, bbox, outline=(255, 0, 0), width=2):
        """在 PIL Image 上绘制 xywh bbox（像素坐标），返回新的 PIL Image。若 bbox 为全 0 则不绘制。"""
        if pil_image is None:
            return None
        try:
            x, y, w, h = map(float, bbox)
        except Exception:
            return pil_image
        if w <= 0 or h <= 0:
            return pil_image
        img = pil_image.convert('RGB')
        draw = ImageDraw.Draw(img)
        rect = [x, y, x + w, y + h]
        draw.rectangle(rect, outline=outline, width=width)
        return img

    def visualize_sample(self, idx, out_dir=None, draw_gt=True, pred_color=(255,0,0), gt_color=(0,255,0), width=2):
        """根据索引可视化样本的每一帧：在 RGB 图上绘制预测 bbox（与可选 GT），并保存到 out_dir。
        out_dir 默认为 self.pred_out_dir/vis。
        返回生成的文件路径列表。
        """
        if out_dir is None:
            out_dir = os.path.join(self.pred_out_dir, 'vis')
        os.makedirs(out_dir, exist_ok=True)
        sample = self.samples[idx]
        # 使用 __getitem__ 获取 bbox_sequence（会执行同样的检测流程）
        sample_data = self[idx]
        bbox_seq = sample_data['bbox_sequence']  # tensor (time_steps,4)
        # 尝试加载 GT
        gt_path = os.path.join(self.root_dir, sample['class_name'], 'groundtruth_rect.txt')
        gt_rects = self._load_groundtruth_rects(gt_path) if draw_gt else None
        saved = []
        for t in range(self.time_steps):
            rgb_fname = sample['rgb_files'][t]
            rgb_path = os.path.join(sample['rgb_path'], rgb_fname)
            try:
                pil = Image.open(rgb_path).convert('RGB')
                pil = pil.resize(self.target_size)
            except Exception:
                # 如果无法打开原始图像，则尝试用堆叠后的 tensor 转换
                try:
                    tensor_img = sample_data['rgb_sequence'][t]  # (3,H,W)
                    # to PIL
                    arr = (tensor_img.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                    pil = Image.fromarray(arr)
                except Exception:
                    continue
            pred = bbox_seq[t].tolist()
            vis = pil
            if pred is not None and not (pred[2] == 0 and pred[3] == 0 and pred[0] == 0 and pred[1] == 0):
                vis = self.draw_bbox_on_image(vis, pred, outline=pred_color, width=width)
            # 绘制 GT（如果存在且对应帧索引存在）
            if gt_rects is not None:
                frame_idx = sample['start_idx'] + t
                if 0 <= frame_idx < len(gt_rects):
                    gt = gt_rects[frame_idx]
                    if not (gt[2] == 0 and gt[3] == 0):
                        vis = self.draw_bbox_on_image(vis, gt, outline=gt_color, width=width)
            out_name = f"{sample['class_name']}_{sample['start_idx']}_frame{sample['start_idx']+t}_vis.jpg"
            out_path = os.path.join(out_dir, out_name)
            try:
                vis.save(out_path)
                saved.append(out_path)
            except Exception:
                continue
        return saved

    def save_pred_rects_for_sample(self, idx, pred_rects, write_iou=True, visualize=False, vis_out_dir=None):
        """保存外部检测得到的 pred_rects（list of [x,y,w,h] 或数组）到 txt，并可选可视化。
        参数:
          idx: 数据集中样本索引
          pred_rects: iterable，长度可为 self.time_steps 或 1（会广播），每项为 [x,y,w,h]
          write_iou: 是否计算并写入 IoU（需要对应的 groundtruth_rect.txt）
          visualize: 是否把预测框绘制到 RGB 图并保存
          vis_out_dir: 可视化输出目录，默认 self.pred_out_dir/vis
        返回: 保存的 txt 路径和（可选）可视化文件列表
        """
        sample = self.samples[idx]
        # 标准化 pred_rects 到 list 长度 time_steps
        preds = list(pred_rects)
        if len(preds) == 0:
            raise ValueError('pred_rects 为空')
        if len(preds) == 1 and self.time_steps > 1:
            preds = preds * self.time_steps
        if len(preds) != self.time_steps:
            raise ValueError(f'pred_rects 长度应为 {self.time_steps} 或 1, 当前为 {len(preds)}')

        # 加载 GT
        gt_path = os.path.join(self.root_dir, sample['class_name'], 'groundtruth_rect.txt')
        gt_rects = self._load_groundtruth_rects(gt_path) if write_iou else None

        out_fname = f"{sample['class_name']}_{sample['start_idx']}_pred_external.txt"
        out_path = os.path.join(self.pred_out_dir, out_fname)
        os.makedirs(self.pred_out_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write('#frame_idx x y w h iou\n')
            for t in range(self.time_steps):
                pred = preds[t]
                try:
                    x, y, w, h = map(float, pred)
                except Exception:
                    x = y = w = h = 0.0
                frame_idx = sample['start_idx'] + t
                iou_val = ''
                if write_iou and gt_rects is not None:
                    if 0 <= frame_idx < len(gt_rects):
                        gt = gt_rects[frame_idx]
                        iou_val = self._iou_xywh([x, y, w, h], gt)
                    else:
                        iou_val = ''
                out_f.write(f"{frame_idx} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {iou_val}\n")

        vis_saved = []
        if visualize:
            if vis_out_dir is None:
                vis_out_dir = os.path.join(self.pred_out_dir, 'vis_external')
            os.makedirs(vis_out_dir, exist_ok=True)
            # 为每帧绘制并保存
            for t in range(self.time_steps):
                rgb_fname = sample['rgb_files'][t]
                rgb_path = os.path.join(sample['rgb_path'], rgb_fname)
                try:
                    pil = Image.open(rgb_path).convert('RGB')
                    pil = pil.resize(self.target_size)
                except Exception:
                    # 如果无法打开原始图像，尝试用 dataset[idx] 中的 tensor 转换
                    try:
                        sample_data = self[idx]
                        tensor_img = sample_data['rgb_sequence'][t]
                        arr = (tensor_img.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                        pil = Image.fromarray(arr)
                    except Exception:
                        continue
                pred = preds[t]
                try:
                    x, y, w, h = map(float, pred)
                except Exception:
                    x = y = w = h = 0.0
                if not (w == 0 and h == 0):
                    vis = self.draw_bbox_on_image(pil, [x,y,w,h], outline=(255,0,0), width=2)
                else:
                    vis = pil
                # 绘制 GT（如果有）
                if gt_rects is not None:
                    frame_idx = sample['start_idx'] + t
                    if 0 <= frame_idx < len(gt_rects):
                        gt = gt_rects[frame_idx]
                        if not (gt[2] == 0 and gt[3] == 0):
                            vis = self.draw_bbox_on_image(vis, gt, outline=(0,255,0), width=2)
                out_name = f"{sample['class_name']}_{sample['start_idx']}_frame{frame_idx}_extvis.jpg"
                out_path_vis = os.path.join(vis_out_dir, out_name)
                try:
                    vis.save(out_path_vis)
                    vis_saved.append(out_path_vis)
                except Exception:
                    continue

        return out_path, vis_saved

# 增强版时序多模态数据集类（支持数据增强）
class AugmentedMultiModalDataset(MultiModalDataset):
    def __init__(self, root_dir, time_steps=5, target_size=(224, 224), save_preds=False, pred_out_dir=None):
        # 数据增强变换（分别处理RGB和深度图）
        rgb_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        depth_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])

        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        # 调用父类构造器，传入 save_preds/pred_out_dir
        super().__init__(root_dir, time_steps, None, target_size, save_preds=save_preds, pred_out_dir=pred_out_dir)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载RGB图像序列（应用RGB增强）
        rgb_sequence = []
        for rgb_file in sample['rgb_files']:
            rgb_path = os.path.join(sample['rgb_path'], rgb_file)
            rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_image = rgb_image.resize(self.target_size)
            rgb_image = self.rgb_transform(rgb_image)
            rgb_sequence.append(rgb_image)
        
        # 加载深度图像序列（应用深度图增强）
        depth_sequence = []
        for depth_file in sample['depth_files']:
            depth_path = os.path.join(sample['depth_path'], depth_file)
            depth_image = Image.open(depth_path).convert('L')  # 转换为灰度图
            depth_image = depth_image.resize(self.target_size)
            depth_image = self.depth_transform(depth_image)
            # 深度图归一化
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min() + 1e-8)
            depth_sequence.append(depth_image)
        
        # 文本编码
        text_tokens = clip.tokenize([sample['text_description']], truncate=True)
        
        # 转换为张量
        rgb_sequence = torch.stack(rgb_sequence)  # (time_steps, 3, H, W)
        depth_sequence = torch.stack(depth_sequence)  # (time_steps, 1, H, W)
        
        # 基于深度图检测简单边界框（xywh，像素坐标）
        bbox_sequence = torch.zeros((self.time_steps, 4), dtype=torch.float32)
        depth_thresh = 0.02
        for t in range(self.time_steps):
            d = depth_sequence[t]
            bbox = self._bbox_from_depth_tensor(d, threshold=depth_thresh)
            bbox_sequence[t] = torch.tensor(bbox, dtype=torch.float32)

        if self.save_preds:
            try:
                gt_path = os.path.join(self.root_dir, sample['class_name'], 'groundtruth_rect.txt')
                gt_rects = self._load_groundtruth_rects(gt_path)
            except Exception:
                gt_rects = None
            out_fname = f"{sample['class_name']}_{sample['start_idx']}_pred.txt"
            out_path = os.path.join(self.pred_out_dir, out_fname)
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write('#frame_idx x y w h iou\n')
                for t in range(self.time_steps):
                    pred = bbox_sequence[t].tolist()
                    frame_idx = sample['start_idx'] + t
                    iou = ''
                    if gt_rects is not None:
                        if 0 <= frame_idx < len(gt_rects):
                            gt = gt_rects[frame_idx]
                            iou = self._iou_xywh(pred, gt)
                        else:
                            iou = ''
                    out_f.write(f"{frame_idx} {pred[0]:.2f} {pred[1]:.2f} {pred[2]:.2f} {pred[3]:.2f} {iou}\n")
        
        return {
            'rgb_sequence': rgb_sequence,
            'depth_sequence': depth_sequence,
            'text_tokens': {'input_ids': text_tokens},
            'bbox_sequence': bbox_sequence
        }


# 测试代码（可选）
if __name__ == "__main__":
    dataset = MultiModalDataset(root_dir="./data", time_steps=5, target_size=(224, 224))
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"RGB序列形状: {sample['rgb_sequence'].shape}")
        print(f"深度序列形状: {sample['depth_sequence'].shape}")
        print(f"文本tokens形状: {sample['text_tokens']['input_ids'].shape}")
        print(f"边界框序列形状: {sample['bbox_sequence'].shape}")
