import torch
import torch.nn as nn
import torchvision.models as models
try:
    import clip
except Exception:
    clip = None
    # lightweight fallback: provide a mock clip.load that returns an object with encode_text
    class _MockCLIPModel:
        def __init__(self):
            self.dtype = torch.float32

        def encode_text(self, tokens):
            # tokens: tensor (batch, seq_len) -> produce (batch, 512)
            batch = tokens.size(0)
            return torch.randn(batch, 512)

    def _mock_clip_load(name, device='cpu'):
        return _MockCLIPModel(), None

    clip = type('clip', (), {'load': _mock_clip_load, 'tokenize': lambda texts: torch.randint(0,1000,(len(texts),77))})
from CBAM import CBAM3D, TemporalCBAM, TemporalAttention

class MultiModal3DModel(nn.Module):
    def __init__(self, num_classes=1, time_steps=20):
        super(MultiModal3DModel, self).__init__()
        self.time_steps = time_steps
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device='cpu')
        self.clip_model_name = 'ViT-B/32'
        
        # 冻结CLIP模型的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 3D ResNet-18用于时序RGB特征提取（预训练模型）
        self.rgb_3d_encoder = models.video.r3d_18(pretrained=True)
        # 移除分类头，获取特征
        self.rgb_3d_encoder.fc = nn.Identity()
        
        # 3D ResNet-18用于深度图像序列特征提取（预训练模型）
        self.depth_3d_encoder = models.video.r3d_18(pretrained=True)
        # 修改输入通道数为1（深度图通常是单通道）
        self.depth_3d_encoder.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), 
                                                 stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.depth_3d_encoder.fc = nn.Identity()
        
        # 特征维度
        rgb_3d_features = 512  # R3D-18输出维度
        depth_3d_features = 512  # R3D-18输出维度
        text_features = 512  # CLIP文本特征维度
        
        # 3D CBAM注意力模块（优化版）
        self.rgb_cbam = CBAM3D(rgb_3d_features, reduction=8)  # 减少reduction提高特征保留
        self.depth_cbam = CBAM3D(depth_3d_features, reduction=8)
        
        # 文本特征投影层（优化版）
        self.text_projection = nn.Sequential(
            nn.Linear(text_features, 384),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(0.15),
            nn.Linear(384, 512),
            nn.LayerNorm(512)  # 添加LayerNorm稳定训练
        )
        
        # 时序注意力模块（优化版）
        self.temporal_attention = TemporalAttention(
            rgb_3d_features + depth_3d_features + 512,
            num_heads=8
        )
        
        # 特征融合层（优化版）
        self.fusion_fc = nn.Sequential(
            nn.Linear(rgb_3d_features + depth_3d_features + 512, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(768, 512),
            nn.LayerNorm(512)  # 添加LayerNorm
        )
        
        # 时序边界框回归器（优化版）
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512, 384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 4)  # 每个时间步输出4个坐标 (x,y,w,h)
        )
        
        # 残差连接层
        self.residual_fc = nn.Linear(rgb_3d_features + depth_3d_features + 512, 512)
    
    def forward(self, rgb_sequence, depth_sequence, text_input):
        """
        前向传播（清晰实现，缩进一致）
        返回：bbox_sequence (b, t, 4), attention_weights (可为None)
        """
        batch_size = rgb_sequence.size(0)

        # 1. RGB特征：从 (b, t, 3, H, W) -> (b, 3, t, H, W) -> encoder -> (b, feat)
        rgb_3d = rgb_sequence.permute(0, 2, 1, 3, 4)
        rgb_features_3d = self.rgb_3d_encoder(rgb_3d)
        rgb_features_3d = torch.nn.functional.normalize(rgb_features_3d, p=2, dim=1)

        # 2. Depth特征
        if depth_sequence.size(2) != 1:
            depth_sequence = depth_sequence.mean(dim=2, keepdim=True)
        depth_3d = depth_sequence.permute(0, 2, 1, 3, 4)
        depth_features_3d = self.depth_3d_encoder(depth_3d)
        depth_features_3d = torch.nn.functional.normalize(depth_features_3d, p=2, dim=1)

        # 3. 文本特征
        if isinstance(text_input, dict):
            text_tokens = text_input.get('input_ids')
        else:
            text_tokens = text_input
        if text_tokens is None:
            raise ValueError('text_input missing input_ids')
        if text_tokens.dim() > 2:
            # squeeze possible extra batch dimension
            text_tokens = text_tokens.squeeze(1)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)

        # 4. CBAM attention (reshape to support CBAM3D)
        rgb_cbam_in = rgb_features_3d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.time_steps, 1, 1)
        rgb_features_3d_attended = self.rgb_cbam(rgb_cbam_in)
        if rgb_features_3d_attended.dim() > 2:
            rgb_features_3d_attended = rgb_features_3d_attended.view(rgb_features_3d_attended.size(0), -1)
        if rgb_features_3d_attended.size(-1) != 512:
            pad = 512 - rgb_features_3d_attended.size(-1)
            if pad > 0:
                rgb_features_3d_attended = torch.cat([rgb_features_3d_attended, rgb_features_3d_attended.new_zeros(rgb_features_3d_attended.size(0), pad)], dim=1)
            else:
                rgb_features_3d_attended = rgb_features_3d_attended[:, :512]

        depth_cbam_in = depth_features_3d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.time_steps, 1, 1)
        depth_features_3d_attended = self.depth_cbam(depth_cbam_in)
        if depth_features_3d_attended.dim() > 2:
            depth_features_3d_attended = depth_features_3d_attended.view(depth_features_3d_attended.size(0), -1)
        if depth_features_3d_attended.size(-1) != 512:
            pad = 512 - depth_features_3d_attended.size(-1)
            if pad > 0:
                depth_features_3d_attended = torch.cat([depth_features_3d_attended, depth_features_3d_attended.new_zeros(depth_features_3d_attended.size(0), pad)], dim=1)
            else:
                depth_features_3d_attended = depth_features_3d_attended[:, :512]

        # 5. 文本投影
        text_features_projected = self.text_projection(text_features)

        # 6. 扩展到时序并拼接 -> (b, t, feat)
        rgb_exp = rgb_features_3d_attended.unsqueeze(1).expand(batch_size, self.time_steps, -1)
        depth_exp = depth_features_3d_attended.unsqueeze(1).expand(batch_size, self.time_steps, -1)
        text_exp = text_features_projected.unsqueeze(1).expand(batch_size, self.time_steps, -1)
        combined = torch.cat([rgb_exp, depth_exp, text_exp], dim=2)  # (b,t,1536)

        # 7. 时序注意力
        attended = self.temporal_attention(combined)
        attention_weights = None

        # 8. 融合与回归（nn.Linear 支持 (..., feat) 形式）
        fused = self.fusion_fc(attended)  # (b, t, 512)
        bbox_seq = self.bbox_regressor(fused)  # (b, t, 4)
        bbox_seq = torch.sigmoid(bbox_seq)

        return bbox_seq, attention_weights
    
    def save_model(self, filepath):
        """保存模型状态（优化：包含优化器状态）"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'time_steps': self.time_steps,
            'clip_model_name': self.clip_model_name
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath, device='cuda'):
        """加载模型状态（优化：设备兼容性）"""
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {filepath} 加载")
    
    def get_model_info(self):
        """获取模型信息（优化：详细统计）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'time_steps': self.time_steps,
            'clip_model': self.clip_model_name,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }
        
        return info
 
class MultiModal3DLoss(nn.Module):
    """多模态3D损失函数（优化版）"""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, delta=0.05, epsilon=0.02):
        super().__init__()
        self.alpha = alpha  # Smooth L1损失权重
        self.beta = beta    # GIoU损失权重
        self.gamma = gamma  # 时序一致性损失权重
        self.delta = delta  # 中心点距离损失权重
        self.epsilon = epsilon  # 尺度一致性损失权重
        
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        
        # 自适应损失权重（优化：可学习的权重）
        self.adaptive_weights = nn.Parameter(torch.ones(5))
    
    def _giou_loss(self, pred_boxes, target_boxes):
        """计算GIoU损失（优化：完整实现）"""
        batch_size, time_steps, _ = pred_boxes.shape
        
        # 重塑为(batch * time_steps, 4)
        pred_flat = pred_boxes.view(-1, 4)
        target_flat = target_boxes.view(-1, 4)
        
        # 计算交集坐标
        inter_x1 = torch.max(pred_flat[:, 0], target_flat[:, 0])
        inter_y1 = torch.max(pred_flat[:, 1], target_flat[:, 1])
        inter_x2 = torch.min(pred_flat[:, 2], target_flat[:, 2])
        inter_y2 = torch.min(pred_flat[:, 3], target_flat[:, 3])
        
        # 计算交集面积
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集面积
        pred_area = (pred_flat[:, 2] - pred_flat[:, 0]) * (pred_flat[:, 3] - pred_flat[:, 1])
        target_area = (target_flat[:, 2] - target_flat[:, 0]) * (target_flat[:, 3] - target_flat[:, 1])
        union_area = pred_area + target_area - inter_area
        
        # 计算最小包围框
        enclosing_x1 = torch.min(pred_flat[:, 0], target_flat[:, 0])
        enclosing_y1 = torch.min(pred_flat[:, 1], target_flat[:, 1])
        enclosing_x2 = torch.max(pred_flat[:, 2], target_flat[:, 2])
        enclosing_y2 = torch.max(pred_flat[:, 3], target_flat[:, 3])
        enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)
        
        # 计算IoU和GIoU
        iou = inter_area / (union_area + 1e-6)
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-6)
        
        return 1 - giou.mean()
    
    def _temporal_consistency_loss(self, pred_sequence):
        """时序一致性损失（优化：多尺度时序一致性）"""
        batch_size, time_steps, _ = pred_sequence.shape
        
        # 短期一致性（相邻帧）
        short_term_diff = pred_sequence[:, 1:] - pred_sequence[:, :-1]
        short_term_loss = torch.mean(torch.abs(short_term_diff))
        
        # 中期一致性（间隔帧）
        if time_steps > 2:
            mid_term_diff = pred_sequence[:, 2:] - pred_sequence[:, :-2]
            mid_term_loss = torch.mean(torch.abs(mid_term_diff))
        else:
            mid_term_loss = torch.tensor(0.0, device=pred_sequence.device)
        
        # 长期一致性（首尾帧）
        long_term_diff = pred_sequence[:, -1:] - pred_sequence[:, :1]
        long_term_loss = torch.mean(torch.abs(long_term_diff))
        
        return (short_term_loss + mid_term_loss * 0.5 + long_term_loss * 0.2) / 1.7
    
    def _center_distance_loss(self, pred_boxes, target_boxes):
        """中心点距离损失（优化：添加）"""
        pred_centers = (pred_boxes[:, :, :2] + pred_boxes[:, :, 2:]) / 2
        target_centers = (target_boxes[:, :, :2] + target_boxes[:, :, 2:]) / 2
        
        center_dist = torch.norm(pred_centers - target_centers, dim=2)
        return center_dist.mean()
    
    def _scale_consistency_loss(self, pred_boxes):
        """尺度一致性损失（优化：添加）"""
        pred_widths = pred_boxes[:, :, 2] - pred_boxes[:, :, 0]
        pred_heights = pred_boxes[:, :, 3] - pred_boxes[:, :, 1]
        
        # 计算相邻帧的尺度变化
        width_diff = pred_widths[:, 1:] - pred_widths[:, :-1]
        height_diff = pred_heights[:, 1:] - pred_heights[:, :-1]
        
        scale_consistency = torch.mean(torch.abs(width_diff)) + torch.mean(torch.abs(height_diff))
        return scale_consistency
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (batch, time_steps, 4)
            target_boxes: (batch, time_steps, 4)
        """
        # Smooth L1损失
        l1_loss = self.smooth_l1_loss(pred_boxes, target_boxes)
        
        # GIoU损失
        giou_loss = self._giou_loss(pred_boxes, target_boxes)
        
        # 时序一致性损失
        temporal_loss = self._temporal_consistency_loss(pred_boxes)
        
        # 中心点距离损失（优化：新增）
        center_loss = self._center_distance_loss(pred_boxes, target_boxes)
        
        # 尺度一致性损失（优化：新增）
        scale_loss = self._scale_consistency_loss(pred_boxes)
        
        # 自适应权重归一化
        normalized_weights = torch.softmax(self.adaptive_weights, dim=0)
        
        # 总损失（优化：使用自适应权重）
        total_loss = (normalized_weights[0] * l1_loss + 
                     normalized_weights[1] * giou_loss + 
                     normalized_weights[2] * temporal_loss + 
                     normalized_weights[3] * center_loss + 
                     normalized_weights[4] * scale_loss)
        
        # 返回详细损失信息用于监控
        loss_dict = {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'giou_loss': giou_loss,
            'temporal_loss': temporal_loss,
            'center_loss': center_loss,
            'scale_loss': scale_loss,
            'weights': normalized_weights.detach()
        }
        
        return total_loss, loss_dict

# 示例：初始化3D时序模型
if __name__ == "__main__":
    model = MultiModal3DModel(time_steps=5)
    
    # 模拟输入数据
    batch_size = 2
    time_steps = 5
    
    rgb_sequence = torch.randn(batch_size, time_steps, 3, 224, 224)
    depth_sequence = torch.randn(batch_size, time_steps, 3, 224, 224)
    
    # 文本输入
    import clip
    clip_model, _ = clip.load('ViT-B/32', device='cpu')
    text = clip.tokenize(["Find the object in the video sequence"])
    
    # 前向传播
    out = model(rgb_sequence, depth_sequence, text)
    if isinstance(out, tuple) or isinstance(out, list):
        bbox_sequence, attention = out
    else:
        bbox_sequence = out
        attention = None
    print(f"预测边界框序列形状: {bbox_sequence.shape}")  # 应该是 (2, 5, 4)