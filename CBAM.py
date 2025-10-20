import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D通道注意力
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1, 1)
        return x * out.expand_as(x)

# 3D空间注意力
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv1(x_cat))
        return x * attention_map

# 3D CBAM注意力模块
class CBAM3D(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_planes, reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# 时序注意力模块
class TemporalAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # 输入形状: (batch, time_steps, channels)
        attended, _ = self.multihead_attention(x, x, x)
        x = self.layer_norm(x + attended)
        return x

# 时序CBAM模块（结合时序和空间注意力）
class TemporalCBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(TemporalCBAM, self).__init__()
        self.temporal_attention = TemporalAttention(in_channels)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入形状: (batch, time_steps, channels)
        b, t, c = x.size()
        
        # 时序注意力
        x = self.temporal_attention(x)
        
        # 通道注意力
        channel_weights = self.channel_attention(x.mean(dim=1))  # (b, c)
        channel_weights = channel_weights.unsqueeze(1)  # (b, 1, c)
        
        return x * channel_weights

# 保持原有的2D CBAM模块兼容性
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv1(x_cat))
        return x * attention_map

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


