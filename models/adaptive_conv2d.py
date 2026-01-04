import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveStripConv(nn.Module):
    """
    自适应带状卷积模块
    使用1×3和3×1卷积自适应组合替代3×3卷积
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super(AdaptiveStripConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 水平带状卷积 (1×3)
        self.horizontal_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3),
            stride=stride, bias=False
        )

        # 垂直带状卷积 (3×1)
        self.vertical_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1),
            stride=stride, bias=False
        )

        # 对角线增强卷积 (可选)
        self.diagonal_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride,  bias=False
        )

        # 自适应权重生成模块
        self.attention = ChannelSpatialAttention(out_channels * 3, out_channels)

        # 批量归一化
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 分别应用三种卷积
        horizontal_out = self.horizontal_conv(x)
        vertical_out = self.vertical_conv(x)
        diagonal_out = self.diagonal_conv(x)

        # 拼接特征图
        concat_features = torch.cat([horizontal_out, vertical_out, diagonal_out], dim=1)


        # 生成自适应权重
        weighted_features = self.attention(concat_features)

        # 最终输出
        out = self.bn(weighted_features)
        out = self.relu(out)

        return out


class ChannelSpatialAttention(nn.Module):
    """
    通道-空间注意力模块，用于自适应权重生成
    """

    def __init__(self, channels, reduction_ratio=16):
        super(ChannelSpatialAttention, self).__init__()
        self.channel_attention = ChannelAttention(channels*3, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 通道注意力
        x_channel = self.channel_attention(x)
        # 空间注意力
        x_final = self.spatial_attention(x_channel)
        return x_final


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(in_channels // reduction_ratio, 4)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class LightweightAdaptiveStripConv(nn.Module):
    """
    轻量级版本的自适应带状卷积
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(LightweightAdaptiveStripConv, self).__init__()

        # 水平和垂直卷积
        self.horizontal_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3),
            stride=stride, padding=(0, 1), groups=min(in_channels, out_channels)
        )

        self.vertical_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1),
            stride=stride, padding=(1, 0), groups=min(in_channels, out_channels)
        )

        # 简单的自适应权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        horizontal_out = self.horizontal_conv(x)
        vertical_out = self.vertical_conv(x)

        # 自适应融合
        out = self.alpha * horizontal_out + self.beta * vertical_out
        out = self.bn(out)
        out = self.activation(out)

        return out


# 测试代码
def test_adaptive_strip_conv():
    # 创建测试输入
    batch_size, channels, height, width = 4, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)

    # 测试完整版本
    print("Testing AdaptiveStripConv...")
    conv = AdaptiveStripConv(in_channels=64, out_channels=128, stride=1)
    out = conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # 测试轻量级版本
    print("\nTesting LightweightAdaptiveStripConv...")
    light_conv = LightweightAdaptiveStripConv(in_channels=64, out_channels=128, stride=1)
    light_out = light_conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {light_out.shape}")

    # 参数量对比
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    standard_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    print(f"\nParameter comparison:")
    print(f"Standard 3x3 Conv: {count_parameters(standard_conv):,} parameters")
    print(f"AdaptiveStripConv: {count_parameters(conv):,} parameters")
    print(f"Lightweight Version: {count_parameters(light_conv):,} parameters")


if __name__ == "__main__":
    test_adaptive_strip_conv()