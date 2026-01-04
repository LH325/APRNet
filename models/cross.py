import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelSequenceEnhancement(nn.Module):
    """
    通道依赖增强 + 序列全局&局部依赖增强模块
    输入: [B, C, L]
    输出: [B, C, L]
    """

    def __init__(self, channels, sequence_length, reduction=16, kernel_size=7, num_heads=None):
        super(ChannelSequenceEnhancement, self).__init__()
        self.channels = channels
        self.sequence_length = sequence_length

        # 自动调整num_heads，确保能被通道数整除
        if num_heads is None:
            # 找到通道数的最大因数（不超过16，避免头数过多）
            max_heads = min(16, channels)
            for i in range(max_heads, 0, -1):
                if channels % i == 0:
                    num_heads = i
                    break
            else:
                num_heads = 1  # 如果没有找到合适的因数，使用单头注意力

        print(f"通道数: {channels}, 使用头数: {num_heads}")

        # 通道依赖增强模块
        self.channel_attention = ChannelAttention(channels, reduction)

        # 序列全局依赖增强模块（支持任意通道数）
        self.global_attention = GlobalSequenceAttention(channels, sequence_length, num_heads)

        # 序列局部依赖增强模块
        self.local_attention = LocalSequenceAttention(channels, kernel_size)

        # 自适应权重学习
        self.alpha = nn.Parameter(torch.tensor(0.33))
        self.beta = nn.Parameter(torch.tensor(0.33))
        self.gamma = nn.Parameter(torch.tensor(0.33))


    def forward(self, x):
        # 输入维度: [B, C, L]
        batch_size, channels, seq_len = x.shape

        # 通道依赖增强
        channel_enhanced = self.channel_attention(x)

        # 序列全局依赖增强
        global_enhanced = self.global_attention(x)

        # 序列局部依赖增强
        local_enhanced = self.local_attention(x)

        # 自适应加权融合
        alpha = torch.softmax(self.alpha, dim=0)
        beta = torch.softmax(self.beta, dim=0)
        gamma = torch.softmax(self.gamma, dim=0)

        total_weight = alpha + beta + gamma
        w1 = alpha / total_weight
        w2 = beta / total_weight
        w3 = gamma / total_weight

        # 残差连接 + 加权融合
        output = w1 * channel_enhanced + w2 * global_enhanced + w3 * local_enhanced + x

        return output


class ChannelAttention(nn.Module):
    """通道注意力机制 - 增强通道间依赖关系"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 确保reduction不会太大
        reduced_channels = max(1, channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        avg_out = self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1)
        # 全局最大池化
        max_out = self.fc(self.max_pool(x).squeeze(-1)).unsqueeze(-1)

        # 注意力权重
        attention_weights = self.sigmoid(avg_out + max_out)

        return x * attention_weights


class GlobalSequenceAttention(nn.Module):
    """全局序列注意力 - 支持任意通道数的多头注意力"""

    def __init__(self, channels, sequence_length, num_heads=8):
        super(GlobalSequenceAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        # 动态计算每个头的维度
        self.head_dim = channels // num_heads
        # 如果无法整除，使用填充策略
        if channels % num_heads != 0:
            self.need_projection = True
            # 使用线性投影确保维度匹配
            self.proj_query = nn.Linear(channels, num_heads * self.head_dim)
            self.proj_key = nn.Linear(channels, num_heads * self.head_dim)
            self.proj_value = nn.Linear(channels, num_heads * self.head_dim)
            self.proj_output = nn.Linear(num_heads * self.head_dim, channels)
        else:
            self.need_projection = False
            self.head_dim = channels // num_heads
            self.query = nn.Linear(channels, channels)
            self.key = nn.Linear(channels, channels)
            self.value = nn.Linear(channels, channels)

        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        # 输入: [B, C, L] -> [B, L, C]
        x_permuted = x.permute(0, 2, 1)
        batch_size, seq_len, channels = x_permuted.shape

        if self.need_projection:
            # 使用投影确保维度匹配
            x_proj = x_permuted
            Q = self.proj_query(x_proj).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            K = self.proj_key(x_proj).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = self.proj_value(x_proj).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        else:

            Q = self.query(x_permuted).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            K = self.key(x_permuted).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = self.value(x_permuted).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


        # 注意力计算
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = self.softmax(attention_scores)

        # 应用注意力
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        if self.need_projection:
            # 投影回原始通道数
            attention_output = attention_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
            attention_output = self.proj_output(attention_output)
        else:
            attention_output = attention_output.view(batch_size, seq_len, channels)

        # 残差连接和层归一化
        output = self.layer_norm(attention_output + x_permuted)

        # 恢复原始维度: [B, L, C] -> [B, C, L]
        return output.permute(0, 2, 1)


class FlexibleGlobalAttention(nn.Module):
    """更灵活的全局注意力 - 使用分组注意力避免维度问题"""

    def __init__(self, channels, sequence_length, num_groups=4):
        super(FlexibleGlobalAttention, self).__init__()
        self.channels = channels
        self.num_groups = min(num_groups, channels)  # 确保组数不超过通道数
        self.group_size = channels // self.num_groups

        # 每组独立的注意力机制
        self.group_attentions = nn.ModuleList([
            SingleHeadAttention(self.group_size, sequence_length)
            for _ in range(self.num_groups)
        ])

        # 处理剩余通道（如果有）
        self.remaining_channels = channels % self.num_groups
        if self.remaining_channels > 0:
            self.remaining_attention = SingleHeadAttention(self.remaining_channels, sequence_length)

        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        # 输入: [B, C, L] -> [B, L, C]
        x_permuted = x.permute(0, 2, 1)
        batch_size, seq_len, channels = x_permuted.shape

        outputs = []
        start_idx = 0

        # 处理每个组
        for i in range(self.num_groups):
            end_idx = start_idx + self.group_size
            group_x = x_permuted[:, :, start_idx:end_idx]
            group_output = self.group_attentions[i](group_x)
            outputs.append(group_output)
            start_idx = end_idx

        # 处理剩余通道
        if self.remaining_channels > 0:
            remaining_x = x_permuted[:, :, start_idx:]
            remaining_output = self.remaining_attention(remaining_x)
            outputs.append(remaining_output)

        # 合并所有组
        attention_output = torch.cat(outputs, dim=-1)

        # 残差连接和层归一化
        output = self.layer_norm(attention_output + x_permuted)

        # 恢复原始维度
        return output.permute(0, 2, 1)


class SingleHeadAttention(nn.Module):
    """单头注意力，用于分组注意力"""

    def __init__(self, channels, sequence_length):
        super(SingleHeadAttention, self).__init__()
        self.channels = channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.position_encoding = PositionalEncoding(channels, sequence_length)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        # 添加位置编码
        x_encoded = self.position_encoding(x)

        # 线性变换
        Q = self.query(x_encoded)
        K = self.key(x_encoded)
        V = self.value(x_encoded)

        # 注意力计算
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(channels)
        attention_weights = self.softmax(attention_scores)

        # 应用注意力
        attention_output = torch.matmul(attention_weights, V)

        return attention_output


class LocalSequenceAttention(nn.Module):
    """局部序列注意力 - 捕获局部时间模式"""

    def __init__(self, channels, kernel_size=7):
        super(LocalSequenceAttention, self).__init__()
        # 自适应调整kernel_size，确保为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,  # 深度可分离卷积
            bias=False
        )
        self.bn = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 局部特征提取
        local_features = self.conv1d(x)
        local_features = self.bn(local_features)

        # 生成局部注意力权重
        attention_weights = self.sigmoid(local_features)

        return x * attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 使用示例
if __name__ == "__main__":
    # 测试不同通道数的情况
    test_channels = [7, 64, 321]

    for channels in test_channels:
        print(f"\n测试通道数: {channels}")

        # 模拟输入数据
        batch_size, seq_len = 32, 100
        x = torch.randn(batch_size, channels, seq_len)

        # 创建增强模块
        enhancement_module = ChannelSequenceEnhancement(
            channels=channels,
            sequence_length=seq_len
        )

        # 前向传播
        output = enhancement_module(x)

        print(f"输入维度: {x.shape}")
        print(f"输出维度: {output.shape}")
        print(f"模块参数: {sum(p.numel() for p in enhancement_module.parameters()):,}")