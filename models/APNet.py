import torch
import torch.nn as nn
from layers.RevIN import RevIN
import numpy as np
from einops import rearrange
import pandas as pd
from sklearn.manifold import TSNE
import torch
import torch.fft
import matplotlib.pyplot as plt
# from .DKANLinear import KANLinear
# from .DKANLinear import KANLinear
from .FKANLinear import KANLinear
import torch.nn.functional as F

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------- 可选：用谱熵作为连续颜色（不需要ADF/KSS）---------
def spectral_entropy_batch(batch_x_np, eps=1e-12):
    """
    batch_x_np: [B, L, C]
    return: [B]  每个样本一个谱熵（对通道取平均）
    """
    B, L, C = batch_x_np.shape
    out = np.zeros((B,), dtype=np.float64)
    for i in range(B):
        se_c = []
        for c in range(C):
            x = batch_x_np[i, :, c].astype(np.float64)
            x = x - x.mean()
            spec = np.fft.rfft(x)
            psd = (spec.real**2 + spec.imag**2) + eps
            p = psd / psd.sum()
            H = -np.sum(p * np.log(p + eps))
            H_norm = H / np.log(len(p) + eps)
            se_c.append(H_norm)
        out[i] = float(np.mean(se_c))
    return out

@torch.no_grad()
def collect_feats_and_color(model, loader, device, max_batches=50, color_mode="spectral_entropy"):
    """
    color_mode:
      - "none": 不上色
      - "spectral_entropy": 用输入片段谱熵上色（连续值）
      - "variance": 用输入片段方差上色（连续值）
    """
    model.eval()
    feats_list, color_list = [], []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):

        # 适配常见 dataloader 返回： (x, y, x_mark, y_mark)
        # batch_x = batch[0].to(device)
        # # 下面这些如果你的loader没有，就传 None/占位（模型里没用的话）
        # batch_y = batch[1].to(device) if len(batch) > 1 else batch_x
        # batch_x_mark = batch[2].to(device) if len(batch) > 2 else None
        # batch_y_mark = batch[3].to(device) if len(batch) > 3 else None
        batch_x = batch_x.float()
        pred, feat = model(batch_x, batch_x_mark, batch_y, batch_y_mark, return_feat=True)
        feats_list.append(feat.detach().cpu().numpy())   # [B, C]

        if color_mode == "none":
            continue

        bx = batch_x.detach().cpu().numpy()  # [B, L, C]
        if color_mode == "spectral_entropy":
            color = spectral_entropy_batch(bx)           # [B]
        elif color_mode == "variance":
            color = bx.var(axis=(1, 2))                  # [B]
        else:
            raise ValueError("Unknown color_mode")
        color_list.append(color)

    feats = np.concatenate(feats_list, axis=0)
    colors = None if color_mode == "none" else np.concatenate(color_list, axis=0)
    return feats, colors

def plot_tsne(feats, colors=None, title="t-SNE", perplexity=30, random_state=42):
    feats = StandardScaler().fit_transform(feats)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state
    )
    emb = tsne.fit_transform(feats)

    plt.figure(figsize=(6.2, 5.2))
    if colors is None:
        plt.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.7)
    else:
        sc = plt.scatter(emb[:, 0], emb[:, 1], c=colors, s=8, alpha=0.7)
        plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.title(title,fontsize=24)
    plt.tight_layout()
    plt.show()

# --------- 可选：对比 w/o APLC vs with APLC（更像 ablation 佐证）---------
class IdentityFusion(torch.nn.Module):
    def forward(self, x):
        return x

def tsne_compare_with_without_aplc(model_full, loader, device, max_batches=50, color_mode="spectral_entropy"):
    import copy
    # with APLC
    feats1, col1 = collect_feats_and_color(model_full, loader, device, max_batches=max_batches, color_mode=color_mode)
    plot_tsne(feats1, col1, title=f"color={color_mode}")

    # w/o APLC
    model_wo = copy.deepcopy(model_full)
    if hasattr(model_wo, "attn1"):
        model_wo.attn1 = IdentityFusion()
    else:
        raise AttributeError("Model has no attribute `attn1`. Please replace the correct module.")
    feats0, col0 = collect_feats_and_color(model_wo, loader, device, max_batches=max_batches, color_mode=color_mode)
    plot_tsne(feats0, col0, title=f"t-SNE (w/o APLC) | color={color_mode}")

class CrossDomainAttention(nn.Module):
    def __init__(self, L):
        super(CrossDomainAttention, self).__init__()
        self.L = L

        # 线性变换层（用于计算Query, Key, Value）
        self.query = nn.Linear(L, self.L)  # 降维
        self.key = nn.Linear(L, self.L)    # 降维
        self.value = nn.Linear(L, L)               # 保持原维度

        # 融合层（可选）
        self.fusion = nn.Linear(L, L)  # 最终输出调整

        # 归一化
        self.layer_norm = nn.LayerNorm(L)

    def forward(self, a, b):

        a = a.permute(0,2,1)
        b = b.permute(0,2,1)
        B, C, L = a.shape

        a_flat = a
        b_flat = b
        # 2. 计算Query, Key, Value
        q = self.query(a_flat)
        k = self.key(b_flat)
        v = self.value(b_flat)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = F.softmax(attn_scores / (self.L ** 0.5), dim=-1)  # 归一化

        out = torch.matmul(attn_scores, v)  # [B*C, reduced_L, L]

        out = out + a  # 残差连接

        out = self.layer_norm(out)  # LayerNorm在L维度

        out = out.reshape(B,L,C)

        return out
class FreqAttention(nn.Module):
    def __init__(self, in_channels, embed_size, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.amp_attention = nn.Sequential(
            KANLinear(in_channels, (in_channels // reduction)),
            nn.Linear((in_channels // reduction), in_channels),
            nn.Sigmoid()
        )

        self.phase_attention = nn.Sequential(
            KANLinear(in_channels, (in_channels // reduction)),
            nn.Linear((in_channels // reduction), in_channels),
            nn.Tanh()
        )
        self.cross1 = CrossDomainAttention(7)
        self.cross2 = CrossDomainAttention(7)

    def forward(self, x):
        B, L, C = x.shape

        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')  # 输出 [B, L//2+1, C]

        amp = torch.abs(x_freq)  # 振幅 [B, F, C]
        phase = torch.angle(x_freq)  # 相位 [B, F, C]

        cross_amp = self.cross1(phase,amp)
        cross_phase = self.cross2(amp,phase)

        amp_att = self.amp_attention(cross_amp)  # [B, C, F]
        amp_att = amp_att  # [B, F, C]
        amp_weighted = amp * amp_att

        # 4. 相位注意力机制
        phase_att = self.phase_attention(cross_phase)  # [B, C, F]
        phase_att = phase_att # [B, F, C]
        phase_weighted = phase + phase_att

        # 5. 重建复数频域信号
        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)  # [B, F, C]

        # 6. 频域 -> 时域 (逆FFT)
        x_new = torch.fft.irfft(x_freq_new, n=self.embed_size, dim=2, norm='ortho')  # [B, L, C]

        return x_new
class ConvAttentionN(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.in_channels = in_channels
        self.amp_attention = nn.Sequential(
            KANLinear(in_channels, (in_channels // reduction)+7),
            nn.Linear((in_channels // reduction)+7, in_channels),
            nn.Sigmoid()
        )
        self.phase_attention = nn.Sequential(
            KANLinear(in_channels, (in_channels // reduction)+7),
            nn.Linear((in_channels // reduction)+7, in_channels),
            nn.Tanh()
        )

        self.cross11 = CrossDomainAttention(in_channels)
        self.cross22 = CrossDomainAttention(in_channels)
    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0,2,1)
        # 1. 时域 -> 频域 (实数FFT)
        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')  # 输出 [B, L//2+1, C]

        # 2. 提取振幅和相位
        amp = torch.abs(x_freq)  # [B, C, F]
        phase = torch.angle(x_freq)  # [B, C, F]

        # 调整为CrossDomainAttention期望的输入形状
        amp_perm = amp.permute(0, 2, 1)  # [B, C, F] -> [B, F, C]
        phase_perm = phase.permute(0, 2, 1)  # [B, C, F] -> [B, F, C]

        cross_amp = self.cross11(phase_perm, amp_perm)  # 输出 [B, F, C]
        cross_phase = self.cross22(amp_perm, phase_perm)  # 输出 [B, F, C]

        # 修复点：使用permute代替reshape
        cross_amp = cross_amp.permute(0, 2, 1).contiguous()  # [B, F, C] -> [B, C, F]
        cross_phase = cross_phase.permute(0, 2, 1).contiguous()   # [B, F, C] -> [B, C, F]

        # 3. 振幅注意力机制

        amp_att = self.amp_attention(cross_amp)  # [B, C, F]
        amp_weighted = amp * amp_att

        phase_att = self.phase_attention(cross_phase)  # [B, C, F]
        phase_weighted = phase + phase_att

        # 5. 重建复数频域信号
        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)  # [B, F, C]

        # 6. 频域 -> 时域 (逆FFT)
        x_new = torch.fft.irfft(x_freq_new, n=L, dim=2, norm='ortho')  # [B, L, C]
        x_new = x_new.permute(0,2,1)
        return x_new

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2  # 边缘复制填充
        )

    def forward(self, x):

        # 输入维度: [Batch, Length, Channels]
        # 转换为卷积维度: [Batch, Channels, Length]
        x_perm = x.permute(0, 2, 1)

        # 计算趋势项 (移动平均)
        trend = self.avg_pool(x_perm)

        # 转换回原始维度
        trend = trend.permute(0, 2, 1)

        # 季节项 = 原始信号 - 趋势项
        seasonal = x - trend

        return seasonal, trend
class FusionModule(nn.Module):
    def __init__(self, in_channels, embed_size, enc_in, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.freq_attention_c = FreqAttention(in_channels, embed_size, reduction)
        self.conv_attention_n = ConvAttentionN(enc_in, reduction)
        # 可学习的融合权重（初始化为1）
        self.weight_c = nn.Parameter(torch.tensor(1.0))
        self.weight_n = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):

        x_c = self.freq_attention_c(x)  # C维度特征 [B, N, C]
        x_n = self.conv_attention_n(x)   # N维度特征 [B, N, C]

        # 自适应融合 + 残差连接
        out = x + self.weight_c * x_c + self.weight_n * x_n
        return out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.enc_in = configs.enc_in
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc1 = nn.Sequential(
            KANLinear(self.embed_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.pred_len)
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(self.embed_size, self.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size, self.pred_len)
        # )

        self.attn1 = FusionModule(self.embed_size//2+1,self.embed_size,self.enc_in//2+1)

        self.T_embedding = nn.Linear(self.seq_len, self.embed_size)

        self.layernormT = nn.LayerNorm(self.embed_size)
        self.layernormT1 = nn.LayerNorm(self.embed_size)


    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None,return_feat=False):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z
        T = x

        T = T.permute(0, 2, 1)

        T = self.T_embedding(T)

        T = self.layernormT(T)

        T = self.attn1(T)

        T = self.layernormT1(T)
        T_after = T
        T = self.fc1(T)

        x = T.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z
        if return_feat:
            # 推荐：mean-pooling 得到 [B, C]，t-SNE 更稳定
            feat = T_after.mean(dim=1)
            return x, feat
        return x
