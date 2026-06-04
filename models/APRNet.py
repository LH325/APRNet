import torch.nn as nn
from layers.RevIN import RevIN

import torch
import torch.fft

from .DKAN import KANLinear
import torch.nn.functional as F

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
    def __init__(self, in_channels, embed_size, reduction=16, ETT=True, cross=True):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.ETT = ETT
        self.cross = cross
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

        if self.cross:
            self.cross1 = CrossDomainAttention(7)
            self.cross2 = CrossDomainAttention(7)

    def forward(self, x):
        B, L, C = x.shape

        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')  # 输出 [B, L//2+1, C]

        amp = torch.abs(x_freq)  # 振幅 [B, F, C]
        phase = torch.angle(x_freq)  # 相位 [B, F, C]
        
        if self.cross:
            cross_amp = self.cross1(phase,amp)
            cross_phase = self.cross2(amp,phase)

            amp_att = self.amp_attention(cross_amp)  # [B, C, F]
            amp_att = amp_att  # [B, F, C]
            amp_weighted = amp * amp_att


            phase_att = self.phase_attention(cross_phase)  # [B, C, F]
            phase_att = phase_att # [B, F, C]
            phase_weighted = phase + phase_att
        
        else:
            amp_att = self.amp_attention(amp)  # [B, C, F]
            amp_att = amp_att  # [B, F, C]
            amp_weighted = amp * amp_att

            phase_att = self.phase_attention(phase)  # [B, C, F]
            phase_att = phase_att # [B, F, C]
            phase_weighted = phase + phase_att

        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)  # [B, F, C]

        x_new = torch.fft.irfft(x_freq_new, n=self.embed_size, dim=2, norm='ortho')  # [B, L, C]

        return x_new
class ConvAttentionN(nn.Module):
    def __init__(self, in_channels, reduction=16, ETT=True, cross=True):
        super().__init__()

        self.in_channels = in_channels
        self.ETT = ETT
        self.cross = cross
        if self.ETT:
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
        else:
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
        if self.cross:

            self.cross11 = CrossDomainAttention(in_channels)
            self.cross22 = CrossDomainAttention(in_channels)
    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0,2,1)

        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')  # 输出 [B, L//2+1, C]

        # 2. 提取振幅和相位
        amp = torch.abs(x_freq)  # [B, C, F]
        phase = torch.angle(x_freq)  # [B, C, F]

        if self.cross:

            amp_perm = amp.permute(0, 2, 1)  # [B, C, F] -> [B, F, C]
            phase_perm = phase.permute(0, 2, 1)  # [B, C, F] -> [B, F, C]

            cross_amp = self.cross11(phase_perm, amp_perm)  # 输出 [B, F, C]
            cross_phase = self.cross22(amp_perm, phase_perm)  # 输出 [B, F, C]

            cross_amp = cross_amp.permute(0, 2, 1).contiguous()  # [B, F, C] -> [B, C, F]
            cross_phase = cross_phase.permute(0, 2, 1).contiguous()   # [B, F, C] -> [B, C, F]

            amp_att = self.amp_attention(cross_amp)  # [B, C, F]
            amp_weighted = amp * amp_att

            phase_att = self.phase_attention(cross_phase)  # [B, C, F]
            phase_weighted = phase + phase_att
        
        else:
            amp_att = self.amp_attention(amp)  # [B, C, F]
            amp_weighted = amp * amp_att

            phase_att = self.phase_attention(phase)  # [B, C, F]
            phase_weighted = phase + phase_att

        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)  # [B, F, C]

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


        x_perm = x.permute(0, 2, 1)

        trend = self.avg_pool(x_perm)

        trend = trend.permute(0, 2, 1)

        seasonal = x - trend

        return seasonal, trend
class FusionModule(nn.Module):
    def __init__(self, in_channels, embed_size, enc_in, ETT, cross, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.freq_attention_c = FreqAttention(in_channels, embed_size, reduction, ETT, cross)
        self.conv_attention_n = ConvAttentionN(enc_in, reduction, ETT, cross)

        self.weight_c = nn.Parameter(torch.tensor(1.0))
        self.weight_n = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):

        x_c = self.freq_attention_c(x)  # C维度特征 [B, N, C]
        x_n = self.conv_attention_n(x)   # N维度特征 [B, N, C]

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
        self.kan_linear = configs.use_kanlinear
        self.ETT = configs.ETT
        self.cross = configs.cross

        if self.kan_linear:        
            self.fc1 = nn.Sequential(
                KANLinear(self.embed_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.pred_len)
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(self.embed_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.pred_len)
            )

        self.attn1 = FusionModule(self.embed_size//2+1,self.embed_size,self.enc_in//2+1, self.ETT, self.cross)


        self.T_embedding = nn.Linear(self.seq_len, self.embed_size)

        self.layernormT = nn.LayerNorm(self.embed_size)
        self.layernormT1 = nn.LayerNorm(self.embed_size)


    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z
        T = x


        T = T.permute(0, 2, 1)
        T = self.T_embedding(T)

        T = self.layernormT(T)

        T = self.attn1(T)

        T = self.layernormT1(T)

        T = self.fc1(T)

        x = T.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x
