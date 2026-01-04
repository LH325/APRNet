import torch
import torch.nn as nn
from layers.RevIN import RevIN
import torch.fft
from .FKANLinear import KANLinear
import torch.nn.functional as F


class CrossDomainAttention(nn.Module):
    def __init__(self, L):
        super(CrossDomainAttention, self).__init__()
        self.L = L
        self.query = nn.Linear(L, self.L)
        self.key = nn.Linear(L, self.L)
        self.value = nn.Linear(L, L)
        self.fusion = nn.Linear(L, L)
        self.layer_norm = nn.LayerNorm(L)

    def forward(self, a, b):
        # a,b: [B, L, C] -> [B, C, L]
        a = a.permute(0, 2, 1)
        b = b.permute(0, 2, 1)
        B, C, L = a.shape

        q = self.query(a)
        k = self.key(b)
        v = self.value(b)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = F.softmax(attn_scores / (self.L ** 0.5), dim=-1)

        out = torch.matmul(attn_scores, v)
        out = out + a
        out = self.layer_norm(out)

        # back to [B, L, C]
        out = out.reshape(B, L, C)
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
        """
        x: [B, L, C]
        这里的 rfft(dim=2) 是对最后一维 C 做 FFT
        所以输出是 [B, L, C//2+1] （不是你原注释里的 [B, L//2+1, C]）
        """
        B, L, C = x.shape

        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, L, F] where F=C//2+1
        amp = torch.abs(x_freq)                          # [B, L, F]
        phase = torch.angle(x_freq)                      # [B, L, F]

        # cross_amp = self.cross1(phase, amp)
        # cross_phase = self.cross2(amp, phase)

        amp_att = self.amp_attention(amp)          # [B, L, F]
        amp_weighted = amp * amp_att

        phase_att = self.phase_attention(phase)    # [B, L, F]
        phase_weighted = phase + phase_att

        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)           # [B, L, F]

        # irfft 回到原长度 = embed_size（也就是 C）
        x_new = torch.fft.irfft(x_freq_new, n=self.embed_size, dim=2, norm='ortho')  # [B, L, C]
        return x_new


class ConvAttentionN(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

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

        self.cross11 = CrossDomainAttention(in_channels)
        self.cross22 = CrossDomainAttention(in_channels)

    def forward(self, x):
        """
        x: [B, L, C]
        内部对 L 做 FFT（dim=2 after permute）
        """
        B, L, C = x.shape
        x_ = x.permute(0, 2, 1)  # [B, C, L]

        x_freq = torch.fft.rfft(x_, dim=2, norm='ortho')  # [B, C, F] where F=L//2+1
        amp = torch.abs(x_freq)                           # [B, C, F]
        phase = torch.angle(x_freq)                       # [B, C, F]

        # amp_perm = amp.permute(0, 2, 1)                   # [B, F, C]
        # phase_perm = phase.permute(0, 2, 1)               # [B, F, C]

        # cross_amp = self.cross11(phase_perm, amp_perm)    # [B, F, C]
        # cross_phase = self.cross22(amp_perm, phase_perm)  # [B, F, C]

        # cross_amp = cross_amp.permute(0, 2, 1).contiguous()     # [B, C, F]
        # cross_phase = cross_phase.permute(0, 2, 1).contiguous() # [B, C, F]

        amp_att = self.amp_attention(amp)           # [B, C, F]
        amp_weighted = amp * amp_att

        phase_att = self.phase_attention(phase)     # [B, C, F]
        phase_weighted = phase + phase_att

        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)            # [B, C, F]

        x_new = torch.fft.irfft(x_freq_new, n=L, dim=2, norm='ortho')  # [B, C, L]
        x_new = x_new.permute(0, 2, 1)                    # [B, L, C]
        return x_new


class FusionModule(nn.Module):
    def __init__(self, in_channels, embed_size, enc_in, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.freq_attention_c = FreqAttention(in_channels, embed_size, reduction)
        self.conv_attention_n = ConvAttentionN(enc_in, reduction)
        self.weight_c = nn.Parameter(torch.tensor(1.0))
        self.weight_n = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_c = self.freq_attention_c(x)
        x_n = self.conv_attention_n(x)
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

        # self.fc1 = nn.Sequential(
        #     KANLinear(self.embed_size, self.hidden_size),
        #     nn.Linear(self.hidden_size, self.pred_len)
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )
        # 保持你原结构（不改）
        self.attn1 = FusionModule(self.embed_size // 2 + 1, self.embed_size, self.enc_in // 2 + 1)


        self.T_embedding = nn.Linear(self.seq_len, self.embed_size)
        self.layernormT = nn.LayerNorm(self.embed_size)
        self.layernormT1 = nn.LayerNorm(self.embed_size)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None, return_T: bool = False):
        """
        return_T=False: 只返回预测 y
        return_T=True : 返回 (y, T0, T1)
          - T0: attn1 前的表示（你要的“原始T”）
          - T1: attn1 后的表示（你要的“增加attn1后的T”）
        """
        z = self.revin_layer(x, 'norm')       # [B, seq, C]

        T = z.permute(0, 2, 1)               # [B, C, seq]
        T = self.T_embedding(T)              # [B, C, embed]

        T0 = self.layernormT(T)              # ===== 原始T（未经过 attn1）=====
        t = T0
        T1 = self.attn1(T0)
        b = T1
        # ===== 加了 attn1 的 T =====
        T1 = self.layernormT1(T1)

        T_pred = self.fc1(T1)                # [B, C, pred]
        y = T_pred.permute(0, 2, 1)          # [B, pred, C]
        y = self.revin_layer(y, 'denorm')

        if return_T:
            return y, t, b
        return y
