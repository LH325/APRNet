import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

from layers.RevIN import RevIN
from .DKAN import KANLinear


class TCFMOutput(NamedTuple):
    """Frequency representations produced by TCFM."""

    temporal_amp: torch.Tensor
    temporal_phase: torch.Tensor
    channel_amp: torch.Tensor
    channel_phase: torch.Tensor
    channel_length: int


class CrossDomainAttention(nn.Module):
    """Bidirectional amplitude-phase cross-attention."""

    def __init__(self, L):
        super(CrossDomainAttention, self).__init__()
        self.L = L

        self.query = nn.Linear(L, self.L)
        self.key = nn.Linear(L, self.L)
        self.value = nn.Linear(L, L)

        self.fusion = nn.Linear(L, L)
        self.layer_norm = nn.LayerNorm(L)

    def forward(self, a, b):
        a = a.permute(0, 2, 1)
        b = b.permute(0, 2, 1)
        B, C, L = a.shape

        a_flat = a
        b_flat = b

        q = self.query(a_flat)
        k = self.key(b_flat)
        v = self.value(b_flat)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = F.softmax(attn_scores / (self.L ** 0.5), dim=-1)

        out = torch.matmul(attn_scores, v)
        out = out + a
        out = self.layer_norm(out)
        out = out.reshape(B, L, C)

        return out


class TCFM(nn.Module):
    """
    Temporal-Channel Frequency Modeling.

    The normalized input is first projected from the original sequence space
    to the high-dimensional latent space controlled by s1. Dual-view Fourier
    modeling is then performed along the latent temporal and channel axes.
    """

    def __init__(self, in_, high_dimension):
        super().__init__()

        self.in_ = in_
        self.high_dimension = high_dimension

        self.T_embedding = nn.Linear(self.in_, self.high_dimension)
        self.layernormT = nn.LayerNorm(self.high_dimension)

    def forward(self, T):
        # T: [B, N, C]

        T = T.permute(0, 2, 1)  # [B, C, N]

        T = self.T_embedding(T)  # [B, C, d_model * s1]
        T = self.layernormT(T)

        B, C, L = T.shape

        # Temporal-frequency view: FFT along the high-dimensional latent axis.
        temporal_freq = torch.fft.rfft(T, dim=2, norm='ortho')
        temporal_amp = torch.abs(temporal_freq)
        temporal_phase = torch.angle(temporal_freq)

        # Channel-frequency view: FFT along the variable/channel axis.
        channel_input = T.permute(0, 2, 1)  # [B, L, C]
        channel_freq = torch.fft.rfft(channel_input, dim=2, norm='ortho')
        channel_amp = torch.abs(channel_freq)
        channel_phase = torch.angle(channel_freq)

        frequency_state = TCFMOutput(
            temporal_amp=temporal_amp,
            temporal_phase=temporal_phase,
            channel_amp=channel_amp,
            channel_phase=channel_phase,
            channel_length=C,
        )

        return T, frequency_state

class _TemporalAPGC(nn.Module):
    """APGC branch for the temporal-frequency representation."""

    def __init__(self, freq_bins, high_dimension, channel, reduction=16, ETT=True, cross=True):
        super().__init__()

        self.freq_bins = freq_bins
        self.high_dimension = high_dimension
        self.ETT = ETT
        self.cross = cross
        self.amp_attention = nn.Sequential(
            KANLinear(freq_bins, (freq_bins // reduction)),
            nn.Linear((freq_bins // reduction), freq_bins),
            nn.Sigmoid()
        )

        self.phase_attention = nn.Sequential(
            KANLinear(freq_bins, (freq_bins // reduction)),
            nn.Linear((freq_bins // reduction), freq_bins),
            nn.Tanh()
        )

        if self.cross:

            self.cross1 = CrossDomainAttention(channel)
            self.cross2 = CrossDomainAttention(channel)


    def forward(self, amp, phase):
        if self.cross:

            cross_amp = self.cross1(phase, amp)
            cross_phase = self.cross2(amp, phase)

            amp_att = self.amp_attention(cross_amp)  # [B, C, F]
            amp_att = amp_att  # [B, F, C]
            amp_weighted = amp * amp_att

            phase_att = self.phase_attention(cross_phase)  # [B, C, F]
            phase_att = phase_att  # [B, F, C]
            phase_weighted = phase + phase_att

        else:
            amp_att = self.amp_attention(amp)  # [B, C, F]
            amp_att = amp_att  # [B, F, C]
            amp_weighted = amp * amp_att

            phase_att = self.phase_attention(phase)  # [B, C, F]
            phase_att = phase_att  # [B, F, C]
            phase_weighted = phase + phase_att

        return amp_weighted, phase_weighted


class _ChannelAPGC(nn.Module):
    """APGC branch for the channel-frequency representation."""

    def __init__(self, channel_freq_bins, reduction=16, ETT=True, cross=True):
        super().__init__()

        self.channel_freq_bins = channel_freq_bins
        self.ETT = ETT
        self.cross = cross
        if self.ETT:
            self.amp_attention = nn.Sequential(
                KANLinear(channel_freq_bins, (channel_freq_bins // reduction) + 7),
                nn.Linear((channel_freq_bins // reduction) + 7, channel_freq_bins),
                nn.Sigmoid()
            )
            self.phase_attention = nn.Sequential(
                KANLinear(channel_freq_bins, (channel_freq_bins // reduction) + 7),

                nn.Linear((channel_freq_bins // reduction) + 7, channel_freq_bins),
                nn.Tanh()
            )
        else:
            self.amp_attention = nn.Sequential(
                KANLinear(channel_freq_bins, (channel_freq_bins // reduction)),
                nn.Linear((channel_freq_bins // reduction), channel_freq_bins),
                nn.Sigmoid()
            )
            self.phase_attention = nn.Sequential(
                KANLinear(channel_freq_bins, (channel_freq_bins // reduction)),
                nn.Linear((channel_freq_bins // reduction), channel_freq_bins),
                nn.Tanh()
            )
        if self.cross:
            self.cross11 = CrossDomainAttention(channel_freq_bins)
            self.cross22 = CrossDomainAttention(channel_freq_bins)

    def forward(self, amp, phase):
        if self.cross:

            amp_perm = amp.permute(0, 2, 1)  # [B, C, F] -> [B, F, C]
            phase_perm = phase.permute(0, 2, 1)  # [B, C, F] -> [B, F, C]

            cross_amp = self.cross11(phase_perm, amp_perm)  # 输出 [B, F, C]
            cross_phase = self.cross22(amp_perm, phase_perm)  # 输出 [B, F, C]

            cross_amp = cross_amp.permute(0, 2, 1).contiguous()  # [B, F, C] -> [B, C, F]
            cross_phase = cross_phase.permute(0, 2, 1).contiguous()  # [B, F, C] -> [B, C, F]

            amp_att = self.amp_attention(cross_amp)  # [B, C, F]
            amp_weighted = amp * amp_att

            phase_att = self.phase_attention(cross_phase)  # [B, C, F]
            phase_weighted = phase + phase_att

        else:
            amp_att = self.amp_attention(amp)  # [B, C, F]
            amp_weighted = amp * amp_att

            phase_att = self.phase_attention(phase)  # [B, C, F]
            phase_weighted = phase + phase_att

        return amp_weighted, phase_weighted


class APGC(nn.Module):
    """
    Amplitude-Phase Global Correlation.

    It performs amplitude-phase cross-correlation, D-KAN-based global
    calibration, dual-view inverse Fourier reconstruction, and residual fusion.
    """

    def __init__(self, temporal_freq_bins, high_dimension, channel_freq_bins, channel,
                 ETT, cross,reduction=16):
        super().__init__()

        self.high_dimension = high_dimension

        self.temporal_apgc = _TemporalAPGC(
            temporal_freq_bins,
            high_dimension,
            channel,
            reduction,
            ETT,
            cross
        )
        self.channel_apgc = _ChannelAPGC(
            channel_freq_bins,
            reduction,
            ETT,
            cross,
        )

        self.temporal_weight = nn.Parameter(torch.tensor(1.0))
        self.channel_weight = nn.Parameter(torch.tensor(1.0))

        self.layernormT1 = nn.LayerNorm(self.high_dimension)

    @staticmethod
    def _compose_spectrum(amp, phase):
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return torch.complex(real, imag)

    def forward(self, x, frequency_state):
        temporal_amp, temporal_phase = self.temporal_apgc(
            frequency_state.temporal_amp,
            frequency_state.temporal_phase,
        )

        temporal_spectrum = self._compose_spectrum(temporal_amp, temporal_phase)
        temporal_feature = torch.fft.irfft(
            temporal_spectrum,
            n=self.high_dimension,
            dim=2,
            norm='ortho',
        )

        channel_amp, channel_phase = self.channel_apgc(
            frequency_state.channel_amp,
            frequency_state.channel_phase,
        )

        channel_spectrum = self._compose_spectrum(channel_amp, channel_phase)
        channel_feature = torch.fft.irfft(
            channel_spectrum,
            n=frequency_state.channel_length,
            dim=2,
            norm='ortho',
        )
        channel_feature = channel_feature.permute(0, 2, 1)

        y = (
                x
                + self.temporal_weight * temporal_feature
                + self.channel_weight * channel_feature
        )

        out = self.layernormT1(y)
        return out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)


        self.in_ = configs.seq_len
        self.d_model = configs.d_model
        self.s1 = configs.s1
        self.s2 = configs.s2

        self.channel_modeling = configs.enc_in

        # s1 expands the latent representation before frequency analysis.
        self.high_dimension = self.d_model * self.s1

        # s2 expands the prediction function space after reconstruction.
        self.predictor_dimension = self.d_model * self.s2

        self.w = nn.Parameter(
            self.scale * torch.randn(1, self.high_dimension)
        )
        self.kan_linear = configs.use_kanlinear
        self.ETT = configs.ETT
        self.cross = configs.cross

        if self.kan_linear:
            self.mlp = nn.Sequential(
                KANLinear(self.high_dimension, self.predictor_dimension),
                nn.Linear(self.predictor_dimension, self.pred_len)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.high_dimension, self.predictor_dimension),
                nn.ReLU(),
                nn.Linear(self.predictor_dimension, self.pred_len)
            )
        self.tcfm = TCFM(
            in_=self.in_,
            high_dimension=self.d_model * self.s1
        )
        self.apgc = APGC(
            temporal_freq_bins=(self.d_model * self.s1) // 2 + 1,
            high_dimension=self.d_model * self.s1,
            channel_freq_bins=self.channel_modeling // 2 + 1,
            channel = self.channel_modeling,
            ETT=self.ETT,
            cross=self.cross
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        T = z
        'tcfm'
        T, frequency_state = self.tcfm(T)
        'apgc'
        T = self.apgc(T, frequency_state)
        'mlp'
        T = self.mlp(T)
        x = T.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x