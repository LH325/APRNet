# coupling_analysis.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

# ======== 你的依赖 ========
from layers.RevIN import RevIN
from .FKANLinear import KANLinear


# -------------------------
# Utils: phase unwrap + delta
# -------------------------
def unwrap_phase_torch(phase: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    phase: Tensor, in radians
    unwrap along dim (default along L dimension)
    """
    # torch.unwrap is available in newer versions; implement a robust manual unwrap
    # unwrap: add/subtract 2pi when jump > pi
    diff = torch.diff(phase, dim=dim)
    pi = torch.tensor(np.pi, device=phase.device, dtype=phase.dtype)
    twopi = 2 * pi

    # map to (-pi, pi]
    diff_mod = (diff + pi) % (twopi) - pi

    # special-case: if diff_mod == -pi and diff > 0, set to pi
    diff_mod = torch.where((diff_mod == -pi) & (diff > 0), pi, diff_mod)

    # cumulative sum of corrected diffs
    phase_unwrapped = torch.cat(
        [phase.select(dim, 0).unsqueeze(dim),
         phase.select(dim, 0).unsqueeze(dim) + torch.cumsum(diff_mod, dim=dim)],
        dim=dim
    )
    return phase_unwrapped


def phase_increment(phase: torch.Tensor, dim: int = 1, unwrap: bool = True) -> torch.Tensor:
    """
    phase: [B, L, F] (or similar)
    return delta_phi: [B, L-1, F]
    """
    if unwrap:
        phase = unwrap_phase_torch(phase, dim=dim)
    dphi = torch.diff(phase, dim=dim)
    return dphi


# -------------------------
# Coupling metrics (MI / Corr)
# -------------------------
def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().reshape(-1).numpy()


def compute_coupling_per_freq(
    amp: torch.Tensor,
    phase: torch.Tensor,
    delta_dim: int = 1,
    unwrap: bool = True,
):
    """
    amp:   [B, L, F]
    phase: [B, L, F]
    compute coupling between A_{t,f} and Δφ_{t,f} per frequency f

    return:
      corr_f: [F]
      mi_f:   [F]
    """
    assert amp.ndim == 3 and phase.ndim == 3, "Expect amp/phase shape [B, L, F]"
    B, L, Freq = amp.shape

    dphi = phase_increment(phase, dim=delta_dim, unwrap=unwrap)  # [B, L-1, F]
    # align amp to delta_phi time axis
    amp_aligned = amp[:, 1:, :]  # [B, L-1, F]

    corr_f = np.zeros(Freq, dtype=np.float64)
    mi_f = np.zeros(Freq, dtype=np.float64)

    for f in range(Freq):
        a = _to_numpy_1d(amp_aligned[:, :, f])
        p = _to_numpy_1d(dphi[:, :, f])

        # Corr
        if np.std(a) < 1e-12 or np.std(p) < 1e-12:
            corr_f[f] = 0.0
        else:
            corr_f[f] = pearsonr(a, p)[0]

        # MI (regression MI)
        # note: mutual_info_regression expects X: [N, d], y: [N]
        try:
            mi_f[f] = mutual_info_regression(a.reshape(-1, 1), p, discrete_features=False, random_state=0)[0]
        except Exception:
            mi_f[f] = np.nan

    return corr_f, mi_f


def compute_coupling_heatmap_freq_channel(
    amp: torch.Tensor,
    phase: torch.Tensor,
    delta_dim: int = 1,
    unwrap: bool = True,
    metric: str = "mi",
):
    """
    给耦合热力图用：输出 [L, F] 或 [F, L] 都可。
    这里我们按 "channel/position(L) × frequency(F)" 输出 [L, F]。

    amp/phase: [B, L, F]
    metric: "mi" or "corr"
    """
    assert metric in ["mi", "corr"]
    dphi = phase_increment(phase, dim=delta_dim, unwrap=unwrap)  # [B, L-1, F]
    amp_aligned = amp[:, 1:, :]  # [B, L-1, F]

    B, Lm1, Freq = amp_aligned.shape
    heat = np.zeros((Lm1, Freq), dtype=np.float64)

    for l in range(Lm1):
        for f in range(Freq):
            a = _to_numpy_1d(amp_aligned[:, l, f])
            p = _to_numpy_1d(dphi[:, l, f])

            if metric == "corr":
                if np.std(a) < 1e-12 or np.std(p) < 1e-12:
                    heat[l, f] = 0.0
                else:
                    heat[l, f] = pearsonr(a, p)[0]
            else:
                try:
                    heat[l, f] = mutual_info_regression(a.reshape(-1, 1), p, discrete_features=False, random_state=0)[0]
                except Exception:
                    heat[l, f] = np.nan

    return heat  # [L-1, F]


# -------------------------
# Plotting
# -------------------------
def plot_coupling_strength_vs_freq(corr_f, mi_f, title="Coupling Strength vs Frequency"):
    freq = np.arange(len(mi_f))

    plt.figure(figsize=(7, 4))
    plt.plot(freq, mi_f, label="Mutual Information")
    plt.plot(freq, corr_f, label="Correlation")
    plt.xlabel("Frequency Index")
    plt.ylabel("Coupling Strength")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_coupling_heatmap(heat, title="Coupling Heatmap (L × F)", xlabel="Frequency Index", ylabel="Channel/Position Index"):
    """
    heat: [L-1, F]
    """
    plt.figure(figsize=(8, 4.2))
    plt.imshow(heat, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# Your model blocks (with diag support)
# =========================
class CrossDomainAttention(nn.Module):
    def __init__(self, L):
        super(CrossDomainAttention, self).__init__()
        self.L = L
        self.query = nn.Linear(L, self.L)
        self.key   = nn.Linear(L, self.L)
        self.value = nn.Linear(L, L)
        self.fusion = nn.Linear(L, L)
        self.layer_norm = nn.LayerNorm(L)

    def forward(self, a, b):
        # a, b: [B, L, C] -> permute to [B, C, L]
        a = a.permute(0, 2, 1)
        b = b.permute(0, 2, 1)
        B, C, L = a.shape

        q = self.query(a)                # [B, C, L]
        k = self.key(b)                  # [B, C, L]
        v = self.value(b)                # [B, C, L]

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, C, C]
        attn_scores = F.softmax(attn_scores / (self.L ** 0.5), dim=-1)

        out = torch.matmul(attn_scores, v)  # [B, C, L]
        out = out + a
        out = self.layer_norm(out)

        out = out.reshape(B, L, C)  # back to [B, L, C]
        return out


class FreqAttention(nn.Module):
    """
    注意：你的实现里 rfft 是在 dim=2 上做的，
    所以如果 x: [B, L, C_last], 则：
      x_freq: [B, L, F] (F=C_last//2+1)
      amp/phase: [B, L, F]
    """
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

        # 这里 L=7 是你原写死的；更稳妥做法是传入 L 或用 LazyLinear。
        # 我保持你原逻辑：如果 L 不是 7，请把这里改成真实 L。
        self.cross1 = CrossDomainAttention(7)
        self.cross2 = CrossDomainAttention(7)

    def forward(self, x, return_diag=False):
        # x: [B, L, C_last]
        B, L, C_last = x.shape

        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, L, F]
        amp = torch.abs(x_freq)                          # [B, L, F]
        phase = torch.angle(x_freq)                      # [B, L, F]

        cross_amp   = self.cross1(phase, amp)            # [B, L, F] (shape matches by design)
        cross_phase = self.cross2(amp, phase)            # [B, L, F]

        amp_att = self.amp_attention(cross_amp)          # [B, L, F]
        amp_weighted = amp * amp_att

        phase_att = self.phase_attention(cross_phase)    # [B, L, F]
        phase_weighted = phase + phase_att

        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)           # [B, L, F]

        x_new = torch.fft.irfft(x_freq_new, n=self.embed_size, dim=2, norm='ortho')  # [B, L, C_last]

        if return_diag:
            return x_new, {"amp": amp, "phase": phase, "cross_amp": cross_amp, "cross_phase": cross_phase}

        return x_new


class ConvAttentionN(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        self.amp_attention = nn.Sequential(
            KANLinear(in_channels, (in_channels // reduction) + 7),
            nn.Linear((in_channels // reduction) + 7, in_channels),
            nn.Sigmoid()
        )
        self.phase_attention = nn.Sequential(
            KANLinear(in_channels, (in_channels // reduction) + 7),
            nn.Linear((in_channels // reduction) + 7, in_channels),
            nn.Tanh()
        )

        self.cross11 = CrossDomainAttention(in_channels)
        self.cross22 = CrossDomainAttention(in_channels)

    def forward(self, x, return_diag=False):
        # x: [B, L, C]
        B, L, C = x.shape
        x = x.permute(0, 2, 1)                           # [B, C, L]

        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, C, F]
        amp = torch.abs(x_freq)                          # [B, C, F]
        phase = torch.angle(x_freq)                      # [B, C, F]

        amp_perm = amp.permute(0, 2, 1)                  # [B, F, C]
        phase_perm = phase.permute(0, 2, 1)              # [B, F, C]

        cross_amp = self.cross11(phase_perm, amp_perm)   # [B, F, C]
        cross_phase = self.cross22(amp_perm, phase_perm) # [B, F, C]

        cross_amp = cross_amp.permute(0, 2, 1).contiguous()     # [B, C, F]
        cross_phase = cross_phase.permute(0, 2, 1).contiguous() # [B, C, F]

        amp_att = self.amp_attention(cross_amp)          # [B, C, F]
        amp_weighted = amp * amp_att

        phase_att = self.phase_attention(cross_phase)    # [B, C, F]
        phase_weighted = phase + phase_att

        real = amp_weighted * torch.cos(phase_weighted)
        imag = amp_weighted * torch.sin(phase_weighted)
        x_freq_new = torch.complex(real, imag)           # [B, C, F]

        x_new = torch.fft.irfft(x_freq_new, n=L, dim=2, norm='ortho')  # [B, C, L]
        x_new = x_new.permute(0, 2, 1)                   # [B, L, C]

        if return_diag:
            # 这里为了统一耦合分析接口，建议把 amp/phase 也转成 [B, L, F] 形式
            # 目前 amp/phase 是 [B, C, F]，转为 [B, L, F] 需要你定义 L 对应哪一维；
            # 我这里原样返回，同时也给一个 [B, L, F] 的版本（把 C 当作 L）
            diag = {
                "amp": amp.permute(0, 1, 2),          # [B, C, F]
                "phase": phase.permute(0, 1, 2),      # [B, C, F]
                "cross_amp": cross_amp,               # [B, C, F]
                "cross_phase": cross_phase,           # [B, C, F]
            }
            return x_new, diag

        return x_new


class FusionModule(nn.Module):
    def __init__(self, in_channels, embed_size, enc_in, reduction=16):
        super().__init__()
        self.freq_attention_c = FreqAttention(in_channels, embed_size, reduction)
        self.conv_attention_n = ConvAttentionN(enc_in, reduction)
        self.weight_c = nn.Parameter(torch.tensor(1.0))
        self.weight_n = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, return_diag=False):
        if return_diag:
            x_c, diag_c = self.freq_attention_c(x, return_diag=True)
            x_n, diag_n = self.conv_attention_n(x, return_diag=True)
            out = x + self.weight_c * x_c + self.weight_n * x_n
            return out, {"freqC": diag_c, "freqN": diag_n}
        else:
            x_c = self.freq_attention_c(x, return_diag=False)
            x_n = self.conv_attention_n(x, return_diag=False)
            out = x + self.weight_c * x_c + self.weight_n * x_n
            return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.enc_in = configs.enc_in

        self.fc1 = nn.Sequential(
            KANLinear(self.embed_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        # 注意：你这里的 in_channels / enc_in 是基于 rfft 后维度的写法，我保持不改
        self.attn1 = FusionModule(self.embed_size // 2 + 1, self.embed_size, self.enc_in // 2 + 1)

        self.T_embedding = nn.Linear(self.seq_len, self.embed_size)
        self.layernormT = nn.LayerNorm(self.embed_size)
        self.layernormT1 = nn.LayerNorm(self.embed_size)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_diag=False):
        # x: [B, seq_len, enc_in]
        z = self.revin_layer(x, 'norm')       # [B, seq_len, enc_in]

        T = z.permute(0, 2, 1)               # [B, enc_in, seq_len]
        T = self.T_embedding(T)              # [B, enc_in, embed_size]
        T = self.layernormT(T)

        if return_diag:
            T, diag = self.attn1(T, return_diag=True)
        else:
            T = self.attn1(T, return_diag=False)

        T = self.layernormT1(T)
        T = self.fc1(T)                      # [B, enc_in, pred_len]

        out = T.permute(0, 2, 1)             # [B, pred_len, enc_in]
        out = self.revin_layer(out, 'denorm')

        if return_diag:
            return out, diag
        return out


# =========================
# One-call analysis entry (for test loop)
# =========================
@torch.no_grad()
def analyze_batch_coupling(
    model: nn.Module,
    batch_x: torch.Tensor,
    title_prefix: str = "APRNet",
    use_freq_branch: str = "freqC",   # "freqC" or "freqN"
    device: str = "cuda",
    delta_dim_for_L: int = 1,         # 对 [B, L, F] 的 L 维做 Δφ
    unwrap: bool = True,
    plot: bool = True,
):
    """
    在 test 里直接调用：
        out, diag = model(batch_x, ..., return_diag=True)
        然后本函数会对 diag 做 MI/corr 并画图

    这里默认用 freqC 分支的 amp/phase（形状 [B, L, F]）
    """
    model.eval()
    model = model.to(device)
    batch_x = batch_x.to(device)

    out, diag = model(batch_x, None, None, None, return_diag=True)

    if use_freq_branch not in diag:
        raise KeyError(f"diag has keys {list(diag.keys())}, cannot find {use_freq_branch}")

    # 取你最关心的分支
    branch = diag[use_freq_branch]

    # freqC 的 amp/phase 是 [B, L, F]，可直接做 MI/corr
    amp = branch["amp"]
    phase = branch["phase"]

    # safety: ensure [B, L, F]
    if amp.ndim != 3:
        raise ValueError(f"Expect amp ndim=3, got {amp.shape}")
    if phase.shape != amp.shape:
        raise ValueError(f"phase shape {phase.shape} != amp shape {amp.shape}")

    corr_f, mi_f = compute_coupling_per_freq(amp, phase, delta_dim=delta_dim_for_L, unwrap=unwrap)

    # heatmaps: [L-1, F]
    heat_mi = compute_coupling_heatmap_freq_channel(amp, phase, delta_dim=delta_dim_for_L, unwrap=unwrap, metric="mi")
    heat_corr = compute_coupling_heatmap_freq_channel(amp, phase, delta_dim=delta_dim_for_L, unwrap=unwrap, metric="corr")

    if plot:
        plot_coupling_strength_vs_freq(
            corr_f, mi_f,
            title=f"{title_prefix} | {use_freq_branch} | Coupling Strength vs Frequency"
        )
        plot_coupling_heatmap(
            heat_mi,
            title=f"{title_prefix} | {use_freq_branch} | MI Heatmap (L-1 × F)",
            xlabel="Frequency Index", ylabel="Channel/Position Index"
        )
        plot_coupling_heatmap(
            heat_corr,
            title=f"{title_prefix} | {use_freq_branch} | Corr Heatmap (L-1 × F)",
            xlabel="Frequency Index", ylabel="Channel/Position Index"
        )

    return {
        "corr_per_freq": corr_f,
        "mi_per_freq": mi_f,
        "heat_mi": heat_mi,
        "heat_corr": heat_corr,
        "out": out.detach().cpu(),
    }
