import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import einops

# ===== 你的工程里的依赖（确保可导入） =====
from layers.RevIN import RevIN
from layers.vlm_manager import VLMManager


# ---------------------------
# 工具函数：指标与相关性
# ---------------------------
def _row_normalize(a, eps=1e-8):
    s = a.sum(dim=-1, keepdim=True) + eps
    return a / s

def _kl_mean(P, Q, eps=1e-8):
    kl = (P * (torch.log(P + eps) - torch.log(Q + eps))).sum(dim=-1)  # [*, Tq]
    return kl.mean().item()

def _cka_linear(X, Y, eps=1e-8):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    K = X @ X.t()
    L = Y @ Y.t()
    hsic = (K * L).sum()
    norm = torch.sqrt((K * K).sum() * (L * L).sum() + eps)
    return (hsic / (norm + eps)).item()

def _info_nce_proxy(H, F, tau=0.07):
    H = torch.nn.functional.normalize(H, dim=-1)
    F = torch.nn.functional.normalize(F, dim=-1)
    sim = H @ F.t()           # [N,N]
    pos = sim.diag()          # [N]
    logits = sim / tau
    lse = torch.logsumexp(logits, dim=1)
    return (pos / tau - lse).mean().item()

def _to_ND_tokens(t):
    """
    将特征张量映射为 [N, D]，便于 CKA/InfoNCE：
    - [B, C, N]  -> [B*N, C]
    - [B, N, C]  -> [B*N, C]
    """
    if t.dim() != 3:
        raise ValueError("Expect 3D tensor for token features.")
    if t.size(1) <= t.size(2):  # [B, C, N]
        B, C, N = t.shape
        return t.transpose(1, 2).reshape(B * N, C)
    else:                       # [B, N, C]
        B, N, C = t.shape
        return t.reshape(B * N, C)

def _subsample_tokens(ND, max_tokens=4096):
    N = ND.size(0)
    if N <= max_tokens:
        return ND
    idx = torch.randperm(N, device=ND.device)[:max_tokens]
    return ND[idx]


# ---------------------------
# 注意力包装器：严格因果 + 软因果（诊断）
# ---------------------------
class CausalMultiheadAttentionWrapper(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.4, soft_penalty=-4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.soft_penalty = soft_penalty  # e.g., -4.0
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    @staticmethod
    def _causal_mask(Tq, Tk, device, soft=False, soft_penalty=-4.0):
        i = torch.arange(Tq, device=device).unsqueeze(1)  # [Tq,1]
        j = torch.arange(Tk, device=device).unsqueeze(0)  # [1,Tk]
        future = (j > i)
        mask = torch.zeros(Tq, Tk, device=device)
        return mask.masked_fill(future, float(soft_penalty if soft else float('-inf')))

    def forward(self, x, y):
        """
        x: [B, Tq, D] as query
        y: [B, Tk, D] as key/value
        返回：
          out_strict/out_soft: [B, Tq, D]
          attn_strict/attn_soft: [B, H, Tq, Tk]
        """
        B, Tq, D = x.shape
        Tk = y.size(1)

        strict_mask = self._causal_mask(Tq, Tk, x.device, soft=False)
        soft_mask   = self._causal_mask(Tq, Tk, x.device, soft=True, soft_penalty=self.soft_penalty)

        out_strict, attn_strict = self.mha(
            query=x, key=y, value=y,
            attn_mask=strict_mask, need_weights=True, average_attn_weights=False
        )  # attn_strict: [B,H,Tq,Tk]

        with torch.no_grad():
            out_soft, attn_soft = self.mha(
                query=x, key=y, value=y,
                attn_mask=soft_mask, need_weights=True, average_attn_weights=False
            )

        return out_strict, out_soft, attn_strict, attn_soft


# ---------------------------
# 位置编码 / 图像化
# ---------------------------
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, channels=7, height=32, width=32, temperature=100000.0):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.temperature = temperature
        self.register_buffer('pos_encoding', self._create_pos_encoding())

    def _create_pos_encoding(self):
        y_pos = torch.arange(self.height).float()
        x_pos = torch.arange(self.width).float()
        y_pos = y_pos / (self.height - 1) if self.height > 1 else y_pos
        x_pos = x_pos / (self.width - 1) if self.width > 1 else x_pos
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing='ij')
        half_dim = self.channels // 2
        dim_t = torch.arange(half_dim).float()
        dim_t = self.temperature ** (2 * dim_t / max(half_dim, 1))
        pos_x = grid_x.unsqueeze(-1) / dim_t
        pos_y = grid_y.unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        pos_encoding = pos_x + pos_y
        if pos_encoding.shape[-1] > self.channels:
            pos_encoding = pos_encoding[..., :self.channels]
        elif pos_encoding.shape[-1] < self.channels:
            padding = torch.zeros(*pos_encoding.shape[:-1], self.channels - pos_encoding.shape[-1])
            pos_encoding = torch.cat([pos_encoding, padding], dim=-1)
        pos_encoding = pos_encoding.permute(2, 0, 1).unsqueeze(0)
        return pos_encoding

    def forward(self, x):
        return x + self.pos_encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(100000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    B, seq_len, nvars = x_enc.shape
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    x_2d_intervel = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', p=periodicity)
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x_resized_2d_intervel = F.interpolate(x_2d_intervel, size=(image_size, image_size), mode='bilinear', align_corners=False)
    images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)
    images_intervel = einops.repeat(x_resized_2d_intervel, 'b 1 h w -> b c h w', c=3)
    video = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)
    images = video.mean(dim=1)
    images_intervel = einops.rearrange(images_intervel, '(b n) c h w -> b n c h w', b=B, n=nvars)
    images_intervel = images_intervel.mean(dim=1)
    return images, images_intervel


# ---------------------------
# 简单的时序/图像 MLP（保持你的结构）
# ---------------------------
class temporal(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed, hidden_size=1024):
        super().__init__()
        self.temporal_ = nn.Sequential(
            nn.Linear(embed, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed)
        )
    def forward(self, x):
        return x + self.temporal_(x)

class imagesorocess(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed, hidden_size=1024):
        super().__init__()
        self.images = nn.Sequential(
            nn.Linear(embed, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed)
        )
    def forward(self, x):
        return x + self.images(x)

class FusionModule(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed):
        super().__init__()
        self.model = temporal(seq_len, pred_len, enc_in, embed)
        self.modeli = imagesorocess(seq_len, pred_len, enc_in, embed)
        self.gate = nn.Sequential(
            nn.Linear(embed * 2, embed),
            nn.GELU(),
            nn.Linear(embed, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x, img):
        temporal_out = self.model(x)
        multimodal_features = self.modeli(img)
        fuse = torch.cat([temporal_out, multimodal_features], dim=-1)
        gate_weights_seq = self.gate(fuse)
        fused = gate_weights_seq[:, :, 0:1] * temporal_out + gate_weights_seq[:, :, 1:2] * multimodal_features
        return fused


# ---------------------------
# 主模型（实时记录 z 与三指标，自动落盘）
# ---------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.vlm_manager = VLMManager(configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.enc_in = configs.enc_in

        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        self.attn1 = FusionModule(self.seq_len, self.pred_len, self.enc_in, self.embed_size)

        self.V_embedding = nn.Sequential(
            nn.Conv2d(self.enc_in, self.enc_in, kernel_size=3, stride=3),
            nn.ReLU()
        )
        self.layernormV = nn.LayerNorm(self.embed_size)

        self.T_embedding = nn.Linear(self.seq_len, self.embed_size)

        self.I_embedding = nn.Sequential(
            nn.Conv2d(self.enc_in, self.enc_in, kernel_size=3, stride=3),
            nn.ReLU()
        )

        self.pos_emb = PositionalEncoding(d_model=self.embed_size)
        self.img_pos_emb = SinusoidalPositionEncoding()
        self.layernormI = nn.LayerNorm(self.embed_size)
        self.layernormT = nn.LayerNorm(self.embed_size)
        self.layernormT1 = nn.LayerNorm(self.embed_size)

        self.cross  = CausalMultiheadAttentionWrapper(
            embed_dim=self.embed_size, num_heads=8, dropout=0.4, soft_penalty=-4.0
        )
        self.crossv = CausalMultiheadAttentionWrapper(
            embed_dim=self.embed_size, num_heads=8, dropout=0.4, soft_penalty=-4.0
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size * 2, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size)
        )

        # —— 指标缓存 & 自动落盘控制 ——
        self._metrics_rows = []           # 累积每个 batch 的一行
        self.auto_log_metrics = True      # forward(..., return_metrics=True) 时自动记录
        self.auto_flush_excel_path = None # 若设置路径，则每次 forward 都覆写落盘

    # ====== Metrics logging helpers ======
    def enable_metric_logging(self, enabled: bool = True):
        self.auto_log_metrics = bool(enabled)

    def enable_auto_flush_excel(self, excel_path: str):
        """每次 forward 都把当前缓存写入到 excel_path（覆盖写）。"""
        self.auto_flush_excel_path = str(excel_path)

    def reset_metrics(self):
        self._metrics_rows = []

    def log_metrics(self, metrics: dict, **metadata):
        row = dict(metrics) if metrics is not None else {}
        for k, v in metadata.items():
            row[k] = v
        self._metrics_rows.append(row)

    def save_metrics_excel(self, excel_path: str = "coupling_analysis.xlsx",
                           compute_correlations: bool = True):
        if len(self._metrics_rows) == 0:
            print("⚠️ No metrics to save. _metrics_rows is empty.")
            return None, None

        df = pd.DataFrame(self._metrics_rows)

        corr_df = None
        if compute_correlations and ("mse" in df.columns):
            def _pearsonr(x, y, eps=1e-12):
                x = np.asarray(x); y = np.asarray(y)
                x = x - x.mean(); y = y - y.mean()
                denom = (np.linalg.norm(x) * np.linalg.norm(y) + eps)
                return float((x @ y) / denom)

            def _spearmanr(x, y):
                x = np.asarray(x); y = np.asarray(y)
                rx = x.argsort().argsort()
                ry = y.argsort().argsort()
                return _pearsonr(rx, ry)

            corrs = []
            def add_corr(name, x, y):
                p = _pearsonr(x, y)
                s = _spearmanr(x, y)
                corrs.append({"metric": name, "pearson": p, "spearman": s})

            if "delta_kl" in df.columns:
                add_corr("mse_vs_delta_kl", df["delta_kl"].values, df["mse"].values)
            if "delta_1mcka" in df.columns:
                add_corr("mse_vs_delta_1mcka", df["delta_1mcka"].values, df["mse"].values)
            if "delta_in" in df.columns:
                add_corr("mse_vs_delta_in", df["delta_in"].values, df["mse"].values)

            corr_df = pd.DataFrame(corrs)

        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="per_window", index=False)
                if corr_df is not None and len(corr_df) > 0:
                    corr_df.to_excel(writer, sheet_name="correlations", index=False)
            print(f"✅ Metrics exported to {excel_path}")
        except Exception as e:
            print(f"❌ Failed to save metrics to Excel: {e}")

        return df, corr_df

    # ====== 视觉辅助 ======
    def _normalize_images(self, images):
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        images = (images - min_vals) / scale
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        images, videos = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        images = self._normalize_images(images)
        videos = self._normalize_images(videos)
        return images, videos

    # ====== 前向：实时记录 z 与三指标；可自动落盘 ======
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, return_metrics=False, y_true=None, max_tokens=4096, **meta):

        # ---- RevIN 归一 ----
        z = self.revin_layer(x, 'norm')   # 需要“实时记录 z”
        B = z.shape[0]

        # ---- 视觉分支 ----
        images, videos = self.vision_augmented_learner(z, 96, self.seq_len, 96)
        images = images.float().to(z.device).sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)
        videos = videos.float().to(z.device).sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)

        videos_embeddings = self.V_embedding(videos)
        videos_embeddings = self.img_pos_emb(videos_embeddings)
        videos_embeddings = rearrange(videos_embeddings, 'B C W H->B C (W H)')
        videos_embeddings = self.layernormV(videos_embeddings)

        vision_embeddings = self.I_embedding(images)
        vision_embeddings = self.img_pos_emb(vision_embeddings)
        vision_embeddings = rearrange(vision_embeddings, 'B C W H->B C (W H)')
        vision_embeddings = self.layernormI(vision_embeddings)

        # ---- 双向因果注意力（严格 + 软因果）----
        H_strict, H_soft, A_gl_strict, A_gl_soft = self.cross(vision_embeddings, videos_embeddings)   # global <- local
        F_strict, F_soft, A_lg_strict, A_lg_soft = self.crossv(videos_embeddings, vision_embeddings) # local  <- global

        out = torch.cat([H_strict, F_strict], dim=-1)
        out_embeddings = self.mlp(out)   # 这就是你的 Z_fuse（融合表征）

        # ---- 时序分支 ----
        T = x.permute(0, 2, 1)         # [B, enc_in, seq_len]
        T = self.T_embedding(T)        # [B, enc_in, embed]
        T = self.pos_emb(T)
        T = self.layernormT(T)
        T = self.attn1(T, out_embeddings)
        T = self.layernormT1(T)
        T = self.fc1(T)                # [B, enc_in, pred_len]

        y_pred = T.permute(0, 2, 1)    # [B, pred_len, enc_in]
        y_pred = self.revin_layer(y_pred, 'denorm')

        # if not return_metrics:
        #     # 自动落盘（如已开启）
        #     if self.auto_flush_excel_path is not None and len(self._metrics_rows) > 0:
        #         self.save_metrics_excel(self.auto_flush_excel_path, compute_correlations=True)
        #     return y_pred

        # ---- 诊断指标（历史 vs 软因果未来）----
        P_hist = _row_normalize(A_gl_strict.mean(dim=1))
        Q_hist = _row_normalize(A_lg_strict.mean(dim=1).transpose(-1, -2))
        kl_hist = _kl_mean(P_hist, Q_hist)

        P_soft = _row_normalize(A_gl_soft.mean(dim=1))
        Q_soft = _row_normalize(A_lg_soft.mean(dim=1).transpose(-1, -2))
        kl_soft = _kl_mean(P_soft, Q_soft)

        # 展平为 [N,D]，并可下采样 token
        def _to_tokens_and_sample(T3):
            ND = _to_ND_tokens(T3)
            return _subsample_tokens(ND, max_tokens=max_tokens)

        Hs = _to_tokens_and_sample(H_strict)
        Fs = _to_tokens_and_sample(F_strict)
        Ht = _to_tokens_and_sample(H_soft)
        Ft = _to_tokens_and_sample(F_soft)

        cka_hist = _cka_linear(Hs, Fs)
        cka_soft = _cka_linear(Ht, Ft)
        in_hist  = _info_nce_proxy(Hs, Fs)
        in_soft  = _info_nce_proxy(Ht, Ft)

        mse = None
        if y_true is not None:
            mse = F.mse_loss(y_pred, y_true.to(y_pred.device)).item()

        # ---- “实时记录 z 与三指标” ----
        z_mean = z.mean().item()
        z_std  = z.std().item()

        metrics = {
            "z_mean": z_mean,
            "z_std":  z_std,

            # 原子指标
            "kl_hist": kl_hist,
            "kl_soft": kl_soft,
            "cka_hist": cka_hist,
            "cka_soft": cka_soft,
            "in_hist": in_hist,
            "in_soft": in_soft,


            "delta_kl": kl_hist - kl_soft,

            "delta_in": in_soft - in_hist,
        }
        if mse is not None:
            metrics["mse"] = mse

        # 写入内存表
        if self.auto_log_metrics:
            self.log_metrics(metrics, **meta)

        # 若开启自动落盘，则每一步都刷新 Excel
        if self.auto_flush_excel_path is not None:
            self.save_metrics_excel(self.auto_flush_excel_path, compute_correlations=True)

        # 返回预测与当前这一步的 metrics（便于外层也能拿到）
        return y_pred
