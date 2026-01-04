import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import einops
from einops import rearrange
from layers.RevIN import RevIN
from layers.vlm_manager import VLMManager

# =============== Utils: metrics & helpers ===============

def l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def info_nce(z1, z2, temperature=0.07):
    """
    z1, z2: [B, D] -> scalar proxy (higher ~ larger MI)
    """
    z1 = l2_normalize(z1, dim=-1)
    z2 = l2_normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temperature  # [B,B]
    targets = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, targets)
    return loss  # MI proxy

def gram_linear(x):
    return x @ x.t()

@torch.no_grad()
def cka(x, y, eps=1e-6):
    # x,y: [B, D]
    K = gram_linear(x)
    L = gram_linear(y)
    hsic = (K * L).sum()
    norm = torch.sqrt((K * K).sum() * (L * L).sum()) + eps
    return (hsic / norm)

def upper_mask(Tq, Tk, device):
    # 上三角（未来，j>i）=1
    i = torch.arange(Tq, device=device).unsqueeze(1)
    j = torch.arange(Tk, device=device).unsqueeze(0)
    return (j > i).float()  # [Tq, Tk]

def lower_mask(Tq, Tk, device):
    # 下三角（含对角线，历史，j<=i）=1
    i = torch.arange(Tq, device=device).unsqueeze(1)
    j = torch.arange(Tk, device=device).unsqueeze(0)
    return (j <= i).float()  # [Tq, Tk]

def masked_row_renorm(P, M, eps=1e-8):
    """
    P: [B,Tq,Tk]  (已按 head 平均)
    M: [Tq,Tk]    (0/1 mask)
    -> 仅在 M==1 的子域做行内归一化，得到概率矩阵；M==0 位置清零
    """
    Pm = P * M  # [B,Tq,Tk]
    denom = Pm.sum(dim=-1, keepdim=True).clamp_min(eps)
    Pn = (Pm / denom) * M
    return Pn

def causal_energy(P, Q_T, M):
    """
    因果能量：masked Frobenius discrepancy
    P:   [B,Tq,Tk]
    Q_T: [B,Tq,Tk]  (Q^T 已与 P 对齐)
    M:   [Tq,Tk]    (0/1 mask)
    """
    D = ((P - Q_T) ** 2) * M  # [B,Tq,Tk]
    return D.sum(dim=(-1, -2)).mean()

def attn_weighted_output(P, V):  # P:[B,Tq,Tk], V:[B,Tk,D] -> [B,Tq,D]
    return torch.matmul(P, V)

def time_pool(z):  # [B,T,D] -> [B,D]
    return z.mean(dim=1)

# =============== Positional Encodings ===============

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

        pos_encoding = pos_encoding.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
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

# =============== Visual mapping ===============

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

# =============== Causal MHA (strict & soft) ===============

class CausalMultiheadAttentionWrapper(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.5, soft_penalty=-4.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.soft_penalty = soft_penalty
        self.last_attn_strict = None  # [B,H,Tq,Tk]
        self.last_attn_soft = None    # [B,H,Tq,Tk]

    def create_causal_mask(self, Tq, Tk, device, soft=False):
        mask = torch.zeros(Tq, Tk, device=device)
        i = torch.arange(Tq, device=device).unsqueeze(1)
        j = torch.arange(Tk, device=device).unsqueeze(0)
        future = (j > i)
        if soft:
            mask = mask.masked_fill(future, float(self.soft_penalty))
        else:
            mask = mask.masked_fill(future, float('-inf'))
        return mask  # [Tq,Tk]

    def forward(self, x, y):
        """
        x: [B,Tq,D] as query
        y: [B,Tk,D] as key/value
        Returns:
          out_strict: [B,Tq,D]   (严格因果输出)
          A_strict:   [B,H,Tq,Tk]
          A_soft:     [B,H,Tq,Tk] (仅验证)
        """
        Tq, Tk = x.size(1), y.size(1)
        strict_mask = self.create_causal_mask(Tq, Tk, x.device, soft=False)
        soft_mask   = self.create_causal_mask(Tq, Tk, x.device, soft=True)

        out, A_strict = self.mha(
            x, y, y,
            attn_mask=strict_mask,
            need_weights=True,
            average_attn_weights=False
        )

        with torch.no_grad():
            _, A_soft = self.mha(
                x, y, y,
                attn_mask=soft_mask,
                need_weights=True,
                average_attn_weights=False
            )

        self.last_attn_strict = A_strict
        self.last_attn_soft = A_soft
        return out, A_strict, A_soft

# =============== Light blocks ===============

class temporal(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed, hidden_size=1024):
        super().__init__()
        self.embed_size = embed
        self.temporal_ = nn.Sequential(
            nn.Linear(self.embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.embed_size)
        )

    def forward(self, x):
        return x + self.temporal_(x)

class imagesorocess(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed, hidden_size=1024):
        super().__init__()
        self.embed_size = embed
        self.images = nn.Sequential(
            nn.Linear(self.embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.embed_size)
        )

    def forward(self, x):
        return x + self.images(x)

class FusionModule(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed):
        super().__init__()
        self.embed_size = embed
        self.model = temporal(seq_len, pred_len, enc_in, embed)
        self.modeli = imagesorocess(seq_len, pred_len, enc_in, embed)
        self.modelv = imagesorocess(seq_len, pred_len, enc_in, embed)
        self.gate = nn.Sequential(
            nn.Linear(self.embed_size * 2, self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, img, return_gate=False):
        B, n_vars, C = x.shape
        temporal_out = self.model(x)
        multimodal_features = self.modeli(img)
        fuse = torch.cat([temporal_out, multimodal_features], dim=-1)
        gate_weights_seq = self.gate(fuse)  # [B,n_vars,2]
        fused = gate_weights_seq[..., 0:1] * temporal_out + gate_weights_seq[..., 1:2] * multimodal_features
        if return_gate:
            return fused, gate_weights_seq
        return fused

# =============== Main Model with six diagnostics ===============

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
            nn.Sequential(nn.Linear(self.embed_size, self.pred_len),
                          nn.ReLU(),
                          nn.Linear(self.pred_len, self.pred_len),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(self.pred_len, self.pred_len))
        )

        self.attn1 = FusionModule(self.seq_len, self.pred_len, self.enc_in, self.embed_size)

        self.V_embedding = nn.Sequential(nn.Conv2d(self.enc_in, self.enc_in, kernel_size=3, stride=3),
                                         nn.ReLU())
        self.layernormV = nn.LayerNorm(self.embed_size)

        self.T_embedding = nn.Linear(self.seq_len, self.embed_size)

        self.I_embedding = nn.Sequential(nn.Conv2d(self.enc_in, self.enc_in, kernel_size=3, stride=3),
                                         nn.ReLU())
        self.pos_emb = PositionalEncoding(d_model=self.embed_size)
        self.img_pos_emb = SinusoidalPositionEncoding()

        self.layernormI = nn.LayerNorm(self.embed_size)
        self.layernormT = nn.LayerNorm(self.embed_size)
        self.layernormT1 = nn.LayerNorm(self.embed_size)

        self.cross  = CausalMultiheadAttentionWrapper(embed_dim=self.embed_size, num_heads=8, dropout=0.5)
        self.crossv = CausalMultiheadAttentionWrapper(embed_dim=self.embed_size, num_heads=8, dropout=0.5)

        self.mlp = nn.Sequential(nn.Linear(self.embed_size * 2, self.embed_size),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_size, self.embed_size))

        # metrics buffers
        self.metrics = {
            "step": [],
            "E_hist": [], "E_fut": [],
            "MI_hist": [], "MI_fut": [],
            "CKA_hist": [], "CKA_fut": []
        }
        self._global_step = 0
        self.log_diagnostics = True

    @torch.no_grad()
    def _record(self, key, val):
        if key in self.metrics:
            if torch.is_tensor(val):
                val = val.detach().float().cpu().item()
            self.metrics[key].append(float(val))

    def save_metrics_to_excel(self, path: str, sheet_name: str = 'metrics'):
        df = pd.DataFrame(self.metrics)
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    def reset_metrics(self):
        for k in self.metrics:
            self.metrics[k] = []
        self._global_step = 0

    def _normalize_images(self, images):
        B = images.size(0)
        flat = images.view(B, -1)
        min_vals = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
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

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x: [B, L, C] -> returns [B, pred_len, C]
        """
        self._global_step += 1
        device = x.device

        # RevIN normalize
        z = self.revin_layer(x, 'norm')
        x_norm = z  # [B,L,C]

        # Visionization
        images, videos = self.vision_augmented_learner(x_norm, 96, self.seq_len, 24)
        images = images.float().to(device)
        videos = videos.float().to(device)

        # Build two visual streams
        images = images.sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)
        videos = videos.sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)

        videos_embeddings0 = self.V_embedding(videos)                 # [B,C,W,H]
        videos_embeddings0 = self.img_pos_emb(videos_embeddings0)
        videos_embeddings0 = rearrange(videos_embeddings0, 'B C W H->B C (W H)')
        videos_embeddings0 = self.layernormV(videos_embeddings0)      # [B,C,D]

        vision_embeddings0 = self.I_embedding(images)
        vision_embeddings0 = self.img_pos_emb(vision_embeddings0)
        vision_embeddings0 = rearrange(vision_embeddings0, 'B C W H->B C (W H)')
        vision_embeddings0 = self.layernormI(vision_embeddings0)      # [B,C,D]

        # Keep copies as K/V for masked outputs
        V_local  = videos_embeddings0    # 被视作 local 分支的 V
        V_global = vision_embeddings0    # 被视作 global 分支的 V

        # Cross attentions
        # global <- local
        out_gl, A_gl_strict, A_gl_soft = self.cross(vision_embeddings0, videos_embeddings0)
        # local  <- global
        out_lg, A_lg_strict, A_lg_soft = self.crossv(videos_embeddings0, vision_embeddings0)

        # fuse two visual branches
        vis_fused = torch.cat([out_gl, out_lg], dim=-1)  # [B,C,2D]
        vis_fused = self.mlp(vis_fused)                  # [B,C,D]

        # Temporal path
        T = x_norm.permute(0, 2, 1)            # [B,C,L]
        T = self.T_embedding(T)                # [B,C,D]
        T = self.pos_emb(T)
        T = self.layernormT(T)

        T_fused, gate_w = self.attn1(T, vis_fused, return_gate=True)
        T_fused = self.layernormT1(T_fused)
        Y = self.fc1(T_fused)                  # [B,C,pred_len]
        Y = Y.permute(0, 2, 1)                 # [B,pred_len,C]
        Y = self.revin_layer(Y, 'denorm')

        # ================== Six Diagnostics ==================
        if self.log_diagnostics:
            # head-mean attentions
            A1_s = A_gl_strict.mean(dim=1)  # [B,Tq,Tk] (global<-local)
            A2_s = A_lg_strict.mean(dim=1)  # [B,Tq,Tk] (local <-global)
            A1_soft = A_gl_soft.mean(dim=1)
            A2_soft = A_lg_soft.mean(dim=1)

            B, Tq, Tk = A1_s.shape
            M_hist = lower_mask(Tq, Tk, device)   # j<=i
            M_fut  = upper_mask(Tq, Tk, device)   # j>i

            # ---- Historical domain (strict, lower-tri) ----
            P_hist_gl = masked_row_renorm(A1_s, M_hist)                # global<-local
            P_hist_lg = masked_row_renorm(A2_s, M_hist)                # local <-global
            # align shapes for energy: compare P_hist_gl with P_hist_lg^T
            P_hist_lg_T = P_hist_lg.transpose(-2, -1)

            E_hist = causal_energy(P_hist_gl, P_hist_lg_T, M_hist)

            # masked outputs for MI/CKA
            Zg_hist = attn_weighted_output(P_hist_gl, V_local)         # use local values
            Zl_hist = attn_weighted_output(P_hist_lg, V_global)        # use global values
            zg_hist = time_pool(Zg_hist)   # [B,D]
            zl_hist = time_pool(Zl_hist)

            MI_hist = info_nce(zg_hist, zl_hist)
            with torch.no_grad():
                CKA_hist = cka(zg_hist, zl_hist)

            # ---- Future domain (soft, upper-tri) ----
            P_fut_gl = masked_row_renorm(A1_soft, M_fut)               # global<-local (future region)
            P_fut_lg = masked_row_renorm(A2_soft, M_fut)               # local <-global
            P_fut_lg_T = P_fut_lg.transpose(-2, -1)

            E_fut = causal_energy(P_fut_gl, P_fut_lg_T, M_fut)

            Zg_fut = attn_weighted_output(P_fut_gl, V_local)
            Zl_fut = attn_weighted_output(P_fut_lg, V_global)
            zg_fut = time_pool(Zg_fut)
            zl_fut = time_pool(Zl_fut)

            MI_fut = info_nce(zg_fut, zl_fut)
            with torch.no_grad():
                CKA_fut = cka(zg_fut, zl_fut)

            # ---- record ----
            self._record("step", self._global_step)
            self._record("E_hist", E_hist)
            self._record("E_fut",  E_fut)
            self._record("MI_hist", MI_hist)
            self._record("MI_fut",  MI_fut)
            self._record("CKA_hist", CKA_hist)
            self._record("CKA_fut",  CKA_fut)

        return Y
