import torch
import torch.nn as nn
from layers.RevIN import RevIN
import numpy as np
import pandas as pd
import torch.nn.functional as F
import einops
from einops import rearrange
from layers.vlm_manager import VLMManager

import math

# -----------------------
# Utils for diagnostics
# -----------------------
def l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def info_nce(z1, z2, temperature=0.07):
    """
    z1, z2: [B, D]  -> returns scalar proxy (higher ~ larger MI)
    """
    z1 = l2_normalize(z1, dim=-1)
    z2 = l2_normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temperature    # [B,B]
    targets = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, targets)
    return -loss  # MI proxy: larger is better

def gram_linear(x):
    return x @ x.t()

def cka(x, y, eps=1e-6):
    # x,y: [B, D]
    K = gram_linear(x)
    L = gram_linear(y)
    hsic = (K * L).sum()
    norm = torch.sqrt((K * K).sum() * (L * L).sum()) + eps
    return (hsic / norm)

def flat_grad(y, x):
    (g,) = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)
    if g is None:
        return None

    return g.reshape(-1)

def grad_cosine(g1, g2, eps=1e-8):
    if (g1 is None) or (g2 is None):
        return None, None, None

    n1 = g1.norm() + eps
    n2 = g2.norm() + eps

    cos = (g1 @ g2 / (n1 * n2))
    return cos, n1, n2


# -----------------------
# Position encodings
# -----------------------
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


# -----------------------
# Visual mapping
# -----------------------
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


# -----------------------
# Causal MHA wrapper
# -----------------------
class CausalMultiheadAttentionWrapper(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 采用显式线性投影，确保注意力概率处在计算图中
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.drop = nn.Dropout(dropout)

    def create_causal_mask(self, Tq, Tk, device):
        # 上三角为 -inf，阻断未来
        mask = torch.triu(torch.ones(Tq, Tk, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))  # [Tq, Tk]
        return mask

    def _shape(self, x):
        # (B, T, D) -> (B, H, T, Dh)
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x

    def forward(self, x, y, return_attn=True):
        """
        x: query  [B, Tq, D]
        y: key/val [B, Tk, D]
        returns:
          out:    [B, Tq, D]
          attn_w: [B, H, Tq, Tk]  (requires_grad=True)
        """
        B, Tq, _ = x.shape
        Tk = y.size(1)

        device = x.device

        q = self._shape(self.q_proj(x))  # [B,H,Tq,Dh]
        k = self._shape(self.k_proj(y))  # [B,H,Tk,Dh]
        v = self._shape(self.v_proj(y))  # [B,H,Tk,Dh]

        # scaled dot-product logits
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,Tq,Tk]

        # causal mask（广播到批维/头维）
        causal_mask = self.create_causal_mask(Tq, Tk, device)                     # [Tq,Tk]
        logits = logits + causal_mask  # [B,H,Tq,Tk]

        # 注意：softmax 概率将携带梯度
        attn_probs = torch.softmax(logits, dim=-1)                                 # [B,H,Tq,Tk]
        attn_probs = self.drop(attn_probs)

        # 加权求和得到输出
        context = torch.matmul(attn_probs, v)                                      # [B,H,Tq,Dh]
        context = context.permute(0, 2, 1, 3).contiguous().view(B, Tq, self.embed_dim)  # [B,Tq,D]
        out = self.out_proj(context)                                               # [B,Tq,D]

        if return_attn:
            return out, attn_probs
        else:
            return out, None


# -----------------------
# Temporal / Image MLPs & Fusion
# -----------------------
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

    def forward(self, x, img, return_gate=True):
        B, n_vars, C = x.shape
        temporal_out = self.model(x)
        multimodal_features = self.modeli(img)
        fuse = torch.cat([temporal_out, multimodal_features], dim=-1)
        gate_weights_seq = self.gate(fuse)  # [B, n_vars, 2]
        fused_features_seq = (
            gate_weights_seq[:, :, 0:1] * temporal_out +
            gate_weights_seq[:, :, 1:2] * multimodal_features
        )
        if return_gate:
            return fused_features_seq, gate_weights_seq
        else:
            return fused_features_seq


# -----------------------
# Main Model with diagnostics
# -----------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.vlm_manager = VLMManager(configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.enc_in = configs.enc_in

        self.fc1 = nn.Sequential(
            nn.Sequential(nn.Linear(self.embed_size, self.pred_len),
                          nn.ReLU(),
                          nn.Linear(self.pred_len, self.pred_len),
                          nn.ReLU(),
                          nn.Dropout(0.6),
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

        self.cross = CausalMultiheadAttentionWrapper(embed_dim=self.embed_size, num_heads=8, dropout=0.5)
        self.crossv = CausalMultiheadAttentionWrapper(embed_dim=self.embed_size, num_heads=8, dropout=0.5)

        self.mlp = nn.Sequential(nn.Linear(self.embed_size * 2, self.embed_size),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_size, self.embed_size))

        # ---- diagnostics buffers ----
        self.metrics = {
            "step": [],
            "E_causal": [],
            "I_proxy": [],
            "CKA": [],
            "grad_ratio": [],
            "grad_cos": [],
            "gate_temporal": [],
            "gate_visual": []
        }
        self._global_step = 0
        self.log_diagnostics = True  # 需要时可关闭

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

    @torch.no_grad()
    def _record_scalar(self, key, val):
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

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x: [B, L, C] -> returns [B, pred_len, C]
        """
        self._global_step += 1
        device = x.device

        # RevIN normalize
        z = self.revin_layer(x, 'norm')
        x_norm = z  # [B, L, C]

        # Visionization
        images, videos = self.vision_augmented_learner(x_norm, 96, self.seq_len, 48)
        images = images.float().to(device)
        videos = videos.float().to(device)

        images = images.sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)
        videos = videos.sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)

        videos_embeddings = self.V_embedding(videos)                  # [B,C,32,32]
        videos_embeddings = self.img_pos_emb(videos_embeddings)
        videos_embeddings = rearrange(videos_embeddings, 'B C W H->B C (W H)')
        videos_embeddings = self.layernormV(videos_embeddings)        # [B,C,1024]

        vision_embeddings = self.I_embedding(images)
        vision_embeddings = self.img_pos_emb(vision_embeddings)
        vision_embeddings = rearrange(vision_embeddings, 'B C W H->B C (W H)')
        vision_embeddings = self.layernormI(vision_embeddings)        # [B,C,1024]

        # MHA 要求 [B, T, D]，这里 T = C(变量数)，D = 1024
        q1 = vision_embeddings
        k1 = v1 = videos_embeddings
        q2 = videos_embeddings
        k2 = v2 = vision_embeddings

        # g <- l
        out_g2l, A_g2l = self.cross(q1, k1)          # out: [B,C,D], A: [B,H,C,C]
        # l <- g
        out_l2g, A_l2g = self.crossv(q2, k2)         # out: [B,C,D], A: [B,H,C,C]

        # fuse two visual branches
        vis_fused = torch.cat([out_g2l, out_l2g], dim=-1)  # [B,C,2D]
        vis_fused = self.mlp(vis_fused)                    # [B,C,D]

        # Temporal path
        T = x_norm.permute(0, 2, 1)            # [B,C,L]
        T = self.T_embedding(T)                # [B,C,D]
        T = self.pos_emb(T)
        T = self.layernormT(T)

        # ACGF fusion (并返回门控)
        T_fused, gate_w = self.attn1(T, vis_fused, return_gate=True)  # gate_w: [B,C,2]
        T_fused = self.layernormT1(T_fused)
        Y = self.fc1(T_fused)                  # [B,C,pred_len]
        Y = Y.permute(0, 2, 1)                 # [B,pred_len,C]

        # RevIN denorm
        Y = self.revin_layer(Y, 'denorm')

        # ---------------------
        # Diagnostics logging
        # ---------------------
        if self.log_diagnostics and (self.training or True):
            B, C, D = out_g2l.shape
            # 因果能量（A_g2l 与 A_l2g^T 的差）
            A1 = A_g2l.mean(dim=1)  # [B,C,C]
            A2 = A_l2g.mean(dim=1)  # [B,C,C]
            E_causal = ((A1 - A2.transpose(-2, -1)) ** 2).mean()

            # MI 代理：用两路视觉输出的均值池化
            z_g = out_g2l.mean(dim=1)  # [B,D]
            z_l = out_l2g.mean(dim=1)  # [B,D]
            I_proxy = info_nce(z_g, z_l)

            # CKA（无梯度）
            with torch.no_grad():
                CKA_val = cka(z_g, z_l)

            # 梯度比例与夹角：对 I_proxy w.r.t A_g2l、对 E_causal w.r.t A_l2g
            # 使用 head-mean 的注意力以减轻显存
            gI_full = torch.autograd.grad(I_proxy, A_g2l, retain_graph=True, create_graph=False, allow_unused=False)[0]
            gE_full = torch.autograd.grad(E_causal, A_l2g, retain_graph=True, create_graph=False, allow_unused=False)[0]

            # 与你之前的统计口径保持一致：对 head 取均值，再展平做余弦
            gI = gI_full.mean(dim=1).reshape(-1)  # [B*Tq*Tk]
            gE = gE_full.mean(dim=1).reshape(-1)

            print(gI)
            cos_val, nI, nE = grad_cosine(gI, gE)

            # 门控权重（取 batch 和变量均值）
            with torch.no_grad():
                gate_temporal = gate_w[..., 0].mean()
                gate_visual = gate_w[..., 1].mean()

            # 记录
            self._record_scalar("step", self._global_step)
            self._record_scalar("E_causal", E_causal)
            self._record_scalar("I_proxy", I_proxy)
            self._record_scalar("CKA", CKA_val)
            if (nI is not None) and (nE is not None) and (cos_val is not None):
                self._record_scalar("grad_ratio", (nI / (nE + 1e-8)))
                self._record_scalar("grad_cos", cos_val)
            else:
                self._record_scalar("grad_ratio", 0.0)
                self._record_scalar("grad_cos", 0.0)
            self._record_scalar("gate_temporal", gate_temporal)
            self._record_scalar("gate_visual", gate_visual)


        return Y
