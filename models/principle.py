import torch
import torch.nn as nn
from layers.RevIN import RevIN
import numpy as np
from einops import rearrange
import pandas as pd
import torch.fft
import torch.nn.functional as F
import einops
from layers.vlm_manager import VLMManager
from einops import rearrange

import math

class CausalMultiheadAttentionWrapper(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def create_causal_mask(sefl, seq_len, device):
        """创建因果注意力掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x,y):
        seq_len = x.size(1)
        causal_mask = self.create_causal_mask(862, x.device)

        output, _ = self.mha(
            query=x,
            key=y,
            value=y,
            attn_mask=causal_mask,
            need_weights=False
        )
        return output
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, channels=862, height=32, width=32, temperature=100000.0):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.temperature = temperature

        # 预计算位置编码
        self.register_buffer('pos_encoding', self._create_pos_encoding())

    def _create_pos_encoding(self):
        # 创建网格坐标
        y_pos = torch.arange(self.height).float()
        x_pos = torch.arange(self.width).float()

        # 归一化到 [0, 1]
        y_pos = y_pos / (self.height - 1) if self.height > 1 else y_pos
        x_pos = x_pos / (self.width - 1) if self.width > 1 else x_pos

        # 扩展到2D网格
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing='ij')

        # 对于奇数通道数，调整维度计算
        half_dim = self.channels // 2
        dim_t = torch.arange(half_dim).float()
        dim_t = self.temperature ** (2 * dim_t / max(half_dim, 1))

        pos_x = grid_x.unsqueeze(-1) / dim_t
        pos_y = grid_y.unsqueeze(-1) / dim_t

        # 正弦和余弦编码
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)

        # 合并x和y的位置编码
        pos_encoding = pos_x + pos_y

        # 如果通道数是奇数，需要截断或填充
        if pos_encoding.shape[-1] > self.channels:
            pos_encoding = pos_encoding[..., :self.channels]
        elif pos_encoding.shape[-1] < self.channels:
            # 用零填充
            padding = torch.zeros(*pos_encoding.shape[:-1], self.channels - pos_encoding.shape[-1])
            pos_encoding = torch.cat([pos_encoding, padding], dim=-1)

        # 调整形状为 [1, C, H, W]
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
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')

    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    x_2d_intervel = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', p=periodicity)

    # Resize the time series data
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x_resized_2d_intervel = F.interpolate(x_2d_intervel, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Convert to 3-channel image
    global_images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)  # [B * nvars, 3, H, W]
    local_images = einops.repeat(x_resized_2d_intervel, 'b 1 h w -> b c h w', c=3)

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    g = einops.rearrange(global_images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    global_images = g.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    # videos = video.mean(dim=2)

    l = einops.rearrange(local_images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    local_images = l.mean(dim=1)  # Average over nvars to get [B, 3, H, W]

    return global_images, local_images
class temporal(nn.Module):
    """时间模式建模模块"""

    def __init__(self, seq_len, pred_len, enc_in, embed, hidden_size=1024):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.embed_size = embed

        # 时间卷积
        self.temporal_ = nn.Sequential(
            nn.Linear(self.embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.embed_size)
        )

    def forward(self, x):

        temp = self.temporal_(x)
        y = x+temp

        return y

class imagesorocess(nn.Module):
    """时间模式建模模块"""

    def __init__(self, seq_len, pred_len, enc_in, embed, hidden_size=1024):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.embed_size = embed

        # 时间卷积
        self.images = nn.Sequential(
            nn.Linear(self.embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.embed_size)
        )


    def forward(self, x):

        img = self.images(x)
        y = x + img

        return y

class AdaptiveGateFusionModule(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.embed_size = embed

        # self.cross = CrossAttentionFusion(self.embed_size)
        self.model = temporal(seq_len, pred_len, enc_in, embed)

        self.modeli = imagesorocess(seq_len, pred_len, enc_in, embed)
        self.modelv = imagesorocess(seq_len, pred_len, enc_in, embed)

        self.gate = nn.Sequential(
            nn.Linear(self.embed_size * 2, self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size, 2),
            nn.Softmax(dim=-1)
        )


    def forward(self, x, img):
        # x: [batch_size, seq_len, enc_in]
        B,n_vars,C = x.shape
        temporal_out = self.model(x)
        multimodal_features = self.modeli(img)

        fuse = torch.cat([temporal_out,multimodal_features],dim=-1)

        gate_weights_seq = self.gate(fuse)

        fused_features_seq = (
                gate_weights_seq[:, :, 0:1] * temporal_out +
                gate_weights_seq[:, :, 1:2] * multimodal_features
        )

        fused_features = fused_features_seq

        return fused_features

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

        # self.fc1 = nn.Sequential(
        #     nn.Linear(self.embed_size, self.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size, self.pred_len)
        # )
        self.fc1 = nn.Sequential(
            nn.Sequential(nn.Linear(self.embed_size, self.hidden_size),
                          nn.ReLU(),
                          nn.Linear(self.hidden_size, self.hidden_size),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(self.hidden_size, self.pred_len))
        )

        self.attn1 = AdaptiveGateFusionModule(self.seq_len, self.pred_len, self.enc_in, self.embed_size)

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

        # self.cross = nn.MultiheadAttention(embed_dim=1024, num_heads=8, dropout=0.5)
        self.cross = CausalMultiheadAttentionWrapper()
        self.crossv = CausalMultiheadAttentionWrapper()

        self.mlp = nn.Sequential(nn.Linear(self.embed_size*2, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size))

    def _normalize_images(self,images):

        # Compute min and max per image across all channels and spatial dimensions
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        # Normalize to [0, 1]
        images = (images - min_vals) / scale
        # Scale to [0, 255] and clamp to ensure valid range
        images = (images * 255).clamp(0, 255).to(torch.uint8)

        return images
    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        global_images, local_images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        global_images = self._normalize_images(global_images)
        local_images = self._normalize_images(local_images)
        return global_images, local_images
    def structure_vision_representation(self,z):
        global_images, local_images = self.vision_augmented_learner(z, 96, self.seq_len, 96)
        local_images = local_images.float().to('cuda:0')
        local_images = local_images.sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)
        local_images = self.V_embedding(local_images)
        local_embeddings = self.img_pos_emb(local_images)
        local_embeddings = rearrange(local_embeddings, 'B C W H->B C (W H)')
        local_embeddings = self.layernormV(local_embeddings)

        global_images = global_images.float().to('cuda:0')
        global_images = global_images.sum(1).unsqueeze(1).expand(-1, self.enc_in, -1, -1)
        global_embeddings = self.I_embedding(global_images)
        global_embeddings = self.img_pos_emb(global_embeddings)
        global_embeddings = rearrange(global_embeddings, 'B C W H->B C (W H)')
        global_embeddings = self.layernormI(global_embeddings)

        return global_embeddings, local_embeddings

    def casuality_aware_cross_structure(self,global_embeddings, local_embeddings):
        global_local_embedding = self.cross(global_embeddings, local_embeddings)
        embeddings = self.crossv(local_embeddings, global_local_embedding)

        out = torch.cat([global_local_embedding, embeddings], dim=-1)
        out_embeddings = self.mlp(out)

        return out_embeddings

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):

        z = x
        z = self.revin_layer(z, 'norm')
        x = z
        B = z.shape[0]

        global_embeddings, local_embeddings = self.structure_vision_representation(z)

        out_embeddings = self.casuality_aware_cross_structure(global_embeddings, local_embeddings)

        T = x
        T = T.permute(0, 2, 1)

        T = self.T_embedding(T)  # [batch_size, enc_in, embed_size]
        T = self.pos_emb(T)
        T = self.layernormT(T)

        T = self.attn1(T, out_embeddings)

        T = self.layernormT1(T)
        T = self.fc1(T)  # [batch_size, enc_in, pred_len]

        x = T.permute(0, 2, 1)  # [batch_size, pred_len, enc_in]
        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x