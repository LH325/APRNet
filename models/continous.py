# -*- coding: utf-8 -*-
"""
Topology Visualization & Metrics for FusionModule
=================================================
This script provides:
1) Minimal patch to your FusionModule to expose 2D features for visualization.
2) Metrics: Total Variation (TV ↓), Algebraic Connectivity λ₂ (↑), Moran's I (↑).
3) A single-figure visualization: nodes (colored by 1st PC), kNN edges, and temporal path.
4) Simple entry points to run on your data and save PNG/PDF.

How to use (baseline vs ours):
------------------------------
# Load your baseline checkpoint into `model_base`, and ours into `model_ours`.
visualize_topology(model_base, x, img, H, W, save_png="baseline_fused.png", save_pdf="baseline_fused.pdf")
visualize_topology(model_ours,  x, img, H, W, save_png="ours_fused.png",     save_pdf="ours_fused.pdf")

Then put the two figures side-by-side in LaTeX.

Dependencies: torch, numpy, matplotlib
Optionally: pandas (to save metrics CSV); remove if not needed.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

# =========================
# (A) Minimal patch to FusionModule to expose features
# =========================
class FusionModule(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, embed, num_experts=1, top_k=1, H=None, W=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.embed_size = embed
        self.num_experts = num_experts
        self.top_k = top_k
        # Provide grid size for 2D visualization (SVR grid). If your imagesorocess already returns [B,H,W,D], H/W can be None.
        self.H, self.W = H, W

        # ---- your original components ----
        self.model = temporal(seq_len, pred_len, enc_in, embed)        # -> [B, L, D]
        self.modeli = imagesorocess(seq_len, pred_len, enc_in, embed)  # -> [B, L, D] or [B, H, W, D]

        self.gate = nn.Sequential(
            nn.Linear(self.embed_size * 2, self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size, num_experts),
            nn.Softmax(dim=-1)
        )

        self.experts = nn.Sequential(
            nn.Linear(self.embed_size * 2, self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size, self.embed_size)
        )

    def forward(self, x, img, return_vis: bool = False):
        """
        x:   [B, seq_len, enc_in]
        img: your visual input
        return_vis=True -> returns fused and a dict of intermediate tensors for visualization.
        """
        temporal_out = self.model(x)            # [B, L, D]
        multimodal_features = self.modeli(img)  # [B, L, D] or [B, H, W, D]

        if multimodal_features.dim() == 4:
            mm_2d = multimodal_features                         # [B,H,W,D]
            mm_seq = multimodal_features.view(multimodal_features.size(0), -1, multimodal_features.size(-1))  # [B,L,D]
            H, W = mm_2d.size(1), mm_2d.size(2)
            if self.H is None: self.H = H
            if self.W is None: self.W = W
        else:
            mm_seq = multimodal_features                         # [B,L,D]
            assert self.H is not None and self.W is not None, \
                "Please set model.H, model.W so we can reshape SVR features to [B,H,W,D] for visualization."
            assert self.H * self.W == mm_seq.size(1), "H*W must equal L for reshaping to 2D."
            mm_2d = mm_seq.view(mm_seq.size(0), self.H, self.W, mm_seq.size(-1))

        fuse_input = torch.cat([temporal_out, mm_seq], dim=-1)  # [B,L,2D]
        gate_logits = self.gate(fuse_input)                     # [B,L,E]
        expert_output = self.experts(fuse_input)                # [B,L,D]
        fused_features = expert_output * gate_logits[..., :1]   # [B,L,D], simple broadcast

        if return_vis:
            fused_2d = fused_features.view(fused_features.size(0), self.H, self.W, fused_features.size(-1))
            return fused_features, {
                "temporal_out": temporal_out.detach(),
                "mm_seq": mm_seq.detach(),
                "mm_2d": mm_2d.detach(),
                "fused_seq": fused_features.detach(),
                "fused_2d": fused_2d.detach()
            }
        return fused_features


# =========================
# (B) Topology metrics (TV, λ2, Moran's I)
# =========================
def _neighbors4(H: int, W: int, r: int, c: int):
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            yield rr, cc

def topo_tv_2d(feat_2d: torch.Tensor) -> torch.Tensor:
    """
    feat_2d: [B, H, W, D]
    Returns: scalar tensor (lower is smoother/continuous).
    """
    assert feat_2d.dim() == 4, "feat_2d must be [B,H,W,D]"
    dh = (feat_2d[:, 1:, :, :] - feat_2d[:, :-1, :, :]).pow(2).mean()
    dw = (feat_2d[:, :, 1:, :] - feat_2d[:, :, :-1, :]).pow(2).mean()
    return 0.5 * (dh + dw)

def _build_grid_L(H: int, W: int, device) -> torch.Tensor:
    N = H * W
    A = torch.zeros(N, N, device=device)
    def idx(r, c): return r * W + c
    for r in range(H):
        for c in range(W):
            u = idx(r, c)
            for rr, cc in _neighbors4(H, W, r, c):
                v = idx(rr, cc)
                A[u, v] = 1.0
                A[v, u] = 1.0
    D = torch.diag(A.sum(-1))
    return D - A  # Laplacian L

def topo_lambda2(H: int, W: int, device) -> torch.Tensor:
    """
    Algebraic connectivity (second-smallest eigenvalue of Laplacian).
    Higher => better connectivity.
    """
    L = _build_grid_L(H, W, device)
    # Using symmetric eigendecomp
    lam, _ = torch.linalg.eigh(L)
    lam_sorted, _ = torch.sort(lam)
    return lam_sorted[1]  # λ2

def morans_I(feat_2d: torch.Tensor) -> torch.Tensor:
    """
    Moran's I (spatial autocorrelation), higher => more continuous.
    feat_2d: [B, H, W, D]
    """
    B, H, W, D = feat_2d.shape
    device = feat_2d.device

    # Build Laplacian once, get A = D - L
    L = _build_grid_L(H, W, device)
    A = torch.diag(L.diag()) - L
    Wsum = A.sum()

    I_list = []
    X = feat_2d.view(B, H * W, D)                    # [B,N,D]
    Xc = X - X.mean(dim=1, keepdim=True)             # center per-batch
    for b in range(B):
        # First principal direction
        U, S, Vh = torch.linalg.svd(Xc[b], full_matrices=False)  # [N,D]=[N,r]@[r,r]@[r,D]
        z = X[b] @ Vh[0]                                         # [N]
        zbar = z.mean()
        num = (z - zbar).unsqueeze(0) @ A @ (z - zbar).unsqueeze(-1)
        den = ((z - zbar) ** 2).sum()
        I = (H * W / (Wsum + 1e-12)) * (num.squeeze() / (den + 1e-12))
        I_list.append(I)
    return torch.stack(I_list).mean()


# =========================
# (C) Visualization (single figure)
# =========================
def _plot_single_figure(feat_2d_np: np.ndarray, title: str, save_png: Optional[str] = None, save_pdf: Optional[str] = None):
    """
    feat_2d_np: [H, W, D] numpy array
    Single chart: scatter colored by 1st PC + kNN edges (cosine) + temporal path (raster).
    """
    H, W, D = feat_2d_np.shape
    # Grid coords in [0,1]
    ys, xs = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    coords = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=1)  # [N,2]
    F = feat_2d_np.reshape(-1, D)

    # 1st principal component for color
    Fc = F - F.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    scalar = F @ Vt[0]  # [N]

    # Cosine kNN edges
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    cos = Fn @ Fn.T
    np.fill_diagonal(cos, 0.0)
    k = 4
    edges = []
    for i in range(F.shape[0]):
        nbrs = np.argsort(-cos[i])[:k]
        for j in nbrs:
            edges.append((i, j, float(cos[i, j])))

    # Temporal path: raster order
    order = np.arange(F.shape[0])

    # Draw a SINGLE figure (no subplots)
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    sc = ax.scatter(coords[:, 1], coords[:, 0], s=10, c=scalar, alpha=0.95)
    for i, j, w in edges:
        x1, y1 = coords[i, 1], coords[i, 0]
        x2, y2 = coords[j, 1], coords[j, 0]
        ax.plot([x1, x2], [y1, y2], alpha=0.03 + 0.25 * w)
    ax.plot(coords[order, 1], coords[order, 0], linewidth=0.6, alpha=0.45)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Projected feature (1st PC)")
    plt.tight_layout()
    if save_png: plt.savefig(save_png, dpi=300, bbox_inches="tight")
    if save_pdf: plt.savefig(save_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# (D) End-to-end helper: run model -> get fused 2D -> metrics -> figure
# =========================
@torch.no_grad()
def visualize_topology(model: nn.Module,
                       x: torch.Tensor,
                       img: torch.Tensor,
                       H: Optional[int],
                       W: Optional[int],
                       save_png: Optional[str] = "topology_fused.png",
                       save_pdf: Optional[str] = "topology_fused.pdf",
                       return_metrics: bool = True) -> Dict[str, float]:
    """
    Runs the model with return_vis=True, extracts fused_2d features of one sample,
    computes TV / λ2 / Moran's I, and saves a single-figure visualization.

    Args:
        model: your FusionModule (patched), with model.H, model.W set or pass H, W here.
        x:   [B, seq_len, enc_in]
        img: your visual input
        H,W: grid size for SVR 2D mapping (if model.H/W not already set)
    """
    # If model doesn't have H/W, set them
    if getattr(model, "H", None) is None and H is not None:
        model.H = H
    if getattr(model, "W", None) is None and W is not None:
        model.W = W

    fused, vis = model(x, img, return_vis=True)
    fused_2d = vis["fused_2d"]  # [B, H, W, D]
    assert fused_2d is not None, "fused_2d is None. Please ensure H,W are set and shapes are consistent."

    # Choose one sample to visualize (b=0)
    fused_2d_one = fused_2d[0]

    # Metrics (batch-wise)
    tv_val = topo_tv_2d(fused_2d).item()
    lam2_val = topo_lambda2(model.H, model.W, device=fused_2d.device).item()
    mi_val = morans_I(fused_2d).item()

    # Figure
    _plot_single_figure(fused_2d_one.cpu().numpy(),
                        title="Topology-Preserving Fusion",
                        save_png=save_png, save_pdf=save_pdf)

    metrics = {"TV": tv_val, "lambda2": lam2_val, "MoransI": mi_val}
    print(f"[Topology Metrics] TV ↓ = {tv_val:.6f} | λ₂ ↑ = {lam2_val:.6f} | Moran's I ↑ = {mi_val:.6f}")
    return metrics if return_metrics else {}


# =========================
# (E) Example main (replace with your dataloader/checkpoints)
# =========================
if __name__ == "__main__":
    """
    示例仅用于说明流程。你需要用真实的:
    - temporal / imagesorocess 模块实现
    - 加载你训练好的 checkpoint
    - 替换 x, img 为真实 batch
    - 指定 H, W（若 imagesorocess 直接输出 [B,H,W,D]，可不显式传）
    """

    # ---- dummy deps to avoid NameError in this template ----
    class temporal(nn.Module):
        def __init__(self, seq_len, pred_len, enc_in, embed):
            super().__init__()
            self.out = nn.Linear(enc_in, embed)
        def forward(self, x):  # x: [B, L, C]
            return self.out(x)  # [B, L, D]

    class imagesorocess(nn.Module):
        def __init__(self, seq_len, pred_len, enc_in, embed):
            super().__init__()
            self.out = nn.Linear(enc_in, embed)
        def forward(self, img):  # Suppose img is [B, L, C] or [B, H, W, C] flattened externally
            # Here we mock a [B, L, D] output with simple linear
            if img.dim() == 4:
                B, H, W, C = img.shape
                feat = self.out(img.view(B, H*W, C))
                return feat.view(B, H, W, -1)  # simulate [B,H,W,D]
            else:
                return self.out(img)

    # ---- create a mock model (replace with your trained one) ----
    seq_len, pred_len, enc_in, embed = 96, 96, 7, 1024
    # If imagesorocess returns [B,H,W,D], H/W can be inferred; otherwise set H,W explicitly.
    H, W = 32, 32

    model = FusionModule(seq_len, pred_len, enc_in, embed, H=H, W=W)
    model.eval()

    # ---- fake data (replace with real batch from your loader) ----
    B = 2
    x = torch.randn(B, seq_len, enc_in)     # [B,L,C]
    # pretend the visual input is an "image grid" mapping to HxW tokens, enc_in channels
    img = torch.randn(B, H, W, enc_in)      # [B,H,W,C] -> imagesorocess returns [B,H,W,D]

    # ---- run visualization ----
    _ = visualize_topology(model, x, img, H, W,
                           save_png="ours_fused.png",
                           save_pdf="ours_fused.pdf")

    # If you have a baseline checkpoint/model:
    # baseline_model = FusionModule(seq_len, pred_len, enc_in, embed, H=H, W=W)
    # baseline_model.load_state_dict(torch.load("baseline.ckpt"))
    # baseline_model.eval()
    # _ = visualize_topology(baseline_model, x, img, H, W,
    #                        save_png="baseline_fused.png",
    #                        save_pdf="baseline_fused.pdf")
