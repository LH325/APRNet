from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
warnings.filterwarnings('ignore')
# from models.APNet import collect_feats_and_color, tsne_compare_with_without_aplc, plot_tsne
import time
import torch
import numpy as np
from thop import profile as thop_profile
import matplotlib.pyplot as plt
import torch
import numpy as np
from models.CorrAPRNet import analyze_batch_coupling
def _corr_1d(x, y, eps=1e-8):
    """
    x,y: [..., N]
    return: [...]  (corr along last dim)
    """
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    x_std = x.std(dim=-1, keepdim=True) + eps
    y_std = y.std(dim=-1, keepdim=True) + eps
    return (x * y).mean(dim=-1) / (x_std.squeeze(-1) * y_std.squeeze(-1))

@torch.no_grad()
def coupling_strength_vs_freq(amp, phase, reduce_over_s=True):
    """
    amp, phase: [B, S, F]
    return:
      rho_freq: [F]  (耦合强度 vs 频率)
      rho_map:  [S, F] (耦合热力图，先对 batch 平均)
    """
    # -> [B,S,F]
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)

    # 我们希望对 (B,S) 维度聚合后得到每个频率的相关
    # 先把 B,S 合并： [B*S, F]
    B, S, F = amp.shape
    A = amp.reshape(B * F, S)
    C = cos_p.reshape(B * F, S)
    S1 = sin_p.reshape(B * F, S)

    # corr over samples (B*S) for each frequency
    # 这里要沿 “样本维” 计算相关，所以需要转置成 [F, B*S]
    A_t = A.t()     # [F, B*S]
    C_t = C.t()
    S_t = S1.t()

    r_cos = _corr_1d(A_t, C_t)  # [F]
    r_sin = _corr_1d(A_t, S_t)  # [F]
    rho_freq = torch.sqrt(r_cos**2 + r_sin**2).clamp(0, 1)  # [F]

    # heatmap：对每个 S（token/变量），做 batch 平均后再相关
    # 对每个 s：把 B 作为样本 -> corr over B
    # amp_s: [B,F] cos_s: [B,F] -> 对 B 求相关 => [F]
    A_bs = amp.permute(1,0,2)      # [S,B,F]
    C_bs = cos_p.permute(1,0,2)
    S_bs = sin_p.permute(1,0,2)

    # 转成 [S,F,B] 以便沿 B 做 corr
    A_s = A_bs.permute(0,2,1)  # [S,F,B]
    C_s = C_bs.permute(0,2,1)
    S_s = S_bs.permute(0,2,1)

    rcos_map = _corr_1d(A_s, C_s)   # [S,F]
    rsin_map = _corr_1d(A_s, S_s)   # [S,F]
    rho_map = torch.sqrt(rcos_map**2 + rsin_map**2).clamp(0, 1)  # [S,F]

    if reduce_over_s:
        rho_freq = rho_map.mean(dim=0)  # [F]（更强调“每个 token 的耦合”，而不是全局混合）
    return rho_freq, rho_map

@torch.no_grad()
def accumulate_coupling(diag_list, rho_freq_acc, rho_map_acc, key_amp="amp", key_phase="phase"):
    """
    diag_list: dict, e.g. diag["freqN"] or diag["freqC"]
    """
    amp = diag_list[key_amp]
    phase = diag_list[key_phase]

    # 统一成 [B,S,F]
    # ConvAttentionN: amp/phase 是 [B,C,F] -> 取 S=C
    # FreqAttention:  amp/phase 是 [B,L,F] -> 取 S=L
    if amp.dim() == 3 and phase.dim() == 3:
        if amp.shape[1] != phase.shape[1]:
            raise ValueError("amp/phase shape mismatch")

    # 如果是 [B,C,F]，直接当 [B,S,F]
    # 如果是 [B,L,F]，也直接当 [B,S,F]
    rho_freq, rho_map = coupling_strength_vs_freq(amp, phase, reduce_over_s=True)  # [F], [S,F]

    rho_freq_acc.append(rho_freq.detach().cpu())
    # heatmap 维度 S 可能不同（比如 L 很大），你可以只取前 64 行展示
    rho_map_acc.append(rho_map.detach().cpu())

def plot_coupling_vs_freq(rho_freq_mean, title="Coupling strength vs frequency"):
    F = rho_freq_mean.shape[0]
    x = np.linspace(0, 1, F)  # 归一化频率轴
    plt.figure()
    plt.plot(x, rho_freq_mean)
    plt.title(title)
    plt.xlabel("Normalized frequency")
    plt.ylabel("Coupling strength")
    plt.tight_layout()
    plt.show()

def plot_coupling_heatmap(rho_map_mean, title="Coupling heatmap"):
    # rho_map_mean: [S,F]
    plt.figure()
    plt.imshow(rho_map_mean, aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel("Frequency bin")
    plt.ylabel("Token/Channel index")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
def count_parameters_m(model: torch.nn.Module) -> float:
    """Return trainable parameters in Million (M)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@torch.no_grad()
def build_dummy_inputs(args, device, batch_size=1, dtype=torch.float32):
    """
    Build dummy inputs that match your training/test forward:
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    Shapes:
      batch_x:      [B, seq_len, enc_in]
      batch_x_mark: [B, seq_len, mark_dim] or None
      dec_inp:      [B, label_len + pred_len, dec_in]
      batch_y_mark: [B, label_len + pred_len, mark_dim] or None
    """
    B = batch_size
    seq_len = args.seq_len
    label_len = args.label_len
    pred_len = args.pred_len

    # enc_in/dec_in 通常等于变量数；在很多 repo 里是 args.enc_in / args.dec_in
    # 如果你只有 args.enc_in，就用 enc_in 兜底
    enc_in = getattr(args, "enc_in", None) or getattr(args, "c_in", None) or getattr(args, "input_dim", None)
    dec_in = getattr(args, "dec_in", None) or enc_in
    assert enc_in is not None, "args.enc_in (or c_in/input_dim) not found. Please set enc_in."

    # mark_dim：时间特征维度，ETT 常用 4；如果你的实现不同，改这里
    mark_dim = getattr(args, "mark_dim", None)
    if mark_dim is None:
        # 你 repo 里 batch_x_mark 是 float tensor，一般是 4 或 5
        # 这里给个常见默认值 4
        mark_dim = 4

    batch_x = torch.randn(B, seq_len, enc_in, device=device, dtype=dtype)

    # decoder input: [B, label_len + pred_len, dec_in]
    dec_inp = torch.randn(B, label_len + pred_len, dec_in, device=device, dtype=dtype)

    if ('PEMS' in args.data) or ('Solar' in args.data):
        batch_x_mark = None
        batch_y_mark = None
    else:
        batch_x_mark = torch.randn(B, seq_len, mark_dim, device=device, dtype=dtype)
        batch_y_mark = torch.randn(B, label_len + pred_len, mark_dim, device=device, dtype=dtype)

    return batch_x, batch_x_mark, dec_inp, batch_y_mark


def try_thop_flops(model, inputs):
    """
    Compute MACs/FLOPs via thop.
    thop returns: macs, params
    Many papers report FLOPs ≈ 2*MACs for multiply-add counting.
    Here we output both to be safe.
    """
    model.eval()
    macs, params = thop_profile(model, inputs=inputs, verbose=False)
    # Convert to G
    macs_g = macs / 1e9
    flops_g = (2.0 * macs) / 1e9
    params_m = params / 1e6
    return params_m, macs_g, flops_g


@torch.no_grad()
def measure_latency_ms(
    model,
    inputs,
    warmup=50,
    runs=200,
    use_amp=False
):
    """
    Measure end-to-end forward latency in ms (mean ± std).
    Uses CUDA events for accurate GPU timing.
    """
    model.eval()
    device = inputs[0].device
    assert device.type == "cuda", "Latency measurement here assumes CUDA. For CPU, use time.time()."

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(warmup):
        if use_amp:
            with torch.cuda.amp.autocast():
                _ = model(*inputs)
        else:
            _ = model(*inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        starter.record()
        if use_amp:
            with torch.cuda.amp.autocast():
                _ = model(*inputs)
        else:
            _ = model(*inputs)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

    times = np.array(times, dtype=np.float64)
    return float(times.mean()), float(times.std())


@torch.no_grad()
def measure_peak_memory_mib(model, inputs, use_amp=False):
    """
    Measure peak allocated CUDA memory (MiB) during forward.
    """
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    if use_amp:
        with torch.cuda.amp.autocast():
            _ = model(*inputs)
    else:
        _ = model(*inputs)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / 1024**2)


def profile_model(model, args, batch_size=1, use_amp=False):
    """
    Returns a dict: params(M), MACs(G), FLOPs(G), latency(ms), peak_mem(MiB)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dtype = torch.float16 if (use_amp and device.type == "cuda") else torch.float32
    inputs = build_dummy_inputs(args, device=device, batch_size=batch_size, dtype=dtype)

    # Params (trainable)
    params_train_m = count_parameters_m(model)

    # thop FLOPs/MACs/Params (thop params is all params; trainable maybe slightly different)
    try:
        thop_params_m, macs_g, flops_g = try_thop_flops(model, inputs)
    except Exception as e:
        thop_params_m, macs_g, flops_g = float("nan"), float("nan"), float("nan")
        print(f"[WARN] thop failed: {e}")

    # Latency
    if device.type == "cuda":
        lat_mean_ms, lat_std_ms = measure_latency_ms(model, inputs, warmup=50, runs=200, use_amp=use_amp)
        peak_mem_mib = measure_peak_memory_mib(model, inputs, use_amp=use_amp)
    else:
        # CPU fallback
        model.eval()
        for _ in range(10):
            _ = model(*inputs)
        t0 = time.time()
        runs = 50
        for _ in range(runs):
            _ = model(*inputs)
        t1 = time.time()
        lat_mean_ms = (t1 - t0) * 1000.0 / runs
        lat_std_ms = float("nan")
        peak_mem_mib = float("nan")

    return {
        "params_train_M": params_train_m,
        "params_thop_M": thop_params_m,
        "macs_G": macs_g,
        "flops_G": flops_g,
        "latency_ms_mean": lat_mean_ms,
        "latency_ms_std": lat_std_ms,
        "peak_mem_MiB": peak_mem_mib,
        "batch_size": batch_size,
        "amp": use_amp,
        "device": str(device),
    }

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def profile_complexity(self, batch_size=1, use_amp=False):
        stats = profile_model(self.model, self.args, batch_size=batch_size, use_amp=use_amp)
        print("\n========== Model Complexity ==========")
        print(f"Device: {stats['device']} | Batch: {stats['batch_size']} | AMP: {stats['amp']}")
        print(f"Trainable Params (M): {stats['params_train_M']:.3f}")
        if np.isfinite(stats["params_thop_M"]):
            print(f"THOP Params (M):      {stats['params_thop_M']:.3f}")
        if np.isfinite(stats["macs_G"]):
            print(f"MACs (G):             {stats['macs_G']:.3f}")
        if np.isfinite(stats["flops_G"]):
            print(f"FLOPs (G, ~2*MACs):    {stats['flops_G']:.3f}")
        print(f"Latency (ms):         {stats['latency_ms_mean']:.3f} ± {stats['latency_ms_std']:.3f}")
        print(f"Peak Mem (MiB):       {stats['peak_mem_MiB']:.2f}")
        print("======================================\n")
        return stats

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        self.model._metrics_rows = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))
                # self.model.save_metrics_excel("coupling_analysis.xlsx", compute_correlations=True)
        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # self.model.save_metrics_to_excel("diagnostic.xlsx")

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # loss_float = loss.item()
                    # train_loss.append(loss_float)
                    loss.backward()
                    model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()

        # dummy_input = torch.randn((1, 512, 7)).to('cuda:0')
        # self.model = self.model.to('cuda:0')
        # _ = self.model(dummy_input, None, None, None)
        # self.profile_complexity(batch_size=1, use_amp=self.args.use_amp)

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                stats = analyze_batch_coupling(
                    model=self.model,
                    batch_x=batch_x,  # [B, seq_len, C]
                    title_prefix="APRNet",
                    use_freq_branch="freqC",  # 或 "freqN"
                    device="cuda",
                    delta_dim_for_L=1,  # 对 [B,L,F] 的 L 维做 Δφ
                    unwrap=True,
                    plot=True,
                )

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))


        # feats, colors = collect_feats_and_color(self.model, test_loader, 'cuda:0', max_batches=50, color_mode="none")
        # plot_tsne(feats, colors, title="t-SNE (with APLC)")
        #
        # # 推荐：对比 w/o vs with（并用谱熵连续上色）
        # tsne_compare_with_without_aplc(self.model, test_loader, 'cuda:0', max_batches=50, color_mode="spectral_entropy")

        mse = mse.avg
        mae = mae.avg
        print('mse:{}, mae:{}'.format(mse, mae))

        return
