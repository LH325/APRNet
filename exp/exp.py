import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import os
import time
import warnings
from scipy import fft  # 实数 FFT
import matplotlib.pyplot as plt  # 频谱绘图

warnings.filterwarnings('ignore')
plt.rcParams.update({
    "font.size": 18,          # 文字大小
    "axes.titlesize": 20,     # 标题字体
    "axes.labelsize": 18,     # 坐标轴字体
    "xtick.labelsize": 16,    # x 轴刻度字体
    "ytick.labelsize": 16,    # y 轴刻度字体
    "legend.fontsize": 16,    # 图例字体
    "figure.titlesize": 22    # 整体 figure 标题字体
})

def plot_spectrum(true_seq, pred_seq, fs=1.0, title=None, save_path=True):
    """
    绘制真实序列和预测序列的幅度频谱（使用 rFFT），并标注低/中/高频区域。
    """
    true_seq = np.asarray(true_seq)
    pred_seq = np.asarray(pred_seq)
    L = true_seq.shape[0]

    # 实数 FFT
    T_fft = np.fft.rfft(true_seq)
    P_fft = np.fft.rfft(pred_seq)

    # 幅度谱
    mag_T = np.abs(T_fft)
    mag_P = np.abs(P_fft)

    # 频率轴
    freqs = np.fft.rfftfreq(L, d=1.0/fs)
    n_freqs = len(freqs)

    # 三个区域的边界（按频率数量比例）
    lf_end = int(0.10 * n_freqs)   # 0%–10%
    mf_end = int(0.50 * n_freqs)   # 10%–50%
    hf_end = n_freqs               # 50%–100%

    lf_freq = freqs[lf_end]
    mf_freq = freqs[mf_end]

    plt.figure(figsize=(10, 4))

    # 不指定颜色，使用默认色
    plt.plot(freqs, mag_T, label="True")
    plt.plot(freqs, mag_P, label="Pred")

    # =============================
    # 标注低频区（0–10%）
    plt.axvspan(freqs[0], freqs[lf_end], color="gray", alpha=0.15)
    plt.text(freqs[lf_end] * 0.5, max(mag_T) * 0.9, "Low\n(0–10%)",
             ha="center", va="top", fontsize=18)

    # 标注中频区（10%–50%）
    plt.axvspan(freqs[lf_end], freqs[mf_end], color="gray", alpha=0.10)
    plt.text(freqs[lf_end] + (freqs[mf_end]-freqs[lf_end])*0.5,
             max(mag_T) * 0.9, "Mid\n(10–50%)",
             ha="center", va="top", fontsize=18)

    # 标注高频区（50%–100%）
    plt.axvspan(freqs[mf_end], freqs[hf_end-1], color="gray", alpha=0.05)
    plt.text(freqs[mf_end] + (freqs[hf_end-1]-freqs[mf_end])*0.5,
             max(mag_T) * 0.9, "High\n(50–100%)",
             ha="center", va="top", fontsize=18)
    # =============================

    plt.xlabel("Frequency Index")
    plt.ylabel("Spectral Value")
    if title is not None:
        plt.title(title)

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))

        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        time_now = time.time()

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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                train_loss.append(loss.item())

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

    # =================== Modified test() with Spectral Bias Evaluation + Spectrum Plot ===================
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('Loading saved model...')
            ckpt_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt_path))

        self.model.eval()

        # Frequency metrics accumulators
        lf_rmse_meter = AverageMeter()
        mf_rmse_meter = AverageMeter()
        hf_rmse_meter = AverageMeter()
        gf_rmse_meter = AverageMeter()  # global full-spectrum RMSE
        total_mse = AverageMeter()

        # 控制只画一次图
        spectrum_plotted = False

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                # Model forward
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[:, -self.args.pred_len:, f_dim:]  # [B, pred_len, D]
                true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Move to CPU for NumPy processing
                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()

                batch_size, pred_len, channels = pred.shape

                # Flatten spatial dimensions: treat each channel independently
                pred_flat = pred.reshape(-1, pred_len)  # [B*D, L]
                true_flat = true.reshape(-1, pred_len)

                # Compute overall MSE (spatial average)
                mse_per_sample = ((pred_flat - true_flat) ** 2).mean(axis=1)
                total_mse.update(mse_per_sample.mean(), batch_size)

                # === Spectral Analysis ===
                for j in range(pred_flat.shape[0]):  # loop over each sequence-channel
                    p = pred_flat[j]
                    t = true_flat[j]

                    # 只画一次：第一个 batch 的第一条序列
                    if (not spectrum_plotted) and i == 0 and j == 0:
                        fig_save_dir = os.path.join("checkpoints", setting)
                        fig_save_path = os.path.join(
                            fig_save_dir,
                            "APRNet.pdf"
                        )
                        plot_spectrum(
                            true_seq=t,
                            pred_seq=p,
                            fs=1.0,  # 如有真实采样频率，可在此修改
                            title=f"APRNet",
                            save_path=fig_save_path
                        )
                        print(f"Spectrum figure saved to: {fig_save_path}")
                        spectrum_plotted = True

                    # Real FFT
                    P_fft = fft.rfft(p)  # complex spectrum
                    T_fft = fft.rfft(t)

                    # Get magnitude for sorting
                    mag_P = np.abs(P_fft)
                    mag_T = np.abs(T_fft)

                    n_freqs = len(mag_P)

                    # Sort indices by magnitude (descending)
                    sorted_indices_P = np.argsort(mag_P)[::-1]
                    sorted_indices_T = np.argsort(mag_T)[::-1]

                    # Define top k based on percentage
                    top_10 = n_freqs // 10
                    top_50 = n_freqs // 2

                    # Select dominant frequency components by magnitude
                    lf_idx_P = sorted_indices_P[:top_10]
                    mf_idx_P = sorted_indices_P[top_10:top_50]
                    hf_idx_P = sorted_indices_P[top_50:]

                    lf_idx_T = sorted_indices_T[:top_10]
                    mf_idx_T = sorted_indices_T[top_10:top_50]
                    hf_idx_T = sorted_indices_T[top_50:]

                    # Reconstruct signals using selected freqs (only keep those indices, zero others)
                    def reconstruct_from_indices(spec, indices, length):
                        rec_spec = np.zeros_like(spec, dtype=complex)
                        rec_spec[indices] = spec[indices]
                        return fft.irfft(rec_spec, n=length)

                    # Reconstruct low/mid/high frequency parts
                    p_lf = reconstruct_from_indices(P_fft, lf_idx_P, pred_len)
                    t_lf = reconstruct_from_indices(T_fft, lf_idx_T, pred_len)

                    p_mf = reconstruct_from_indices(P_fft, mf_idx_P, pred_len)
                    t_mf = reconstruct_from_indices(T_fft, mf_idx_T, pred_len)

                    p_hf = reconstruct_from_indices(P_fft, hf_idx_P, pred_len)
                    t_hf = reconstruct_from_indices(T_fft, hf_idx_T, pred_len)

                    # Full reconstruction (global)
                    p_gf = fft.irfft(P_fft, n=pred_len)
                    t_gf = fft.irfft(T_fft, n=pred_len)

                    # Compute RMSE for each band
                    def rmse(x, y):
                        return np.sqrt(np.mean((x - y) ** 2))

                    lf_rmse_meter.update(rmse(t_lf, p_lf))
                    mf_rmse_meter.update(rmse(t_mf, p_mf))
                    hf_rmse_meter.update(rmse(t_hf, p_hf))
                    gf_rmse_meter.update(rmse(t_gf, p_gf))

        # Final results
        print(f"Test Results ({setting}):")
        print(f"Global RMSE: {gf_rmse_meter.avg:.6f}")
        print(f"Low-Freq RMSE (Top 10%): {lf_rmse_meter.avg:.6f}")
        print(f"Mid-Freq RMSE (10%-50%): {mf_rmse_meter.avg:.6f}")
        print(f"High-Freq RMSE (50%-100%): {hf_rmse_meter.avg:.6f}")
        print(f"Overall MSE: {total_mse.avg:.6f}")

        # Optional: save to file or return dict
        results = {
            'setting': setting,
            'global_rmse': gf_rmse_meter.avg,
            'lf_rmse': lf_rmse_meter.avg,
            'mf_rmse': mf_rmse_meter.avg,
            'hf_rmse': hf_rmse_meter.avg,
            'overall_mse': total_mse.avg
        }

        import json
        with open(f"results_spectral_{setting}.json", "w") as f:
            json.dump(results, f, indent=4)

        return results
