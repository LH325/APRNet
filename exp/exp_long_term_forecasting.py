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
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22
})


def plot_spectrum(true_seq, pred_seq, fs=1.0, title=None, save_path=True):
    """
    绘制真实序列和预测序列的幅度频谱（使用 rFFT），并标注低/中/高频区域。
    true_seq, pred_seq: 1D numpy array, shape [L]
    fs: 采样频率（时间步为 1 时可用默认 1.0）
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
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    n_freqs = len(freqs)

    # 按频点索引划分 0–10%、10–50%、50–100%
    lf_end = int(0.10 * n_freqs)   # 0%–10%
    mf_end = int(0.50 * n_freqs)   # 10%–50%
    hf_end = n_freqs               # 50%–100%

    plt.figure(figsize=(10, 4))
    # 使用默认颜色
    plt.plot(freqs, mag_T, label="True")
    plt.plot(freqs, mag_P, label="Pred")

    ymax = max(mag_T.max(), mag_P.max())

    # ===== 区域标注 =====
    # 低频 0–10%
    plt.axvspan(freqs[0], freqs[lf_end], alpha=0.15)
    plt.text(freqs[lf_end] * 0.5, ymax * 0.9, "Low\n(0–10%)",
             ha="center", va="top")

    # 中频 10–50%
    plt.axvspan(freqs[lf_end], freqs[mf_end], alpha=0.10)
    plt.text(freqs[lf_end] + (freqs[mf_end] - freqs[lf_end]) * 0.5,
             ymax * 0.9, "Mid\n(10–50%)",
             ha="center", va="top")

    # 高频 50–100%
    plt.axvspan(freqs[mf_end], freqs[hf_end - 1], alpha=0.05)
    plt.text(freqs[mf_end] + (freqs[hf_end - 1] - freqs[mf_end]) * 0.5,
             ymax * 0.9, "High\n(50–100%)",
             ha="center", va="top")

    plt.xlabel("Frequency Index")
    plt.ylabel("Spectral Value")
    if title is not None:
        plt.title('FreLE')
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
from models.APNet import collect_feats_and_color,tsne_compare_with_without_aplc,plot_tsne
# from thop import profile
# def flops(model,input_shape=(1, 96, 7)):
#     dummy_input = torch.randn(input_shape).cuda()
#
#     flops, params = profile(model, inputs=(dummy_input,None,None,None))
#     print('flops',flops)
# def count_parameters(model):
#
#     print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)  # 单位：百万
#
# def measure_memory(model, input_shape=(1, 96, 7)):
#     model = model.cuda()
#     dummy_input = torch.randn(input_shape).cuda()
#
#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         _ = model(dummy_input,None,None,None)
#
#     mem_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2  # 转为MiB
#     print(f"Mem. (MiB): {mem_alloc:.2f}")
# def measure_speed(model, input_shape=(1, 96, 7), steps=100):
#     model = model.cuda()
#     dummy_input = torch.randn(input_shape).cuda()
#     torch.cuda.synchronize()
#     start = time.time()
#     for _ in range(steps):
#         with torch.no_grad():
#             _ = model(dummy_input,None,None,None)
#     torch.cuda.synchronize()
#     end = time.time()
#     print(f"Speed: {(end - start)/steps:.4f} s/step")
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

        preds = []
        trues = []

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()
        self.model.eval()
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

                pred = outputs
                true = batch_y

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        feats, colors = collect_feats_and_color(self.model, test_loader, 'cuda:0', max_batches=50, color_mode="none")
        plot_tsne(feats, colors, title="t-SNE (with APGC)")

        # 推荐：对比 w/o vs with（并用谱熵连续上色）
        tsne_compare_with_without_aplc(self.model, test_loader, 'cuda:0', max_batches=50, color_mode="spectral_entropy")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        N, L, D = preds.shape
        preds_flat = preds.reshape(-1, L)  # [N*D, L]
        trues_flat = trues.reshape(-1, L)  # [N*D, L]

        spectrum_plotted = False

        for j in range(preds_flat.shape[0]):
            p = preds_flat[j]
            t = trues_flat[j]

            # 第一个样本画频谱图
            if not spectrum_plotted:
                fig_save_dir = os.path.join("checkpoints", setting)
                fig_save_path = os.path.join(fig_save_dir, "spectrum_true_vs_pred.pdf")
                print(fig_save_path)
                plot_spectrum(
                    true_seq=t,
                    pred_seq=p,
                    fs=1.0,  # 如果有真实采样频率可以改这里
                    title="Spectrum of sample (test set)",
                    save_path=fig_save_path
                )
                print(f"Spectrum figure saved to: {fig_save_path}")
                spectrum_plotted = True

        mse = mse.avg
        mae = mae.avg
        print('mse:{}, mae:{}'.format(mse, mae))

        return
