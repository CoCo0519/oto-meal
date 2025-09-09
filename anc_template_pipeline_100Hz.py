"""
心搏模板相减 + ANC（LMS）流水线 @100 Hz（稳健版）
依赖：numpy, scipy, matplotlib
使用：
    from anc_template_pipeline_100Hz import run_pipeline
    out = run_pipeline(ppg, acc, gyro=None, fs=100, mains=50, make_demo_plot=True)

可调关键参数（见 run_pipeline 文档）：
    tpl_N=300, tpl_prominence_scale=0.8, tpl_min_rr_sec=0.35,
    pre_anc_bp=(0.4, 8.0), anc_mu=8e-4, anc_order=16
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import matplotlib.pyplot as plt


# ----------------- 基础滤波 -----------------

def notch_filter(x, fs, mains=50, Q=30):
    y = np.asarray(x).copy()
    nyq = fs / 2.0
    # 仅 notch 落在 (0, nyq) 的谐波，避免 W0==1
    max_harm = int(np.floor((nyq - 1e-9) // mains))
    for k in range(1, max_harm + 1):
        f0 = k * mains
        w0 = f0 / nyq
        bw = w0 / Q
        b, a = iirnotch(w0=w0, Q=w0 / bw)
        y = filtfilt(b, a, y, axis=0)
    return y


def butter_highpass(x, fs, fc=0.1, order=2):
    b, a = butter(order, fc / (fs / 2.0), btype='high')
    # 明确沿时间轴滤波，兼容 (N,) 或 (N, C)
    return filtfilt(b, a, x, axis=0)


def butter_bandpass(x, fs, f1, f2, order=3):
    b, a = butter(order, [f1 / (fs / 2.0), f2 / (fs / 2.0)], btype='band')
    return filtfilt(b, a, x, axis=0)


# ----------------- 心搏模板相减（稳健版） -----------------

def heartbeat_template_subtraction(
    ppg,
    fs=100,
    Ntpl=300,
    prominence_scale=0.8,
    min_rr_sec=0.35,
):
    """
    ppg 需已做去漂移/陷波。
    Ntpl: 模板标准长度（建议 300–400）
    prominence_scale: 峰检门限 = prominence_scale * std(ppg_bp)
    min_rr_sec: 邻峰最小间隔（秒）
    """
    # 1) 心搏带通用于峰检
    ppg_bp = butter_bandpass(ppg, fs, 0.8, 5.0, order=3)
    min_dist = int(max(1, min_rr_sec * fs))
    prom = float(np.std(ppg_bp)) * float(prominence_scale)
    peaks, _ = find_peaks(ppg_bp, distance=min_dist, prominence=prom)

    if len(peaks) < 4:
        return {
            'ppg_bpHR': ppg_bp,
            'peaks': peaks,
            'template': np.zeros(Ntpl),
            'recon': np.zeros_like(ppg),
            'residual': ppg.copy()
        }

    # 2) 片段重采样到固定长度
    segs = []
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        seg = ppg[a:b]
        if len(seg) < int(0.25 * fs):  # 过短跳过
            continue
        xi = np.linspace(0, 1, len(seg))
        xo = np.linspace(0, 1, Ntpl)
        seg_rs = np.interp(xo, xi, seg)
        segs.append(seg_rs)

    if len(segs) < 3:
        return {
            'ppg_bpHR': ppg_bp,
            'peaks': peaks,
            'template': np.zeros(Ntpl),
            'recon': np.zeros_like(ppg),
            'residual': ppg.copy()
        }

    template = np.median(np.vstack(segs), axis=0)

    # 3) 依据每个 RR 长度重建
    recon = np.zeros_like(ppg)
    xo = np.linspace(0, 1, Ntpl)
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        RR = b - a
        if RR <= 1:
            continue
        xi = np.linspace(0, 1, RR)
        tpl_scaled = np.interp(xi, xo, template)
        recon[a:b] = tpl_scaled

    residual = ppg - recon

    return {
        'ppg_bpHR': ppg_bp,
        'peaks': peaks,
        'template': template,
        'recon': recon,
        'residual': residual
    }


# ----------------- 多通道 ANC（NLMS） -----------------

def nlms_anc(d, refs, mu=8e-4, order=16, eps=1e-6):
    """
    d : (N,) 目标通道
    refs : (N, K) 参考矩阵（acc 及其差分/gyro等）
    mu : 步长（适度增大可加快收敛，但风险过拟合）
    order : 每参考通道 FIR 阶
    """
    N = len(d)
    refs = np.asarray(refs)
    if refs.ndim == 1:
        refs = refs[:, None]
    if refs.size == 0 or refs.shape[1] == 0:
        return np.zeros(N), d.copy(), np.zeros((0,))

    N, K = refs.shape
    U = np.zeros((N, K * order))
    for k in range(K):
        x = refs[:, k]
        for o in range(order):
            U[o:N, k * order + o] = x[:N - o]

    w = np.zeros(K * order)
    y_hat = np.zeros(N)
    e = np.zeros(N)

    for n in range(N):
        u = U[n]
        yh = np.dot(w, u)
        y_hat[n] = yh
        err = d[n] - yh
        e[n] = err
        norm = np.dot(u, u) + eps
        w += mu * err * u / norm

    return y_hat, e, w


# ----------------- IMU 特征（短时能量） -----------------

def short_time_energy(x, fs, win_sec=0.5):
    win = np.hanning(int(max(3, win_sec * fs)))
    win /= np.sum(win)
    x2 = x ** 2
    return np.convolve(x2, win, mode='same')


# ----------------- 主流程（可调参数版） -----------------

def run_pipeline(
    ppg,
    acc,
    gyro=None,
    fs=100,
    mains=50,
    make_demo_plot=False,
    # --- 模板相减调参 ---
    tpl_N=300,
    tpl_prominence_scale=0.8,
    tpl_min_rr_sec=0.35,
    # --- ANC 预处理带通 ---
    pre_anc_bp=(0.4, 8.0),
    # --- ANC 调参 ---
    anc_mu=8e-4,
    anc_order=16,
):
    """
    返回字典包含：ppg_hp, ppg_bpHR, peaks, ppg_template, ppg_recon,
                 ppg_residual, ppg_anc, imu_energy, acc_bp, gyro_bp
    """
    ppg = np.asarray(ppg, float).flatten()
    acc = np.asarray(acc, float).reshape(-1, 3)
    gyro_arr = None
    if gyro is not None and np.asarray(gyro).size > 0:
        gyro_arr = np.asarray(gyro, float).reshape(-1, 3)

    # 1) PPG 预处理：陷波 + 高通
    ppg_nf = notch_filter(ppg, fs, mains=mains, Q=30)
    ppg_hp = butter_highpass(ppg_nf, fs, fc=0.1, order=2)

    # 2) 心搏模板相减（稳健参数）
    tpl_out = heartbeat_template_subtraction(
        ppg_hp, fs=fs, Ntpl=tpl_N,
        prominence_scale=tpl_prominence_scale,
        min_rr_sec=tpl_min_rr_sec,
    )
    ppg_res = tpl_out['residual']

    # 3) ANC 参考：acc/gyro 带通 + 一阶差分
    acc_bp = butter_bandpass(acc, fs, 0.5, 15.0, order=3)
    accd = np.vstack([np.r_[0.0, np.diff(acc_bp[:, i])] for i in range(3)]).T

    refs_list = [acc_bp, accd]
    gyro_bp = None
    if gyro_arr is not None:
        gyro_bp = butter_bandpass(gyro_arr, fs, 0.5, 15.0, order=3)
        gyrod = np.vstack([np.r_[0.0, np.diff(gyro_bp[:, i])] for i in range(3)]).T
        refs_list += [gyro_bp, gyrod]

    refs = np.hstack(refs_list) if len(refs_list) > 0 else np.zeros((len(ppg_res), 0))

    # 4) ANC 前温和带通（去极低频漂移）
    if pre_anc_bp is not None:
        f1, f2 = pre_anc_bp
        ppg_res_for_anc = butter_bandpass(ppg_res, fs, f1, f2, order=2)
    else:
        ppg_res_for_anc = ppg_res

    # 5) ANC
    y_hat, e, _ = nlms_anc(ppg_res_for_anc, refs, mu=anc_mu, order=anc_order, eps=1e-6)
    ppg_anc = e

    # 6) IMU 能量
    acc_mag = np.linalg.norm(acc_bp, axis=1)
    en_acc = short_time_energy(acc_mag, fs)
    en_gyro = None
    if gyro_bp is not None:
        gyro_mag = np.linalg.norm(gyro_bp, axis=1)
        en_gyro = short_time_energy(gyro_mag, fs)

    out = {
        'ppg_hp': ppg_hp,
        'ppg_bpHR': tpl_out['ppg_bpHR'],
        'peaks': tpl_out['peaks'],
        'ppg_template': tpl_out['template'],
        'ppg_recon': tpl_out['recon'],
        'ppg_residual': ppg_res,
        'ppg_anc': ppg_anc,
        'imu_energy': {'acc': en_acc, 'gyro': en_gyro},
        'acc_bp': acc_bp,
        'gyro_bp': gyro_bp,
    }

    if make_demo_plot:
        plot_comparison(ppg, out, fs)

    return out


# ----------------- 对比图 -----------------

def plot_comparison(ppg_raw, out, fs=100):
    t = np.arange(len(ppg_raw)) / fs
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, ppg_raw, label='Raw PPG')
    axes[0].plot(t, out['ppg_hp'], alpha=0.7, label='PPG high-pass + notch')
    axes[0].set_title('PPG (raw vs. high-pass)')
    axes[0].legend(loc='upper right')

    axes[1].plot(t, out['ppg_hp'], label='PPG HP')
    axes[1].plot(t, out['ppg_recon'], label='Heartbeat template recon')
    axes[1].plot(t, out['ppg_residual'], label='Residual (after template subtraction)')
    axes[1].set_title('Template subtraction')
    axes[1].legend(loc='upper right')

    axes[2].plot(t, out['ppg_residual'], label='Residual before ANC')
    axes[2].plot(t, out['ppg_anc'], label='After ANC (final)')
    axes[2].set_title('ANC on residual')
    axes[2].legend(loc='upper right')

    axes[3].plot(t, out['imu_energy']['acc'], label='IMU acc energy')
    if out['imu_energy']['gyro'] is not None:
        axes[3].plot(t, out['imu_energy']['gyro'], label='IMU gyro energy')
    axes[3].set_title('IMU short-time energy')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right')

    fig.tight_layout()
    plt.show()


# ----------------- 自测（可选） -----------------
if __name__ == '__main__':
    fs = 100
    N = fs * 30
    t = np.arange(N) / fs

    hr = 1.2
    ppg = 0.8 * np.sin(2 * np.pi * hr * t) + 0.05 * np.sin(2 * np.pi * 0.05 * t) + 0.02 * np.random.randn(N)
    swallow_idx = [int(8 * fs), int(20 * fs)]
    for si in swallow_idx:
        ppg[si:si + int(0.6 * fs)] += np.hstack([np.linspace(0, 0.6, int(0.2 * fs)),
                                                  np.linspace(0.6, 0, int(0.4 * fs))])

    acc = 0.02 * np.random.randn(N, 3)
    for si in swallow_idx:
        acc[si:si + int(0.4 * fs), :] += 0.6 * np.hanning(int(0.4 * fs))[:, None]

    out = run_pipeline(ppg, acc, gyro=None, fs=fs, mains=50, make_demo_plot=True)
