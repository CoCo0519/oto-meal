"""
心搏模板相减 + ANC（LMS）流水线 @100 Hz
依赖：numpy, scipy, matplotlib
支持仅三轴 IMU（仅加速度，gyro 可为 None）
使用：
    from anc_template_pipeline_100Hz import run_pipeline
    out = run_pipeline(ppg, acc, gyro=None, fs=100, mains=50, make_demo_plot=True)

输入：
    ppg: (N,) 原始 PPG
    acc: (N,3) 加速度 x,y,z
    gyro: (N,3) 陀螺仪 x,y,z 或 None

输出（dict）：
    'ppg_hp'       : 高通+陷波后的 PPG
    'ppg_bpHR'     : 心搏带通（用于峰检）
    'peaks'        : 心搏峰索引
    'ppg_template' : 平均心搏模板（标准化长度）
    'ppg_recon'    : 基于模板的重构心搏信号
    'ppg_residual' : 模板相减残差
    'ppg_anc'      : ANC 后 PPG（对 residual 进一步去运动伪影；若无参考通道，则等于 residual）
    'imu_energy'   : {'acc':..., 'gyro':(可选)}
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import matplotlib.pyplot as plt


# ----------------- 基础滤波 -----------------

def notch_filter(x, fs, mains=50, Q=30):
    y = np.asarray(x).copy()
    max_harm = int((fs / 2) // mains)
    for k in range(1, max_harm + 1):
        f0 = k * mains
        bw = f0 / Q
        b, a = iirnotch(w0=f0 / (fs / 2), Q=f0 / bw)
        y = filtfilt(b, a, y)
    return y


def butter_highpass(x, fs, fc=0.1, order=2):
    b, a = butter(order, fc / (fs / 2), btype='high')
    return filtfilt(b, a, x)


def butter_bandpass(x, fs, f1, f2, order=3):
    b, a = butter(order, [f1 / (fs / 2), f2 / (fs / 2)], btype='band')
    # 明确沿时间轴 (axis=0) 滤波，兼容形状 (N,) 或 (N,C)
    return filtfilt(b, a, x, axis=0)


# ----------------- 心搏模板相减 -----------------

def heartbeat_template_subtraction(ppg, fs=100):
    """返回模板、重构与残差。ppg 需已做去漂移/陷波。"""
    # 1) 峰检
    ppg_bp = butter_bandpass(ppg, fs, 0.8, 5.0, order=3)
    min_dist = int(0.3 * fs)  # 至少 300 ms 间隔
    peaks, _ = find_peaks(ppg_bp, distance=min_dist, prominence=np.std(ppg_bp) * 0.5)

    if len(peaks) < 4:
        return {
            'ppg_bpHR': ppg_bp,
            'peaks': peaks,
            'template': np.zeros(200),
            'recon': np.zeros_like(ppg),
            'residual': ppg.copy()
        }

    # 2) 片段重采样到固定长度（200）
    Ntpl = 200
    segs = []
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        seg = ppg[a:b]
        if len(seg) < int(0.25 * fs):
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
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        RR = b - a
        xo = np.linspace(0, 1, Ntpl)
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


# ----------------- 多通道 ANC（LMS / NLMS） -----------------

def nlms_anc(d, refs, mu=5e-4, order=12, eps=1e-6):
    """归一化 LMS ANC"""
    N, K = refs.shape
    if K == 0:
        return np.zeros(N), d.copy(), np.zeros((0,))

    # 构造延时线：U[t] 形状 (K*order,)
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
        y_hat[n] = np.dot(w, u)
        e[n] = d[n] - y_hat[n]
        norm = np.dot(u, u) + eps
        w = w + mu * e[n] * u / norm

    return y_hat, e, w


# ----------------- IMU 特征（短时能量） -----------------

def short_time_energy(x, fs, win_sec=0.5):
    win = np.hanning(int(win_sec * fs))
    win /= np.sum(win)
    x2 = x ** 2
    return np.convolve(x2, win, mode='same')


# ----------------- 主流程 -----------------

def run_pipeline(ppg, acc, gyro=None, fs=100, mains=50, make_demo_plot=False):
    ppg = np.asarray(ppg).astype(float).flatten()
    acc = np.asarray(acc, dtype=float).reshape(-1, 3)
    gyro_arr = None if gyro is None or len(np.asarray(gyro).shape) == 0 else np.asarray(gyro, dtype=float).reshape(-1, 3)

    # 1) PPG 预处理：陷波 + 高通
    ppg_nf = notch_filter(ppg, fs, mains=mains, Q=30)
    ppg_hp = butter_highpass(ppg_nf, fs, fc=0.1, order=2)

    # 2) 心搏模板相减
    tpl_out = heartbeat_template_subtraction(ppg_hp, fs=fs)
    ppg_res = tpl_out['residual']

    # 3) ANC 参考：acc 及其一阶差分（可选 gyro）
    acc_bp = butter_bandpass(acc, fs, 0.5, 15.0, order=3)
    accd = np.vstack([np.r_[0, np.diff(acc_bp[:, i])] for i in range(3)]).T

    refs_list = [acc_bp, accd]
    gyro_bp = None
    if gyro_arr is not None:
        gyro_bp = butter_bandpass(gyro_arr, fs, 0.5, 15.0, order=3)
        gyrod = np.vstack([np.r_[0, np.diff(gyro_bp[:, i])] for i in range(3)]).T
        refs_list += [gyro_bp, gyrod]

    refs = np.hstack(refs_list) if len(refs_list) > 0 else np.zeros((len(ppg_res), 0))

    # 4) ANC（对 residual 再去运动伪影）
    y_hat, e, _ = nlms_anc(ppg_res, refs, mu=5e-4, order=12, eps=1e-6)
    ppg_anc = e  # ANC 输出的残差

    # 5) IMU 能量
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


# ----------------- 简单合成数据自测 -----------------

if __name__ == '__main__':
    fs = 100
    N = fs * 30  # 30 s
    t = np.arange(N) / fs

    # 合成 PPG
    hr = 1.2
    ppg = 0.8 * np.sin(2 * np.pi * hr * t) + 0.05 * np.sin(2 * np.pi * 0.05 * t) + 0.02 * np.random.randn(N)
    swallow_idx = [int(8 * fs), int(20 * fs)]
    for si in swallow_idx:
        ppg[si:si + int(0.6 * fs)] += np.hstack([
            np.linspace(0, 0.6, int(0.2 * fs)),
            np.linspace(0.6, 0, int(0.4 * fs))
        ])

    # 合成三轴加速度
    acc = 0.02 * np.random.randn(N, 3)
    for si in swallow_idx:
        acc[si:si + int(0.4 * fs), :] += 0.6 * np.hanning(int(0.4 * fs))[:, None]

    out = run_pipeline(ppg, acc, gyro=None, fs=fs, mains=50, make_demo_plot=True)
