# -*- coding: utf-8 -*-
import numpy as np
from anc_template_pipeline_100Hz import (
    run_pipeline, notch_filter, butter_highpass, butter_bandpass
)

# === 修改为你的路径、采样率与工频 ===
txt = './hyx_data/喉咙-吞咽6次间隔10秒.txt'
fs = 100
mains = 50  # 或 60

# ---------- 读取 txt（首行中文表头） ----------
# 若 numpy 版本不支持 encoding，可用 open(..., encoding='utf-8')
data = np.loadtxt(txt, skiprows=1, encoding='utf-8')

# 列映射
ppg_green = data[:, 0]
ppg_ir    = data[:, 1]
ppg_red   = data[:, 2]
acc_raw   = data[:, 3:6]  # X, Y, Z
gyro_raw  = None          # 本文件无陀螺

# 选择 PPG 通道（默认绿光；也可换成 ppg_ir / ppg_red）
ppg_raw = ppg_green

# ---------- 等价于 MATLAB 的第一次预处理 ----------
# 1) PPG：工频陷波 + 高通 0.1 Hz   ← 得到 ppg_hp（与 MATLAB ppg_hp 等价）
ppg_nf = notch_filter(ppg_raw, fs, mains=mains, Q=30)
ppg_hp = butter_highpass(ppg_nf, fs, fc=0.1, order=2)

# 2) IMU：高通 ~0.3 Hz 去重力 + 带通 0.5–15 Hz  ← 得到 acc_bp（与 MATLAB acc_bp 等价）
acc_hp = butter_highpass(acc_raw, fs, fc=0.3, order=2)     # axis=0 已在库内修复
acc_bp = butter_bandpass(acc_hp, fs, 0.5, 15.0, order=3)

# （可选）若以后有 gyro，同样 hp 后再 bp
gyro_bp = None
# if gyro_raw is not None and np.asarray(gyro_raw).size > 0:
#     gyro_hp = butter_highpass(gyro_raw, fs, fc=0.3, order=2)
#     gyro_bp = butter_bandpass(gyro_hp, fs, 0.5, 15.0, order=3)

# ---------- 与“MATLAB→.mat→Python”保持一致的调用 ----------
# 传入 ppg_hp、acc_bp（这样总滤波=两次，效果与 mat 流程一致）
out = run_pipeline(
    ppg_hp,
    acc_bp,
    gyro=gyro_bp,
    fs=fs,
    mains=mains,
    make_demo_plot=True,
    # 可按需调整稳健参数：
    # tpl_N=300, tpl_prominence_scale=0.8, tpl_min_rr_sec=0.35,
    # pre_anc_bp=(0.4, 8.0), anc_mu=8e-4, anc_order=16,
)
