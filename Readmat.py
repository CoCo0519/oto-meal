# -*- coding: utf-8 -*-
"""
从 MATLAB 导出的 preprocessed_for_python.mat 读取数据，
调用稳健版 run_pipeline 完成：心搏模板相减 + ANC + 对比图。
可在命令行调参：
    python run_pipeline_from_mat.py --mat preprocessed_for_python.mat \
        --fs 100 --mains 50 \
        --tplN 300 --prom 0.8 --minrr 0.35 \
        --mu 8e-4 --order 16 --prebp 0.4,8.0 --plot 1
"""

import argparse
import numpy as np
import scipy.io as sio
from anc_template_pipeline_100Hz import run_pipeline

def _arr_or_none(val):
    if val is None:
        return None
    arr = np.array(val, dtype=float)
    if arr.size == 0:
        return None
    return arr

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mat",   type=str, default="preprocessed_for_python.mat", help="MATLAB 预处理输出 .mat 路径")
    ap.add_argument("--fs",    type=float, default=None, help="采样率；为空则使用 .mat 中的 fs")
    ap.add_argument("--mains", type=int, default=None, help="工频 50/60；为空则使用 .mat 中的 mains")
    # 模板相减参数
    ap.add_argument("--tplN",  type=int,   default=300,  help="模板标准长度 Ntpl")
    ap.add_argument("--prom",  type=float, default=0.8,  help="峰检门限系数（* std）")
    ap.add_argument("--minrr", type=float, default=0.35, help="最小 RR（秒）")
    # ANC 参数
    ap.add_argument("--mu",    type=float, default=8e-4, help="NLMS 步长")
    ap.add_argument("--order", type=int,   default=16,   help="每参考通道 FIR 阶数")
    ap.add_argument("--prebp", type=str,   default="0.4,8.0",
                    help="ANC 前带通，如 '0.4,8.0'；设为 'none' 关闭")
    ap.add_argument("--plot",  type=int,   default=1,    help="是否绘图（1/0）")
    return ap.parse_args()

def main():
    args = parse_args()

    # 读取 .mat（mat_struct；属性访问）
    mat = sio.loadmat(args.mat, squeeze_me=True, struct_as_record=False)

    # 基本量
    ppg   = np.array(mat["ppg_hp"]).ravel()
    fs    = int(np.array(mat["fs"]).squeeze())    if args.fs    is None else float(args.fs)
    mains = int(np.array(mat["mains"]).squeeze()) if args.mains is None else int(args.mains)

    imuOut = mat["imuOut"]
    acc_bp = _arr_or_none(getattr(imuOut, "acc_bp", None))
    if acc_bp is None:
        raise ValueError("imuOut.acc_bp 缺失或为空，请检查 MATLAB 预处理导出。")
    gyro_bp = _arr_or_none(getattr(imuOut, "gyro_bp", None))

    # 解析 pre_anc 带通
    if isinstance(args.prebp, str) and args.prebp.strip().lower() != "none":
        f1, f2 = [float(x) for x in args.prebp.split(",")]
        prebp = (f1, f2)
    else:
        prebp = None

    # 运行管道
    out = run_pipeline(
        ppg, acc_bp, gyro=gyro_bp, fs=fs, mains=mains,
        make_demo_plot=bool(args.plot),
        tpl_N=args.tplN, tpl_prominence_scale=args.prom, tpl_min_rr_sec=args.minrr,
        pre_anc_bp=prebp, anc_mu=args.mu, anc_order=args.order,
    )

    # 简要输出关键结果信息
    print("OK. keys:", list(out.keys()))
    print(f"peaks={len(out['peaks'])}, template_len={len(out['ppg_template'])}")
    print(f"ppg arrays: hp={out['ppg_hp'].shape}, residual={out['ppg_residual'].shape}, anc={out['ppg_anc'].shape}")
    if out['imu_energy']['gyro'] is None:
        print("gyro: None")
    else:
        print("gyro: present")

if __name__ == "__main__":
    main()
