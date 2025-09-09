# -*- coding: utf-8 -*-
"""
批处理某个目录(形如 xxx_data)下的所有 .txt 文件：
- 读取（首行中文表头），列映射：绿光/红外/红光/加速度XYZ
- 在 Python 端先做等价于 MATLAB 的第一次预处理：
    PPG: 工频陷波 + 高通 0.1Hz -> ppg_hp
    ACC: 高通 0.3Hz + 带通 0.5–15Hz -> acc_bp
- 再调用 anc_template_pipeline_100Hz.run_pipeline(...) 完整流程（模板相减 + ANC）
- 生成与 Figure1 一致的 4 行对比图（不弹窗，直接保存为 PNG）
- 输出图片文件夹：results_<name>_<YYYYMMDD-HHMM>，位于本脚本同目录
- 额外：生成一份 summary.csv 汇总每个文件的关键指标与运行参数

用法示例：
    python batch_process_txt.py --dir ./hyx_data --fs 100 --mains 50
"""

import os
import sys
import glob
import argparse
from datetime import datetime
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后端设为非交互，直接保存图片
import matplotlib.pyplot as plt

from anc_template_pipeline_100Hz import (
    run_pipeline, notch_filter, butter_highpass, butter_bandpass
)


def safe_loadtxt(path):
    """优先按 utf-8 读取；若失败则回退 gbk。首行表头需在调用方 skiprows=1。"""
    try:
        return np.loadtxt(path, skiprows=1, encoding="utf-8")
    except TypeError:
        # 老版本 numpy 没 encoding 参数
        with open(path, encoding="utf-8") as f:
            return np.loadtxt(f, skiprows=1)
    except UnicodeDecodeError:
        # 尝试 GBK
        try:
            return np.loadtxt(path, skiprows=1, encoding="gbk")
        except TypeError:
            with open(path, encoding="gbk") as f:
                return np.loadtxt(f, skiprows=1)


def save_comparison_figure(ppg_raw, out, fs, save_path):
    """复刻 anc_template_pipeline_100Hz.plot_comparison 的 4 行图，但不显示、直接保存。"""
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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def extract_name_from_dir(dir_path):
    """从形如 .../xxx_data 中提取 name=xxx；若不满足，退回使用目录名。"""
    base = os.path.basename(os.path.normpath(dir_path))
    if base.endswith("_data") and len(base) > 5:
        return base[:-5]
    return base


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dir", type=str, required=True, help="包含 .txt 的目录（建议形如 xxx_data）")
    ap.add_argument("--fs", type=float, default=100, help="采样率 Hz")
    ap.add_argument("--mains", type=int, default=50, help="工频 (50 或 60)")
    # 稳健参数（与 anc_template_pipeline_100Hz 默认一致，可按需调整）
    ap.add_argument("--tplN", type=int, default=300, help="模板长度 Ntpl")
    ap.add_argument("--prom", type=float, default=0.8, help="峰检门限系数（* std）")
    ap.add_argument("--minrr", type=float, default=0.35, help="最小 RR（秒）")
    ap.add_argument("--prebp", type=str, default="0.4,8.0",
                    help="ANC 前带通，如 '0.4,8.0'；设为 'none' 关闭")
    ap.add_argument("--mu", type=float, default=8e-4, help="ANC NLMS 步长")
    ap.add_argument("--order", type=int, default=16, help="ANC FIR 阶每参考通道")
    ap.add_argument("--channel", type=str, default="green", choices=["green", "ir", "red"],
                    help="选择 PPG 通道：green/ir/red")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.dir)
    if not os.path.isdir(in_dir):
        print(f"[ERROR] 输入目录不存在：{in_dir}")
        sys.exit(1)

    # 构造结果目录名：results_<name>_<YYYYMMDD-HHMM>，位于脚本所在目录
    name = extract_name_from_dir(in_dir)
    now_str = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"results_{name}_{now_str}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 输出目录：{out_dir}")

    # 解析 pre_ANC 带通
    if isinstance(args.prebp, str) and args.prebp.strip().lower() != "none":
        try:
            f1, f2 = [float(x) for x in args.prebp.split(",")]
            prebp = (f1, f2)
        except Exception:
            print("[WARN] --prebp 解析失败，使用默认 (0.4, 8.0)")
            prebp = (0.4, 8.0)
    else:
        prebp = None

    # 收集所有 txt 文件
    txt_list = sorted(glob.glob(os.path.join(in_dir, "*.txt")))
    if not txt_list:
        print(f("[WARN] 目录下未发现 .txt 文件：{in_dir}"))
        sys.exit(0)

    # 汇总 CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    fieldnames = [
        "file", "channel", "N_samples", "fs", "duration_sec",
        "mains", "tplN", "prom", "minrr", "prebp", "mu", "order",
        "peaks_count",
        "residual_std", "anc_std",
        "residual_rms", "anc_rms",
        "residual_ptp", "anc_ptp",
        "acc_energy_mean", "acc_energy_max",
        "gyro_present", "status", "error_message",
        "png_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        ok, fail = 0, 0
        for txt_path in txt_list:
            base = os.path.splitext(os.path.basename(txt_path))[0]
            save_path = os.path.join(out_dir, f"{base}.png")
            row = {
                "file": base,
                "channel": args.channel,
                "fs": float(args.fs),
                "mains": int(args.mains),
                "tplN": int(args.tplN),
                "prom": float(args.prom),
                "minrr": float(args.minrr),
                "prebp": "none" if prebp is None else f"{prebp[0]},{prebp[1]}",
                "mu": float(args.mu),
                "order": int(args.order),
                "png_path": save_path,
            }
            try:
                # 读取（跳过中文表头）
                data = safe_loadtxt(txt_path)
                N = data.shape[0]
                row["N_samples"] = int(N)
                row["duration_sec"] = float(N / args.fs)

                # 列映射
                ppg_green = data[:, 0]
                ppg_ir    = data[:, 1]
                ppg_red   = data[:, 2]
                acc_raw   = data[:, 3:6]    # X, Y, Z
                gyro_bp   = None            # 默认没有陀螺

                # 选择 PPG 通道
                if args.channel == "green":
                    ppg_raw = ppg_green
                elif args.channel == "ir":
                    ppg_raw = ppg_ir
                else:
                    ppg_raw = ppg_red

                # ========== 第一次预处理（等价 MATLAB） ==========
                # 1) PPG：工频陷波 + 高通 0.1 Hz
                ppg_nf = notch_filter(ppg_raw, args.fs, mains=args.mains, Q=30)
                ppg_hp = butter_highpass(ppg_nf, args.fs, fc=0.1, order=2)

                # 2) ACC：高通 0.3 Hz + 带通 0.5–15 Hz
                acc_hp = butter_highpass(acc_raw, args.fs, fc=0.3, order=2)
                acc_bp = butter_bandpass(acc_hp, args.fs, 0.5, 15.0, order=3)

                # ========== run_pipeline（第二次滤波+模板相减+ANC） ==========
                out = run_pipeline(
                    ppg_hp,
                    acc_bp,
                    gyro=gyro_bp,
                    fs=args.fs,
                    mains=args.mains,
                    make_demo_plot=False,  # 我们自己保存图
                    tpl_N=args.tplN,
                    tpl_prominence_scale=args.prom,
                    tpl_min_rr_sec=args.minrr,
                    pre_anc_bp=prebp,
                    anc_mu=args.mu,
                    anc_order=args.order,
                )

                # 生成并保存图像（Figure1等效）
                save_comparison_figure(ppg_raw, out, args.fs, save_path)

                # --------- 计算指标并写入 CSV ----------
                peaks_count = int(len(out["peaks"]))
                res = out["ppg_residual"]
                anc = out["ppg_anc"]
                acc_energy = out["imu_energy"]["acc"]
                gyro_present = out["imu_energy"]["gyro"] is not None

                row.update({
                    "peaks_count": peaks_count,
                    "residual_std": float(np.std(res)),
                    "anc_std": float(np.std(anc)),
                    "residual_rms": float(np.sqrt(np.mean(res ** 2))),
                    "anc_rms": float(np.sqrt(np.mean(anc ** 2))),
                    "residual_ptp": float(np.ptp(res)),
                    "anc_ptp": float(np.ptp(anc)),
                    "acc_energy_mean": float(np.mean(acc_energy)) if acc_energy is not None else None,
                    "acc_energy_max": float(np.max(acc_energy)) if acc_energy is not None else None,
                    "gyro_present": bool(gyro_present),
                    "status": "OK",
                    "error_message": "",
                })
                writer.writerow(row)
                ok += 1
                print(f"[OK] {base} -> {save_path}")

            except Exception as e:
                row.update({
                    "N_samples": None,
                    "duration_sec": None,
                    "peaks_count": None,
                    "residual_std": None,
                    "anc_std": None,
                    "residual_rms": None,
                    "anc_rms": None,
                    "residual_ptp": None,
                    "anc_ptp": None,
                    "acc_energy_mean": None,
                    "acc_energy_max": None,
                    "gyro_present": None,
                    "status": "FAIL",
                    "error_message": str(e),
                })
                writer.writerow(row)
                fail += 1
                print(f"[FAIL] {base}: {e}")

    print(f"\n[SUMMARY] 输出目录: {out_dir}")
    print(f"[SUMMARY] 汇总表: {csv_path}")
    print(f"[SUMMARY] 成功: {ok}, 失败: {fail}")


if __name__ == "__main__":
    main()
