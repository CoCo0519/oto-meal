#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wavelet-based PPG denoising utility derived from ppg_signal_analysis.py.
Loads wavelet parameters from a JSON file, performs wavelet approximation
denoising on the selected channel, and exports original, denoised, and
comparison plots as PNG images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
# 修复中文字体显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

try:
    import pywt
except ImportError as exc:
    raise SystemExit(
        "pywt is required for wavelet denoising. Install it with 'pip install PyWavelets'."
    ) from exc


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def auto_detect_sampling_rate(data_length: int, expected_duration: float = 60.0) -> float:
    """自动检测采样率"""
    return data_length / expected_duration

def load_ppg_txt(path: Path, fs: float = None, expected_duration: float = 60.0) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        data = np.loadtxt(path, skiprows=1, encoding="utf-8")
    except UnicodeDecodeError:
        data = np.loadtxt(path, skiprows=1, encoding="gbk")
    except TypeError:
        with open(path, "r", encoding="utf-8") as handle:
            data = np.loadtxt(handle, skiprows=1)
    if data.ndim == 1 or data.shape[1] < 3:
        raise ValueError("Expected txt file with at least three PPG columns (green, ir, red).")
    
    # 如果没有指定采样率，自动检测
    if fs is None:
        fs = auto_detect_sampling_rate(data.shape[0], expected_duration)
        print(f"自动检测采样率: {fs:.2f} Hz")
    
    duration = data.shape[0] / fs

    # 支持所有PPG通道和IMU数据
    result = {
        "green": data[:, 0],
        "ir": data[:, 1],
        "red": data[:, 2],
        "duration": duration,
        "samples": data.shape[0],
    }

    # 如果有ACC数据（通常在第4-6列）
    if data.shape[1] >= 6:
        result["acc_x"] = data[:, 3]
        result["acc_y"] = data[:, 4]
        result["acc_z"] = data[:, 5]
        result["imu_present"] = True
    else:
        result["imu_present"] = False

    # 如果有GYRO数据（通常在第7-9列）
    if data.shape[1] >= 9:
        result["gyro_x"] = data[:, 6]
        result["gyro_y"] = data[:, 7]
        result["gyro_z"] = data[:, 8]
        result["gyro_present"] = True
    else:
        result["gyro_present"] = False

    return result


def matlab_bayes_threshold(signal: np.ndarray, wavelet: str, level: int, q_value: float = 0.05) -> list:
    """MATLAB兼容的Bayesian阈值计算"""
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode='symmetric')
    
    # 计算每个层级的阈值
    thresholds = []
    for i in range(1, len(coeffs)):  # 跳过近似系数
        detail = coeffs[i]
        if detail.size == 0:
            thresholds.append(0.0)
            continue
            
        # 计算噪声标准差
        sigma = np.median(np.abs(detail)) / 0.6745
        if sigma == 0:
            thresholds.append(0.0)
            continue
            
        # MATLAB Bayesian阈值公式
        # T = sigma^2 / sqrt(log(N)) * q_value
        N = detail.size
        threshold = sigma * np.sqrt(2 * np.log(N)) * q_value
        thresholds.append(threshold)
    
    return thresholds

def compute_threshold(details: list[np.ndarray], strategy: str, scale: float, manual_value: float | None, 
                     q_value: float = 0.05, noise_estimate: str = "level_independent") -> float:
    """计算阈值，支持MATLAB兼容的参数"""
    if strategy == "manual":
        if manual_value is None:
            raise ValueError("Manual threshold selected but 'manual_value' missing in config.")
        return float(manual_value)
    
    # Level-independent approach (MATLAB default)
    detail_coeffs = np.concatenate([coeff.ravel() for coeff in details], axis=0)
    if detail_coeffs.size == 0:
        return 0.0
    
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    universal = sigma * np.sqrt(2.0 * np.log(detail_coeffs.size))

    if strategy == "bayes":
        # MATLAB Bayesian threshold: T = sigma * sqrt(2*log(N)) * q_value
        N = detail_coeffs.size
        bayes_threshold = sigma * np.sqrt(2 * np.log(N)) * q_value
        return float(scale) * bayes_threshold
    else:
        # Universal threshold
        return float(scale) * universal


def wavelet_denoise(signal: np.ndarray, cfg: Dict[str, object]) -> Tuple[np.ndarray, float]:
    """MATLAB兼容的小波降噪函数 - 层级相关Bayesian方法"""
    wavelet = cfg.get("wavelet", "sym8")
    level = int(cfg.get("decomposition_level", 7))
    mode = cfg.get("mode", "symmetric")
    threshold_cfg = cfg.get("threshold", {})
    strategy = threshold_cfg.get("strategy", "bayes")
    q_value = threshold_cfg.get("q_value", 0.05)
    threshold_mode = threshold_cfg.get("threshold_mode", "soft")
    
    # 执行小波分解
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode=mode)
    
    if strategy == "bayes":
        # MATLAB层级相关Bayesian降噪 - 最佳效果
        new_coeffs = [coeffs[0]]  # 近似系数保持不变
        
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            if detail.size == 0:
                new_coeffs.append(detail)
                continue
            
            # 计算该层级的噪声估计
            sigma = np.median(np.abs(detail)) / 0.6745
            
            if sigma == 0:
                new_coeffs.append(detail)
                continue
            
            # 层级相关的Bayesian阈值 - 关键算法
            N = detail.size
            # 使用层级相关因子，确保高层级（低频）更保守
            level_factor = 0.5 + 0.5 * (i / len(coeffs))  # 0.5到1.0之间
            threshold = sigma * np.sqrt(2 * np.log(N)) * q_value * level_factor
            
            # 应用阈值
            if threshold > 0:
                thresholded_detail = pywt.threshold(detail, threshold, mode=threshold_mode)
            else:
                thresholded_detail = detail
            
            new_coeffs.append(thresholded_detail)
        
        avg_threshold = q_value * 0.75  # 平均阈值估算
    else:
        # Universal阈值方法
        detail_coeffs = coeffs[1:]
        detail_all = np.concatenate([coeff.ravel() for coeff in detail_coeffs])
        
        if detail_all.size > 0:
            sigma = np.median(np.abs(detail_all)) / 0.6745
            universal_threshold = sigma * np.sqrt(2 * np.log(detail_all.size))
            
            new_coeffs = [coeffs[0]]  # 近似系数保持不变
            for i in range(1, len(coeffs)):
                new_coeffs.append(pywt.threshold(coeffs[i], universal_threshold, mode=threshold_mode))
            
            avg_threshold = universal_threshold
        else:
            new_coeffs = coeffs
            avg_threshold = 0.0

    # 重构信号
    try:
        reconstructed = pywt.waverec(new_coeffs, wavelet=wavelet, mode=mode)
    except Exception as e:
        print(f"重构警告: {e}")
        # 如果重构失败，尝试调整边界处理
        reconstructed = pywt.waverec(new_coeffs, wavelet=wavelet, mode='periodization')
    
    # 确保输出长度与输入一致
    if reconstructed.size > signal.size:
        reconstructed = reconstructed[:signal.size]
    elif reconstructed.size < signal.size:
        # 使用边缘填充
        pad_width = signal.size - reconstructed.size
        reconstructed = np.pad(reconstructed, (0, pad_width), mode='edge')
    
    return reconstructed, avg_threshold


def resolve_figsize(cfg: Dict[str, object]) -> Tuple[float, float]:
    plotting = cfg.get("plotting", {})
    size = plotting.get("figsize", [12, 6])
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return float(size[0]), float(size[1])
        except (TypeError, ValueError):
            pass
    return 12.0, 6.0


def build_time_axis(length: int, fs: float) -> np.ndarray:
    return np.arange(length) / fs


def save_single_plot(path: Path, time_axis: np.ndarray, signal: np.ndarray, title: str, ylabel: str, figsize: Tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, signal, linewidth=1.2, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(path: Path, time_axis: np.ndarray, original: np.ndarray, denoised: np.ndarray, figsize: Tuple[float, float]) -> float:
    residual = original - denoised
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.4), sharex=True)
    axes[0].plot(time_axis, original, linewidth=1.0, alpha=0.6, label="Original", color="#1f77b4")
    axes[0].plot(time_axis, denoised, linewidth=1.4, label="Wavelet denoised", color="#d62728")
    axes[0].set_title("Original vs. Wavelet Denoised")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, residual, linewidth=1.0, color="#2ca02c")
    axes[1].set_title("Residual (Original - Denoised)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    snr_improvement = 20.0 * np.log10(
        (np.std(denoised) + 1e-12) / (np.std(residual) + 1e-12)
    )
    return snr_improvement


def process_all_channels(data: Dict[str, np.ndarray], config: Dict[str, object], output_dir: Path, input_name: str):
    """处理所有PPG通道和IMU数据"""
    fs = config.get("fs", 100)
    method = config.get("method", "wavelet")
    results = {}

    # 处理所有PPG通道
    ppg_channels = ["green", "ir", "red"]
    for channel in ppg_channels:
        if channel in data:
            signal = data[channel]
            print(f"Processing {channel} channel with {method} method...")

            if method == "wavelet":
                denoised_signal, threshold = wavelet_denoise(signal, config)
            elif method == "bayes":
                denoised_signal, threshold = wavelet_denoise(signal, config)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # 计算SNR改善
            if 'denoised_signal' in locals():
                residual = signal - denoised_signal
                snr_original = 20 * np.log10(np.std(signal) / (np.std(residual) + 1e-12))
                snr_denoised = 20 * np.log10(np.std(denoised_signal) / (np.std(residual) + 1e-12))
                snr_improvement = snr_denoised - snr_original
            else:
                snr_improvement = 0

            results[channel] = {
                "original": signal,
                "denoised": denoised_signal,
                "threshold": threshold,
                "snr_improvement": snr_improvement
            }

            # 生成图像
            time_axis = build_time_axis(signal.size, fs)
            figsize = resolve_figsize(config)

            # 原始信号
            original_path = output_dir / f"{input_name}_{channel}_original.png"
            save_single_plot(original_path, time_axis, signal, f"Original {channel.upper()} Signal", "Amplitude", figsize)

            # 降噪信号
            denoised_path = output_dir / f"{input_name}_{channel}_denoised.png"
            save_single_plot(denoised_path, time_axis, denoised_signal, f"Denoised {channel.upper()} Signal", "Amplitude", figsize)

            # 对比图
            comparison_path = output_dir / f"{input_name}_{channel}_comparison.png"
            save_comparison_plot(comparison_path, time_axis, signal, denoised_signal, figsize)

    # 处理IMU数据（如果存在）
    if data.get("imu_present", False):
        print("Processing IMU data...")
        imu_results = process_imu_data(data, config, output_dir, input_name)
        results["imu"] = imu_results

    return results

def process_imu_data(data: Dict[str, np.ndarray], config: Dict[str, object], output_dir: Path, input_name: str):
    """处理IMU数据（ACC和GYRO）"""
    fs = config.get("fs", 100)
    method = config.get("method", "wavelet")
    imu_results = {}

    # 处理ACC数据
    if "acc_x" in data:
        print("Processing ACC data...")
        acc_signals = [data["acc_x"], data["acc_y"], data["acc_z"]]

        acc_denoised = []
        for i, signal in enumerate(acc_signals):
            if method == "wavelet":
                denoised_signal, threshold = wavelet_denoise(signal, config)
            elif method == "bayes":
                denoised_signal, threshold = wavelet_denoise(signal, config)
            else:
                denoised_signal = signal

            acc_denoised.append(denoised_signal)

        imu_results["acc"] = {
            "original": acc_signals,
            "denoised": acc_denoised
        }

        # 生成ACC对比图
        time_axis = build_time_axis(len(acc_signals[0]), fs)
        figsize = resolve_figsize(config)

        acc_comparison_path = output_dir / f"{input_name}_acc_comparison.png"
        save_imu_comparison_plot(acc_comparison_path, time_axis, acc_signals, acc_denoised, figsize, "ACC")

    # 处理GYRO数据
    if data.get("gyro_present", False):
        print("Processing GYRO data...")
        gyro_signals = [data["gyro_x"], data["gyro_y"], data["gyro_z"]]

        gyro_denoised = []
        for i, signal in enumerate(gyro_signals):
            if method == "wavelet":
                denoised_signal, threshold = wavelet_denoise(signal, config)
            elif method == "bayes":
                denoised_signal, threshold = wavelet_denoise(signal, config)
            else:
                denoised_signal = signal

            gyro_denoised.append(denoised_signal)

        imu_results["gyro"] = {
            "original": gyro_signals,
            "denoised": gyro_denoised
        }

        # 生成GYRO对比图
        time_axis = build_time_axis(len(gyro_signals[0]), fs)
        figsize = resolve_figsize(config)

        gyro_comparison_path = output_dir / f"{input_name}_gyro_comparison.png"
        save_imu_comparison_plot(gyro_comparison_path, time_axis, gyro_signals, gyro_denoised, figsize, "GYRO")

    return imu_results

def save_imu_comparison_plot(path: Path, time_axis: np.ndarray, original_signals: list, denoised_signals: list, figsize: Tuple[float, float], signal_type: str):
    """保存IMU数据对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(figsize[0], figsize[1] * 1.5), sharex=True)

    axis_labels = ['X', 'Y', 'Z']

    for i in range(3):
        # 原始信号
        axes[i, 0].plot(time_axis, original_signals[i], linewidth=1.0, alpha=0.6, label=f"Original {axis_labels[i]}", color="#1f77b4")
        axes[i, 0].set_ylabel(f"{signal_type} {axis_labels[i]}")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # 降噪信号
        axes[i, 1].plot(time_axis, denoised_signals[i], linewidth=1.2, label=f"Denoised {axis_labels[i]}", color="#d62728")
        axes[i, 1].set_ylabel(f"{signal_type} {axis_labels[i]}")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")

    fig.suptitle(f"Original vs Denoised {signal_type} Signals")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-channel PPG and IMU denoising with wavelet and Bayesian methods.")
    parser.add_argument("--input", "-i", required=True, help="Path to input txt file.")
    parser.add_argument("--config", "-j", default="wavelet_denoise_config.json", help="Path to JSON config.")
    parser.add_argument("--method", choices=["wavelet", "bayes"], default="wavelet", help="Denoising method: wavelet or bayes.")
    parser.add_argument("--output-dir", "-o", help="Directory to store generated images.")
    parser.add_argument("--prefix", help="Filename prefix for exported images.")
    parser.add_argument("--fs", type=float, help="Override sampling rate in Hz.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    config = load_config(config_path)
    fs = float(args.fs) if args.fs is not None else float(config.get("fs", 100))

    # 更新配置中的方法
    config["method"] = args.method

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent / f"{input_path.stem}_multi_denoise"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing with {args.method} method...")
    data = load_ppg_txt(input_path, fs)

    results = process_all_channels(data, config, output_dir, input_path.stem)

    # 生成汇总报告
    summary_path = output_dir / f"{input_path.stem}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Multi-channel Denoising Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Sampling rate: {fs} Hz\n")
        f.write(f"Duration: {data['duration']:.2f} s\n")
        f.write(f"Samples: {data['samples']}\n")
        f.write(f"PPG channels processed: {', '.join([ch for ch in ['green', 'ir', 'red'] if ch in data])}\n")
        f.write(f"IMU data present: {data.get('imu_present', False)}\n")
        f.write(f"GYRO data present: {data.get('gyro_present', False)}\n\n")

        f.write("PPG Channel Results:\n")
        f.write("-" * 20 + "\n")
        for channel, result in results.items():
            if channel in ["green", "ir", "red"]:
                f.write(f"{channel.upper()} channel:\n")
                f.write(f"  SNR improvement: {result['snr_improvement']:.2f} dB\n")
                f.write(f"  Threshold: {result['threshold']:.6f}\n")

        f.write(f"\nOutput directory: {output_dir}\n")

    print("Processing complete!")
    print(f"Input file            : {input_path}")
    print(f"Method                : {args.method}")
    print(f"Sampling rate (Hz)    : {fs}")
    print(f"PPG channels processed: {', '.join([ch for ch in ['green', 'ir', 'red'] if ch in data])}")
    print(f"IMU data processed    : {data.get('imu_present', False)}")
    print(f"GYRO data processed   : {data.get('gyro_present', False)}")
    print(f"Summary report        : {summary_path}")
    print(f"Output directory      : {output_dir}")


if __name__ == "__main__":
    main()
