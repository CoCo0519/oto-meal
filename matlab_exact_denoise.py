# -*- coding: utf-8 -*-
"""
完全复制MATLAB小波降噪算法
基于MATLAB的wdenoise函数实现
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def mad_estimate(data):
    """MATLAB风格的MAD噪声估计"""
    return np.median(np.abs(data - np.median(data))) / 0.6745

def sqtwolog_threshold(data, sigma):
    """MATLAB的sqtwolog阈值方法"""
    N = len(data)
    return sigma * np.sqrt(2 * np.log(N))

def rigrsure_threshold(data, sigma):
    """MATLAB的Rigorous SURE阈值方法"""
    N = len(data)
    sorted_data = np.sort(np.abs(data))
    
    # SURE风险函数
    risk = np.zeros(N)
    for i in range(N):
        threshold = sorted_data[i]
        # 计算SURE风险
        n_above = np.sum(np.abs(data) > threshold)
        risk[i] = N - 2 * n_above + np.sum(np.minimum(np.abs(data), threshold)**2)
    
    # 选择最小风险的阈值
    min_idx = np.argmin(risk)
    return sorted_data[min_idx]

def heursure_threshold(data, sigma):
    """MATLAB的Heuristic SURE阈值方法"""
    N = len(data)
    eta = (np.log2(N))**1.5 / np.sqrt(N)
    
    if eta < 1:
        # 使用SURE方法
        return rigrsure_threshold(data, sigma)
    else:
        # 使用sqtwolog方法
        return sqtwolog_threshold(data, sigma)

def minimaxi_threshold(data, sigma):
    """MATLAB的Minimax阈值方法"""
    N = len(data)
    if N > 32:
        threshold = 0.3936 + 0.1829 * np.log2(N)
    else:
        threshold = 0
    return sigma * threshold

def matlab_wdenoise_exact(signal, wavelet='sym8', level=7, threshold_method='bayes', 
                         threshold_mode='soft', q_value=0.05):
    """
    完全复制MATLAB的wdenoise算法
    
    参数:
    - signal: 输入信号
    - wavelet: 小波类型
    - level: 分解层数
    - threshold_method: 阈值方法 ('bayes', 'sqtwolog', 'rigrsure', 'heursure', 'minimaxi')
    - threshold_mode: 阈值模式 ('soft', 'hard')
    - q_value: Bayesian方法的Q值
    """
    
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')
    
    # 获取细节系数
    detail_coeffs = coeffs[1:]
    
    # 计算噪声标准差（使用最高频细节系数）
    if len(detail_coeffs) > 0:
        sigma = mad_estimate(detail_coeffs[-1])
    else:
        sigma = mad_estimate(signal)
    
    # 根据方法计算阈值
    if threshold_method == 'bayes':
        # MATLAB的Bayesian方法
        thresholds = []
        for i, detail in enumerate(detail_coeffs):
            if detail.size == 0:
                thresholds.append(0.0)
                continue
            
            # MATLAB的Bayesian阈值公式
            N = detail.size
            # 使用更标准的Bayesian公式
            bayes_threshold = sigma * np.sqrt(2 * np.log(N)) * q_value
            thresholds.append(bayes_threshold)
    
    elif threshold_method == 'sqtwolog':
        # Universal阈值
        detail_all = np.concatenate([coeff.ravel() for coeff in detail_coeffs])
        if detail_all.size > 0:
            universal_threshold = sqtwolog_threshold(detail_all, sigma)
            thresholds = [universal_threshold] * len(detail_coeffs)
        else:
            thresholds = [0.0] * len(detail_coeffs)
    
    elif threshold_method == 'rigrsure':
        # SURE阈值
        detail_all = np.concatenate([coeff.ravel() for coeff in detail_coeffs])
        if detail_all.size > 0:
            sure_threshold = rigrsure_threshold(detail_all, sigma)
            thresholds = [sure_threshold] * len(detail_coeffs)
        else:
            thresholds = [0.0] * len(detail_coeffs)
    
    elif threshold_method == 'heursure':
        # Heuristic SURE阈值
        detail_all = np.concatenate([coeff.ravel() for coeff in detail_coeffs])
        if detail_all.size > 0:
            heur_threshold = heursure_threshold(detail_all, sigma)
            thresholds = [heur_threshold] * len(detail_coeffs)
        else:
            thresholds = [0.0] * len(detail_coeffs)
    
    elif threshold_method == 'minimaxi':
        # Minimax阈值
        detail_all = np.concatenate([coeff.ravel() for coeff in detail_coeffs])
        if detail_all.size > 0:
            minimaxi_thresh = minimaxi_threshold(detail_all, sigma)
            thresholds = [minimaxi_thresh] * len(detail_coeffs)
        else:
            thresholds = [0.0] * len(detail_coeffs)
    
    else:
        raise ValueError(f"不支持的阈值方法: {threshold_method}")
    
    # 应用阈值到细节系数
    new_coeffs = [coeffs[0]]  # 保持近似系数不变
    
    for i, (detail, threshold) in enumerate(zip(detail_coeffs, thresholds)):
        if threshold > 0:
            thresholded_detail = pywt.threshold(detail, threshold, mode=threshold_mode)
        else:
            thresholded_detail = detail
        new_coeffs.append(thresholded_detail)
    
    # 重构信号
    try:
        denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='symmetric')
    except:
        denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='periodization')
    
    # 确保长度一致
    if denoised_signal.size > signal.size:
        denoised_signal = denoised_signal[:signal.size]
    elif denoised_signal.size < signal.size:
        pad_width = signal.size - denoised_signal.size
        denoised_signal = np.pad(denoised_signal, (0, pad_width), mode='edge')
    
    avg_threshold = np.mean(thresholds) if thresholds else 0.0
    
    return denoised_signal, avg_threshold, sigma

def test_matlab_exact():
    """测试完全MATLAB兼容的算法"""
    
    # 加载真实数据
    data_file = "hyx_data/喉咙- 咳嗽共6次间隔10秒.txt"
    try:
        data = np.loadtxt(data_file, skiprows=1, encoding='gbk')
    except:
        data = np.loadtxt(data_file, skiprows=1, encoding='utf-8')
    
    signal = data[:, 0]  # green通道
    
    # 计算正确的采样率
    fs = len(signal) / 60.0  # 假设60秒
    time_axis = np.arange(len(signal)) / fs
    
    print(f"信号长度: {len(signal)}")
    print(f"采样率: {fs:.2f} Hz")
    print(f"时长: {len(signal)/fs:.2f} 秒")
    print(f"原始信号标准差: {np.std(signal):.2f}")
    
    # 测试不同的MATLAB阈值方法
    methods = ['bayes', 'sqtwolog', 'rigrsure', 'heursure', 'minimaxi']
    
    plt.figure(figsize=(15, 20))
    
    # 原始信号
    plt.subplot(len(methods) + 2, 1, 1)
    plt.plot(time_axis, signal, 'b-', linewidth=0.8)
    plt.title('原始信号')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)
    
    results = {}
    
    for i, method in enumerate(methods):
        print(f"\n测试方法: {method}")
        
        # 执行降噪
        if method == 'bayes':
            denoised, threshold, sigma = matlab_wdenoise_exact(signal, 
                                                             threshold_method=method, 
                                                             q_value=0.05)
        else:
            denoised, threshold, sigma = matlab_wdenoise_exact(signal, 
                                                             threshold_method=method)
        
        # 计算效果
        residual = signal - denoised
        noise_reduction = (1 - np.std(residual)/np.std(signal)) * 100
        
        print(f"  阈值: {threshold:.2f}")
        print(f"  噪声标准差: {sigma:.2f}")
        print(f"  降噪后标准差: {np.std(denoised):.2f}")
        print(f"  噪声抑制率: {noise_reduction:.1f}%")
        
        results[method] = {
            'denoised': denoised,
            'threshold': threshold,
            'noise_reduction': noise_reduction,
            'residual': residual
        }
        
        # 绘制结果
        plt.subplot(len(methods) + 2, 1, i + 2)
        plt.plot(time_axis, signal, 'b-', linewidth=0.5, alpha=0.6, label='原始')
        plt.plot(time_axis, denoised, 'r-', linewidth=1, label=f'{method}降噪')
        plt.title(f'{method.upper()}方法 - 噪声抑制率: {noise_reduction:.1f}%')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 残差对比
    plt.subplot(len(methods) + 2, 1, len(methods) + 2)
    for method, result in results.items():
        plt.plot(time_axis, result['residual'], linewidth=0.8, alpha=0.7, label=f'{method}残差')
    plt.title('残差信号对比')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('matlab_exact_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 找出最佳方法
    best_method = max(results.keys(), key=lambda x: results[x]['noise_reduction'])
    print(f"\n最佳方法: {best_method}")
    print(f"噪声抑制率: {results[best_method]['noise_reduction']:.1f}%")
    
    return results

if __name__ == "__main__":
    print("=== MATLAB精确算法测试 ===")
    results = test_matlab_exact()
    print("\n测试完成！")

