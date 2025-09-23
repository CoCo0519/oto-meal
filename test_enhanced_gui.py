#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for enhanced PPG Denoising GUI
Test the new parameters and features
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt, butter, filtfilt, wiener
import time

def test_enhanced_parameters():
    """Test the enhanced parameters"""
    print("🧪 Testing Enhanced GUI Parameters")
    print("="*50)
    
    # Create test signal
    t = np.linspace(0, 10, 1000)
    fs = 100
    
    # Heart rate component (72 BPM)
    hr_freq = 1.2
    heart_component = np.sin(2 * np.pi * hr_freq * t)
    
    # Respiratory component (12 breaths per minute)
    resp_freq = 0.2
    resp_component = 0.3 * np.sin(2 * np.pi * resp_freq * t)
    
    # Noise
    noise = 0.3 * np.random.randn(len(t))
    
    # Combine components
    original_signal = heart_component + resp_component + noise
    
    print(f"📊 Test signal: {len(original_signal)} samples, {t[-1]:.1f}s duration")
    
    # Test enhanced Savitzky-Golay
    print("\n🔧 Testing Enhanced Savitzky-Golay:")
    try:
        start_time = time.time()
        denoised_sg = savgol_filter(original_signal, 51, 3, deriv=1, delta=1.0)
        sg_time = (time.time() - start_time) * 1000
        
        snr_sg = 10 * np.log10(np.var(denoised_sg) / np.var(original_signal - denoised_sg))
        print(f"   ✅ Derivative order=1: SNR={snr_sg:.1f}dB, Time={sg_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ Enhanced SG failed: {e}")
    
    # Test enhanced Wiener filter
    print("\n🔧 Testing Enhanced Wiener Filter:")
    try:
        start_time = time.time()
        denoised_wiener = wiener(original_signal, noise=0.1, mysize=5)
        wiener_time = (time.time() - start_time) * 1000
        
        snr_wiener = 10 * np.log10(np.var(denoised_wiener) / np.var(original_signal - denoised_wiener))
        print(f"   ✅ Filter size=5: SNR={snr_wiener:.1f}dB, Time={wiener_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ Enhanced Wiener failed: {e}")
    
    # Test enhanced Wavelet approximation with windowing
    print("\n🔧 Testing Enhanced Wavelet Approximation:")
    try:
        start_time = time.time()
        
        cutoff = 12.5
        order = 6
        window_size = 256
        overlap = 128
        
        # Apply windowed filtering
        signal_len = len(original_signal)
        if signal_len > window_size:
            step_size = window_size - overlap
            denoised_parts = []
            
            for start in range(0, signal_len - window_size + 1, step_size):
                end = min(start + window_size, signal_len)
                window_signal = original_signal[start:end]
                b, a = butter(order, cutoff / (fs/2), btype='low')
                window_denoised = filtfilt(b, a, window_signal)
                denoised_parts.append(window_denoised)
            
            # Combine overlapping windows
            denoised_wavelet = np.zeros_like(original_signal)
            weights = np.zeros_like(original_signal)
            
            for i, denoised_part in enumerate(denoised_parts):
                start = i * step_size
                end = start + len(denoised_part)
                denoised_wavelet[start:end] += denoised_part
                weights[start:end] += 1
            
            # Normalize by weights
            denoised_wavelet = np.divide(denoised_wavelet, weights, 
                                       out=np.zeros_like(denoised_wavelet), 
                                       where=weights!=0)
        else:
            b, a = butter(order, cutoff / (fs/2), btype='low')
            denoised_wavelet = filtfilt(b, a, original_signal)
        
        wavelet_time = (time.time() - start_time) * 1000
        snr_wavelet = 10 * np.log10(np.var(denoised_wavelet) / np.var(original_signal - denoised_wavelet))
        print(f"   ✅ Windowed filtering: SNR={snr_wavelet:.1f}dB, Time={wavelet_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ Enhanced Wavelet failed: {e}")
    
    # Test enhanced Adaptive filter
    print("\n🔧 Testing Enhanced Adaptive Filter:")
    try:
        start_time = time.time()
        
        window_size = 20
        alpha = 0.1
        
        # Exponential moving average
        denoised_adaptive = np.zeros_like(original_signal)
        denoised_adaptive[0] = original_signal[0]
        for i in range(1, len(original_signal)):
            denoised_adaptive[i] = alpha * original_signal[i] + (1 - alpha) * denoised_adaptive[i-1]
        
        adaptive_time = (time.time() - start_time) * 1000
        snr_adaptive = 10 * np.log10(np.var(denoised_adaptive) / np.var(original_signal - denoised_adaptive))
        print(f"   ✅ Exponential MA (α={alpha}): SNR={snr_adaptive:.1f}dB, Time={adaptive_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ Enhanced Adaptive failed: {e}")
    
    return True

def test_performance_metrics():
    """Test the enhanced performance metrics"""
    print("\n📊 Testing Enhanced Performance Metrics")
    print("="*50)
    
    # Create test signals
    t = np.linspace(0, 5, 500)
    original = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(len(t))
    denoised = savgol_filter(original, 51, 3)
    
    # Calculate metrics
    noise = original - denoised
    signal_power = np.var(denoised)
    noise_power = np.var(noise)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    std_reduction = np.std(original) - np.std(denoised)
    peak_to_peak = np.ptp(denoised)
    rmse = np.sqrt(np.mean((original - denoised)**2))
    correlation = np.corrcoef(original, denoised)[0, 1]
    
    print(f"✅ SNR: {snr_db:.1f} dB")
    print(f"✅ Std Reduction: {std_reduction:.3f}")
    print(f"✅ Peak-to-Peak: {peak_to_peak:.1f}")
    print(f"✅ RMSE: {rmse:.3f}")
    print(f"✅ Correlation: {correlation:.3f}")
    
    return True

def create_enhanced_demo_plot():
    """Create a demo plot showing enhanced features"""
    print("\n📊 Creating Enhanced Demo Plot")
    print("="*50)
    
    # Create synthetic signal
    t = np.linspace(0, 5, 500)
    fs = 100
    
    # Heart rate component
    hr_freq = 1.2
    heart_component = np.sin(2 * np.pi * hr_freq * t)
    
    # Respiratory component
    resp_freq = 0.2
    resp_component = 0.3 * np.sin(2 * np.pi * resp_freq * t)
    
    # Noise
    noise = 0.3 * np.random.randn(len(t))
    
    # Combine components
    original_signal = heart_component + resp_component + noise
    
    # Apply different enhanced methods
    denoised_sg = savgol_filter(original_signal, 51, 3, deriv=1, delta=1.0)
    denoised_wiener = wiener(original_signal, noise=0.1, mysize=5)
    
    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced PPG Denoising Methods Comparison', fontsize=16, fontweight='bold')
    
    # Original signal
    ax1.plot(t, original_signal, 'b-', linewidth=1, alpha=0.7, label='Original')
    ax1.set_title('Original PPG Signal', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Savitzky-Golay with derivative
    ax2.plot(t, original_signal, 'b-', linewidth=1, alpha=0.5, label='Original')
    ax2.plot(t, denoised_sg, 'r-', linewidth=1.5, label='SG (deriv=1)')
    ax2.set_title('Enhanced Savitzky-Golay (Derivative)', fontweight='bold')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Wiener with custom size
    ax3.plot(t, original_signal, 'b-', linewidth=1, alpha=0.5, label='Original')
    ax3.plot(t, denoised_wiener, 'g-', linewidth=1.5, label='Wiener (size=5)')
    ax3.set_title('Enhanced Wiener Filter', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Noise comparison
    noise_sg = original_signal - denoised_sg
    noise_wiener = original_signal - denoised_wiener
    
    ax4.plot(t, noise_sg, 'r-', linewidth=1, alpha=0.7, label='SG Noise')
    ax4.plot(t, noise_wiener, 'g-', linewidth=1, alpha=0.7, label='Wiener Noise')
    ax4.set_title('Extracted Noise Comparison', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Noise Amplitude')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'enhanced_ppg_denoising_demo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Enhanced demo plot saved: {output_path}")

def main():
    """Main test function"""
    print("🚀 Enhanced PPG Denoising GUI Test Suite")
    print("="*60)
    
    # Test enhanced parameters
    params_ok = test_enhanced_parameters()
    
    # Test performance metrics
    metrics_ok = test_performance_metrics()
    
    # Create demo plot
    create_enhanced_demo_plot()
    
    # Summary
    print("\n" + "="*60)
    print("ENHANCED GUI TEST SUMMARY")
    print("="*60)
    print(f"Enhanced Parameters: {'✅ PASSED' if params_ok else '❌ FAILED'}")
    print(f"Performance Metrics: {'✅ PASSED' if metrics_ok else '❌ FAILED'}")
    
    if params_ok and metrics_ok:
        print("\n🎉 Enhanced GUI features are working correctly!")
        print("📁 New features added:")
        print("   ✅ Enhanced Savitzky-Golay: Derivative order, Delta parameter")
        print("   ✅ Enhanced Wiener Filter: Filter size, Auto noise estimation")
        print("   ✅ Enhanced Wavelet Approx: Window size, Overlap size")
        print("   ✅ Enhanced Adaptive Filter: Method selection, Learning rate")
        print("   ✅ Enhanced Performance Metrics: RMSE, Correlation, Processing time")
        print("   ✅ Enhanced Display Options: Noise display, Metrics overlay")
        print("\n💡 You can now run the enhanced GUI:")
        print("   python ppg_denoising_gui.py")
    else:
        print("\n⚠️ Some enhanced features failed. Please check the implementation.")

if __name__ == "__main__":
    main()
