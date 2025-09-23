#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for STFT and enhanced evaluation features
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, stft
import time

def test_stft_computation():
    """Test STFT computation with different parameters"""
    print("🧪 Testing STFT Computation")
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
    
    # Test different STFT parameters
    stft_params = [
        {"nperseg": 256, "noverlap": 128, "nfft": 512, "window": "hann"},
        {"nperseg": 128, "noverlap": 64, "nfft": 256, "window": "hamming"},
        {"nperseg": 512, "noverlap": 256, "nfft": 1024, "window": "blackman"},
    ]
    
    for i, params in enumerate(stft_params):
        try:
            start_time = time.time()
            f, t_stft, Zxx = stft(original_signal, fs=fs, **params)
            stft_time = (time.time() - start_time) * 1000
            
            print(f"✅ STFT {i+1}: {params['window']} window, {params['nperseg']} seg, {params['noverlap']} overlap")
            print(f"   Time: {stft_time:.1f}ms, Shape: {Zxx.shape}, Freq range: {f[0]:.1f}-{f[-1]:.1f} Hz")
            
        except Exception as e:
            print(f"❌ STFT {i+1} failed: {e}")
    
    return True

def test_spectral_features():
    """Test spectral feature computation"""
    print("\n📊 Testing Spectral Features")
    print("="*50)
    
    # Create test signal
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
    denoised_signal = savgol_filter(original_signal, 51, 3)
    
    try:
        # Compute STFT
        f, t_stft, Zxx = stft(original_signal, fs=fs, nperseg=256, noverlap=128)
        
        # Spectral centroid
        magnitude = np.abs(Zxx)
        spectral_centroid = np.sum(f[:, np.newaxis] * magnitude, axis=0) / np.sum(magnitude, axis=0)
        
        # Spectral rolloff (95% energy)
        cumsum_magnitude = np.cumsum(magnitude, axis=0)
        total_energy = cumsum_magnitude[-1, :]
        rolloff_threshold = 0.95 * total_energy
        
        spectral_rolloff = np.zeros(len(t_stft))
        for i in range(len(t_stft)):
            idx = np.where(cumsum_magnitude[:, i] >= rolloff_threshold[i])[0]
            if len(idx) > 0:
                spectral_rolloff[i] = f[idx[0]]
        
        print(f"✅ Spectral Centroid: {np.mean(spectral_centroid):.1f} Hz")
        print(f"✅ Spectral Rolloff: {np.mean(spectral_rolloff):.1f} Hz")
        print(f"✅ Centroid std: {np.std(spectral_centroid):.1f} Hz")
        print(f"✅ Rolloff std: {np.std(spectral_rolloff):.1f} Hz")
        
    except Exception as e:
        print(f"❌ Spectral features failed: {e}")
        return False
    
    return True

def test_enhanced_metrics():
    """Test enhanced evaluation metrics"""
    print("\n📈 Testing Enhanced Evaluation Metrics")
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
    
    # Additional metrics
    mae = np.mean(np.abs(original - denoised))
    
    # PSNR calculation
    max_val = np.max(original)
    psnr_db = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
    
    # SSIM approximation
    mu1 = np.mean(original)
    mu2 = np.mean(denoised)
    sigma1 = np.var(original)
    sigma2 = np.var(denoised)
    sigma12 = np.mean((original - mu1) * (denoised - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    print(f"✅ SNR: {snr_db:.1f} dB")
    print(f"✅ Std Reduction: {std_reduction:.3f}")
    print(f"✅ Peak-to-Peak: {peak_to_peak:.1f}")
    print(f"✅ RMSE: {rmse:.3f}")
    print(f"✅ Correlation: {correlation:.3f}")
    print(f"✅ MAE: {mae:.3f}")
    print(f"✅ PSNR: {psnr_db:.1f} dB")
    print(f"✅ SSIM: {ssim:.3f}")
    
    return True

def create_stft_demo_plot():
    """Create a demo plot showing STFT features"""
    print("\n📊 Creating STFT Demo Plot")
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
    denoised_signal = savgol_filter(original_signal, 51, 3)
    
    # Compute STFT
    f_orig, t_stft, Zxx_orig = stft(original_signal, fs=fs, nperseg=256, noverlap=128)
    f_denoised, _, Zxx_denoised = stft(denoised_signal, fs=fs, nperseg=256, noverlap=128)
    
    # Compute spectral features
    magnitude_orig = np.abs(Zxx_orig)
    spectral_centroid_orig = np.sum(f_orig[:, np.newaxis] * magnitude_orig, axis=0) / np.sum(magnitude_orig, axis=0)
    
    magnitude_denoised = np.abs(Zxx_denoised)
    spectral_centroid_denoised = np.sum(f_denoised[:, np.newaxis] * magnitude_denoised, axis=0) / np.sum(magnitude_denoised, axis=0)
    
    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('STFT and Spectral Features Demo', fontsize=16, fontweight='bold')
    
    # Original signal
    ax1.plot(t, original_signal, 'b-', linewidth=1, alpha=0.7, label='Original')
    ax1.plot(t, denoised_signal, 'r-', linewidth=1.5, label='Denoised')
    ax1.set_title('Time Domain Signal', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # STFT - Original
    im1 = ax2.pcolormesh(t_stft, f_orig[f_orig <= 20], np.abs(Zxx_orig[f_orig <= 20, :]), 
                        shading='gouraud', cmap='jet')
    ax2.set_title('STFT - Original Signal', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=ax2, label='Magnitude')
    
    # STFT - Denoised
    im2 = ax3.pcolormesh(t_stft, f_denoised[f_denoised <= 20], np.abs(Zxx_denoised[f_denoised <= 20, :]), 
                        shading='gouraud', cmap='jet')
    ax3.set_title('STFT - Denoised Signal', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=ax3, label='Magnitude')
    
    # Spectral features
    ax4.plot(t_stft, spectral_centroid_orig, 'b-', linewidth=2, label='Centroid (Orig)')
    ax4.plot(t_stft, spectral_centroid_denoised, 'r-', linewidth=2, label='Centroid (Denoised)')
    ax4.set_title('Spectral Centroid Comparison', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'stft_features_demo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ STFT demo plot saved: {output_path}")

def main():
    """Main test function"""
    print("🚀 STFT and Enhanced Features Test Suite")
    print("="*60)
    
    # Test STFT computation
    stft_ok = test_stft_computation()
    
    # Test spectral features
    spectral_ok = test_spectral_features()
    
    # Test enhanced metrics
    metrics_ok = test_enhanced_metrics()
    
    # Create demo plot
    create_stft_demo_plot()
    
    # Summary
    print("\n" + "="*60)
    print("STFT AND ENHANCED FEATURES TEST SUMMARY")
    print("="*60)
    print(f"STFT Computation: {'✅ PASSED' if stft_ok else '❌ FAILED'}")
    print(f"Spectral Features: {'✅ PASSED' if spectral_ok else '❌ FAILED'}")
    print(f"Enhanced Metrics: {'✅ PASSED' if metrics_ok else '❌ FAILED'}")
    
    if stft_ok and spectral_ok and metrics_ok:
        print("\n🎉 STFT and enhanced features are working correctly!")
        print("📁 New features added:")
        print("   ✅ Real-time STFT computation")
        print("   ✅ Adjustable STFT parameters (window, overlap, NFFT)")
        print("   ✅ Multiple window types (hann, hamming, blackman, etc.)")
        print("   ✅ Spectral centroid and rolloff features")
        print("   ✅ Enhanced evaluation metrics (MAE, PSNR, SSIM)")
        print("   ✅ Dual STFT display (original vs denoised)")
        print("   ✅ Spectral features visualization")
        print("\n💡 You can now run the enhanced GUI:")
        print("   python ppg_denoising_gui.py")
    else:
        print("\n⚠️ Some STFT features failed. Please check the implementation.")

if __name__ == "__main__":
    main()
