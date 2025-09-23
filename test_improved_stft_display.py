#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test improved STFT display with proper frequency range
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def test_improved_stft_display():
    """Test the improved STFT display"""
    print("🔧 Testing Improved STFT Display")
    print("="*50)
    
    # Create test signal with known frequency components
    fs = 100  # Sampling rate
    duration = 5  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create signal with multiple frequency components
    # Heart rate: 1.2 Hz (72 BPM)
    hr_freq = 1.2
    heart_component = np.sin(2 * np.pi * hr_freq * t)
    
    # Respiratory: 0.3 Hz (18 breaths/min)
    resp_freq = 0.3
    resp_component = 0.5 * np.sin(2 * np.pi * resp_freq * t)
    
    # Higher frequency noise
    noise_freq = 5.0
    noise_component = 0.2 * np.sin(2 * np.pi * noise_freq * t)
    
    # Random noise
    random_noise = 0.1 * np.random.randn(len(t))
    
    # Combine components
    test_signal = heart_component + resp_component + noise_component + random_noise
    
    print(f"✅ Test signal created:")
    print(f"   Length: {len(test_signal)} samples")
    print(f"   Duration: {duration} seconds")
    print(f"   Sampling rate: {fs} Hz")
    print(f"   Frequency components: {hr_freq} Hz, {resp_freq} Hz, {noise_freq} Hz")
    
    # Test STFT with different parameters
    test_params = [
        (256, 128, 512, "hann", "Standard"),
        (128, 64, 256, "hamming", "High Resolution"),
        (64, 32, 128, "blackman", "Low Resolution")
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Improved STFT Display Test", fontsize=16)
    
    for i, (nperseg, noverlap, nfft, window, name) in enumerate(test_params):
        ax = axes[i]
        
        try:
            # Compute STFT
            f, t_stft, Zxx = signal.stft(test_signal, fs=fs, window=window, 
                                         nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            
            print(f"\n✅ STFT {i+1} ({name}):")
            print(f"   Parameters: nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}")
            print(f"   Frequency bins: {len(f)}")
            print(f"   Time bins: {len(t_stft)}")
            print(f"   Frequency range: {f[0]:.1f} - {f[-1]:.1f} Hz")
            print(f"   Time range: {t_stft[0]:.1f} - {t_stft[-1]:.1f} s")
            
            # Apply frequency mask (show up to 10 Hz)
            max_freq = 10.0
            freq_mask = f <= max_freq
            
            if np.sum(freq_mask) > 0:
                magnitude = np.abs(Zxx[freq_mask, :])
                
                if np.all(np.isfinite(magnitude)) and np.max(magnitude) > 0:
                    # Use imshow for better display
                    extent = [t_stft[0], t_stft[-1], f[freq_mask][0], f[freq_mask][-1]]
                    im = ax.imshow(magnitude, aspect='auto', origin='lower',
                                  extent=extent, cmap='jet')
                    
                    ax.set_title(f"STFT {i+1}: {name} (Window: {window})")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Magnitude')
                    
                    # Set proper limits
                    ax.set_ylim(f[freq_mask][0], f[freq_mask][-1])
                    ax.set_xlim(t_stft[0], t_stft[-1])
                    
                    print(f"   ✅ Display successful: {magnitude.shape}")
                    print(f"   ✅ Magnitude range: {np.min(magnitude):.6f} - {np.max(magnitude):.6f}")
                else:
                    ax.text(0.5, 0.5, 'Invalid magnitude data', 
                           transform=ax.transAxes, ha='center', va='center')
                    print(f"   ❌ Invalid magnitude data")
            else:
                ax.text(0.5, 0.5, 'No frequencies in range', 
                       transform=ax.transAxes, ha='center', va='center')
                print(f"   ❌ No frequencies in range")
                
        except Exception as e:
            ax.text(0.5, 0.5, f'STFT failed: {str(e)[:30]}...', 
                   transform=ax.transAxes, ha='center', va='center')
            print(f"   ❌ STFT failed: {e}")
    
    plt.tight_layout()
    plt.savefig('test_improved_stft_display.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Test plot saved as 'test_improved_stft_display.png'")

def test_frequency_range_validation():
    """Test frequency range validation"""
    print("\n🔍 Testing Frequency Range Validation")
    print("="*50)
    
    # Test with different sampling rates
    test_cases = [
        (100, "PPG Signal"),
        (1000, "High-res Signal"),
        (50, "Low-res Signal")
    ]
    
    for fs, name in test_cases:
        print(f"\n📊 Testing {name} (fs={fs} Hz):")
        
        # Create test signal
        duration = 2
        t = np.linspace(0, duration, int(fs * duration))
        signal_data = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
        
        # Test STFT with adaptive parameters
        nperseg = min(256, len(signal_data) // 2)
        if nperseg % 2 == 0:
            nperseg += 1
        noverlap = nperseg // 2
        
        f, t_stft, Zxx = signal.stft(signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        print(f"   Frequency range: {f[0]:.1f} - {f[-1]:.1f} Hz")
        print(f"   Expected max freq: {fs/2:.1f} Hz")
        print(f"   Frequency resolution: {f[1]-f[0]:.3f} Hz")
        
        # Check if frequency range is correct
        expected_max = fs / 2
        if abs(f[-1] - expected_max) < 0.1:
            print(f"   ✅ Frequency range is correct")
        else:
            print(f"   ❌ Frequency range is incorrect")

def main():
    """Main test function"""
    print("🚀 Improved STFT Display Test Suite")
    print("="*60)
    
    # Test improved STFT display
    test_improved_stft_display()
    
    # Test frequency range validation
    test_frequency_range_validation()
    
    # Summary
    print("\n" + "="*60)
    print("IMPROVED STFT DISPLAY TEST SUMMARY")
    print("="*60)
    print("✅ Key improvements:")
    print("   1. Used imshow instead of pcolormesh for better display")
    print("   2. Added proper extent parameter for correct scaling")
    print("   3. Fixed colorbar duplication issue")
    print("   4. Added proper axis limits")
    print("   5. Enhanced parameter validation")
    print("   6. Added detailed debug information")
    
    print("\n🎉 STFT display should now show correct frequency ranges!")
    print("💡 The frequency range should now be 0 - fs/2 Hz (e.g., 0-50 Hz for fs=100)")
    print("📁 Check the test plot: test_improved_stft_display.png")

if __name__ == "__main__":
    main()
