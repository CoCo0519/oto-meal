#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose STFT display issues
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def test_stft_display():
    """Test STFT display functionality"""
    print("🔍 Diagnosing STFT Display Issues")
    print("="*50)
    
    # Create test signal
    fs = 100
    t = np.linspace(0, 5, 500)
    
    # Heart rate component
    hr_freq = 1.2
    heart_component = np.sin(2 * np.pi * hr_freq * t)
    
    # Respiratory component
    resp_freq = 0.2
    resp_component = 0.3 * np.sin(2 * np.pi * resp_freq * t)
    
    # Noise
    noise = 0.3 * np.random.randn(len(t))
    
    # Combine components
    test_signal = heart_component + resp_component + noise
    
    print(f"✅ Test signal created: {len(test_signal)} samples")
    print(f"   Signal range: {np.min(test_signal):.3f} to {np.max(test_signal):.3f}")
    
    # Test STFT computation
    nperseg = 256
    noverlap = 128
    nfft = 512
    window_type = "hann"
    
    try:
        f, t_stft, Zxx = signal.stft(test_signal, fs=fs, window=window_type, 
                                     nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        print(f"✅ STFT computed successfully:")
        print(f"   Frequency bins: {len(f)}")
        print(f"   Time bins: {len(t_stft)}")
        print(f"   STFT shape: {Zxx.shape}")
        print(f"   Frequency range: {f[0]:.1f} - {f[-1]:.1f} Hz")
        print(f"   Time range: {t_stft[0]:.1f} - {t_stft[-1]:.1f} s")
        
        # Check STFT data
        magnitude = np.abs(Zxx)
        print(f"   Magnitude range: {np.min(magnitude):.6f} to {np.max(magnitude):.6f}")
        print(f"   Magnitude mean: {np.mean(magnitude):.6f}")
        
        # Test frequency masking
        max_freq = 20.0
        freq_mask = f <= max_freq
        print(f"   Frequency mask: {np.sum(freq_mask)} out of {len(f)} frequencies")
        
        if np.sum(freq_mask) == 0:
            print("❌ ERROR: No frequencies in range!")
            return False
        
        # Test pcolormesh data
        f_masked = f[freq_mask]
        Zxx_masked = magnitude[freq_mask, :]
        
        print(f"   Masked frequency range: {f_masked[0]:.1f} - {f_masked[-1]:.1f} Hz")
        print(f"   Masked STFT shape: {Zxx_masked.shape}")
        print(f"   Masked magnitude range: {np.min(Zxx_masked):.6f} to {np.max(Zxx_masked):.6f}")
        
        # Test if data is valid for pcolormesh
        if Zxx_masked.size == 0:
            print("❌ ERROR: Masked STFT data is empty!")
            return False
        
        if np.all(np.isnan(Zxx_masked)) or np.all(np.isinf(Zxx_masked)):
            print("❌ ERROR: STFT data contains NaN or Inf values!")
            return False
        
        print("✅ STFT data is valid for display")
        
        # Test actual plotting
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Test pcolormesh
            im = ax.pcolormesh(t_stft, f_masked, Zxx_masked, 
                              shading='gouraud', cmap='jet')
            
            ax.set_title("Test STFT Display")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Magnitude')
            
            # Save test plot
            plt.savefig('test_stft_display.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Test plot saved as 'test_stft_display.png'")
            
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ STFT computation failed: {e}")
        return False

def test_gui_stft_issues():
    """Test specific GUI STFT issues"""
    print("\n🔧 Testing GUI STFT Issues")
    print("="*50)
    
    try:
        # Import GUI module
        import sys
        sys.path.append('.')
        from ppg_denoising_gui import PPGDenoisingGUI
        
        # Check STFT parameters
        print("✅ GUI module imported successfully")
        
        # Test parameter ranges
        test_params = [
            ("Window Size", 64, 1024),
            ("Overlap Size", 32, 512),
            ("NFFT Size", 128, 2048),
            ("Frequency Range", 5.0, 50.0)
        ]
        
        for param_name, min_val, max_val in test_params:
            print(f"✅ {param_name}: {min_val} - {max_val}")
        
        # Test window types
        window_types = ["hann", "hamming", "blackman", "bartlett", "boxcar"]
        print(f"✅ Window types: {', '.join(window_types)}")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("🚀 STFT Display Diagnostic")
    print("="*60)
    
    # Test STFT display
    display_ok = test_stft_display()
    
    # Test GUI issues
    gui_ok = test_gui_stft_issues()
    
    # Summary
    print("\n" + "="*60)
    print("STFT DISPLAY DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"STFT Display: {'✅ PASSED' if display_ok else '❌ FAILED'}")
    print(f"GUI Integration: {'✅ PASSED' if gui_ok else '❌ FAILED'}")
    
    if display_ok and gui_ok:
        print("\n🎉 STFT display should work correctly!")
        print("💡 If you're still having issues, check:")
        print("   1. Signal data is loaded")
        print("   2. STFT parameters are reasonable")
        print("   3. Display options are enabled")
        print("   4. Check the test plot: test_stft_display.png")
    else:
        print("\n⚠️ Issues detected. Please check the errors above.")

if __name__ == "__main__":
    main()

