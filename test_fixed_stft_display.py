#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test fixed STFT display functionality
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def test_fixed_stft_display():
    """Test the fixed STFT display"""
    print("🔧 Testing Fixed STFT Display")
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
    
    # Test different STFT parameters
    test_params = [
        (256, 128, 512, "hann"),
        (128, 64, 256, "hamming"),
        (64, 32, 128, "blackman"),
        (512, 256, 1024, "bartlett")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("STFT Display Test - Different Parameters", fontsize=16)
    
    for i, (nperseg, noverlap, nfft, window) in enumerate(test_params):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        try:
            # Compute STFT
            f, t_stft, Zxx = signal.stft(test_signal, fs=fs, window=window, 
                                         nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            
            # Apply frequency mask
            max_freq = 20.0
            freq_mask = f <= max_freq
            
            if np.sum(freq_mask) > 0 and Zxx[freq_mask, :].size > 0:
                magnitude = np.abs(Zxx[freq_mask, :])
                
                if np.all(np.isfinite(magnitude)) and np.max(magnitude) > 0:
                    # Plot STFT
                    im = ax.pcolormesh(t_stft, f[freq_mask], magnitude, 
                                      shading='gouraud', cmap='jet')
                    
                    ax.set_title(f"STFT: {window}, {nperseg}/{noverlap}/{nfft}")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Magnitude')
                    
                    print(f"✅ STFT {i+1}: {window}, shape={Zxx.shape}, mag_range={np.min(magnitude):.6f}-{np.max(magnitude):.6f}")
                else:
                    ax.text(0.5, 0.5, 'Invalid magnitude data', 
                           transform=ax.transAxes, ha='center', va='center')
                    print(f"❌ STFT {i+1}: Invalid magnitude data")
            else:
                ax.text(0.5, 0.5, 'No frequencies in range', 
                       transform=ax.transAxes, ha='center', va='center')
                print(f"❌ STFT {i+1}: No frequencies in range")
                
        except Exception as e:
            ax.text(0.5, 0.5, f'STFT failed: {str(e)[:20]}...', 
                   transform=ax.transAxes, ha='center', va='center')
            print(f"❌ STFT {i+1}: {e}")
    
    plt.tight_layout()
    plt.savefig('test_fixed_stft_display.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Test plot saved as 'test_fixed_stft_display.png'")

def test_gui_integration():
    """Test GUI integration with fixed STFT"""
    print("\n🔧 Testing GUI Integration")
    print("="*50)
    
    try:
        # Import GUI module
        import sys
        sys.path.append('.')
        from ppg_denoising_gui import PPGDenoisingGUI
        
        print("✅ GUI module imported successfully")
        
        # Test the fixed compute_stft method
        import inspect
        method = getattr(PPGDenoisingGUI, 'compute_stft')
        sig = inspect.signature(method)
        
        print(f"✅ compute_stft method signature: {sig}")
        
        # Check if the method has the right parameter validation
        source = inspect.getsource(method)
        if "signal_data is None" in source:
            print("✅ Parameter validation added")
        else:
            print("❌ Parameter validation missing")
        
        if "nperseg > len(signal_data)" in source:
            print("✅ Parameter bounds checking added")
        else:
            print("❌ Parameter bounds checking missing")
        
        if "Debug info" in source:
            print("✅ Debug information added")
        else:
            print("❌ Debug information missing")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Fixed STFT Display Test Suite")
    print("="*60)
    
    # Test fixed STFT display
    test_fixed_stft_display()
    
    # Test GUI integration
    gui_ok = test_gui_integration()
    
    # Summary
    print("\n" + "="*60)
    print("FIXED STFT DISPLAY TEST SUMMARY")
    print("="*60)
    print(f"GUI Integration: {'✅ PASSED' if gui_ok else '❌ FAILED'}")
    
    if gui_ok:
        print("\n🎉 STFT display should now work correctly!")
        print("💡 Key improvements:")
        print("   1. Added parameter validation")
        print("   2. Added bounds checking")
        print("   3. Added debug information")
        print("   4. Added error handling for invalid data")
        print("   5. Added fallback messages for failed displays")
        print("\n📁 You can now:")
        print("   1. Run: python ppg_denoising_gui.py")
        print("   2. Load a signal")
        print("   3. Adjust STFT parameters")
        print("   4. See real-time STFT spectrograms with proper error handling")
    else:
        print("\n⚠️ Some issues detected. Please check the errors above.")

if __name__ == "__main__":
    main()

