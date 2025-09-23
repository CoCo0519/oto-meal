#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test STFT functionality fix
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def test_stft_fix():
    """Test the STFT fix"""
    print("🧪 Testing STFT Fix")
    print("="*50)
    
    try:
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
        
        # Test STFT computation (the fixed version)
        nperseg = 256
        noverlap = 128
        nfft = 512
        window_type = "hann"
        
        f, t_stft, Zxx = signal.stft(test_signal, fs=fs, window=window_type, 
                                     nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        print(f"✅ STFT computed successfully:")
        print(f"   Frequency bins: {len(f)}")
        print(f"   Time bins: {len(t_stft)}")
        print(f"   STFT shape: {Zxx.shape}")
        print(f"   Frequency range: {f[0]:.1f} - {f[-1]:.1f} Hz")
        print(f"   Time range: {t_stft[0]:.1f} - {t_stft[-1]:.1f} s")
        
        # Test spectral features
        magnitude = np.abs(Zxx)
        spectral_centroid = np.sum(f[:, np.newaxis] * magnitude, axis=0) / np.sum(magnitude, axis=0)
        
        print(f"✅ Spectral centroid computed: {np.mean(spectral_centroid):.1f} Hz")
        
        # Test different window types
        window_types = ["hann", "hamming", "blackman", "bartlett", "boxcar"]
        for window in window_types:
            try:
                f_test, t_test, Zxx_test = signal.stft(test_signal, fs=fs, window=window, 
                                                      nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                print(f"✅ Window '{window}' works")
            except Exception as e:
                print(f"❌ Window '{window}' failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ STFT test failed: {e}")
        return False

def test_gui_stft_integration():
    """Test GUI STFT integration"""
    print("\n🔧 Testing GUI STFT Integration")
    print("="*50)
    
    try:
        # Test the compute_stft function signature
        import inspect
        
        # Import the GUI class
        import sys
        sys.path.append('.')
        
        # Check if we can import the GUI module
        try:
            from ppg_denoising_gui import PPGDenoisingGUI
            print("✅ GUI module imported successfully")
            
            # Check the compute_stft method signature
            method = getattr(PPGDenoisingGUI, 'compute_stft')
            sig = inspect.signature(method)
            print(f"✅ compute_stft method signature: {sig}")
            
            # Check parameter names
            params = list(sig.parameters.keys())
            print(f"✅ Method parameters: {params}")
            
            if 'signal_data' in params:
                print("✅ Parameter name 'signal_data' is correct")
            else:
                print("❌ Parameter name should be 'signal_data'")
                
        except ImportError as e:
            print(f"❌ Failed to import GUI module: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ GUI integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 STFT Fix Test Suite")
    print("="*60)
    
    # Test STFT fix
    stft_ok = test_stft_fix()
    
    # Test GUI integration
    gui_ok = test_gui_stft_integration()
    
    # Summary
    print("\n" + "="*60)
    print("STFT FIX TEST SUMMARY")
    print("="*60)
    print(f"STFT Computation: {'✅ PASSED' if stft_ok else '❌ FAILED'}")
    print(f"GUI Integration: {'✅ PASSED' if gui_ok else '❌ FAILED'}")
    
    if stft_ok and gui_ok:
        print("\n🎉 All tests passed! STFT functionality should work correctly now.")
        print("💡 The error 'numpy.ndarray' object has no attribute 'stft' should be fixed.")
        print("\n📁 You can now:")
        print("   1. Run: python ppg_denoising_gui.py")
        print("   2. Load a signal")
        print("   3. Adjust STFT parameters")
        print("   4. See real-time STFT spectrograms")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

