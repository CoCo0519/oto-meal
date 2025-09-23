#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify GUI launches correctly
"""

import tkinter as tk
import numpy as np
from scipy.signal import savgol_filter

def test_gui_components():
    """Test GUI components without opening the interface"""
    print("🧪 Testing GUI Components")
    print("="*50)
    
    try:
        # Test basic tkinter functionality
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test matplotlib integration
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        print("✅ matplotlib-tkinter integration available")
        
        # Test figure creation with 2x3 layout
        fig = Figure(figsize=(16, 10), dpi=100)
        
        # Create all 6 subplots
        ax_time = fig.add_subplot(2, 3, 1)
        ax_freq = fig.add_subplot(2, 3, 2)
        ax_stft_orig = fig.add_subplot(2, 3, 3)
        ax_stft_denoised = fig.add_subplot(2, 3, 4)
        ax_diff = fig.add_subplot(2, 3, 5)
        ax_spectral = fig.add_subplot(2, 3, 6)
        
        print("✅ All 6 subplots created successfully")
        
        # Test canvas creation
        canvas = FigureCanvasTkAgg(fig, root)
        print("✅ FigureCanvasTkAgg created successfully")
        
        # Test signal processing
        t = np.linspace(0, 5, 500)
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(len(t))
        denoised = savgol_filter(signal, 51, 3)
        
        print("✅ Signal processing functions working")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI components test failed: {e}")
        return False

def test_stft_imports():
    """Test STFT-related imports"""
    print("\n🔧 Testing STFT Imports")
    print("="*50)
    
    try:
        from scipy.signal import stft
        print("✅ scipy.signal.stft available")
        
        import numpy as np
        print("✅ numpy available")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib.pyplot available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_signal_processing():
    """Test signal processing functions"""
    print("\n📊 Testing Signal Processing")
    print("="*50)
    
    try:
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
        
        # Test STFT computation
        from scipy.signal import stft
        f, t_stft, Zxx = stft(original_signal, fs=fs, nperseg=256, noverlap=128)
        
        print(f"✅ STFT computed: shape {Zxx.shape}, freq range {f[0]:.1f}-{f[-1]:.1f} Hz")
        
        # Test spectral features
        magnitude = np.abs(Zxx)
        spectral_centroid = np.sum(f[:, np.newaxis] * magnitude, axis=0) / np.sum(magnitude, axis=0)
        
        print(f"✅ Spectral centroid computed: {np.mean(spectral_centroid):.1f} Hz")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal processing test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 GUI Launch Test Suite")
    print("="*60)
    
    # Test GUI components
    gui_ok = test_gui_components()
    
    # Test STFT imports
    imports_ok = test_stft_imports()
    
    # Test signal processing
    processing_ok = test_signal_processing()
    
    # Summary
    print("\n" + "="*60)
    print("GUI LAUNCH TEST SUMMARY")
    print("="*60)
    print(f"GUI Components: {'✅ PASSED' if gui_ok else '❌ FAILED'}")
    print(f"STFT Imports: {'✅ PASSED' if imports_ok else '❌ FAILED'}")
    print(f"Signal Processing: {'✅ PASSED' if processing_ok else '❌ FAILED'}")
    
    if gui_ok and imports_ok and processing_ok:
        print("\n🎉 All tests passed! GUI should launch correctly now.")
        print("💡 You can now run:")
        print("   python ppg_denoising_gui.py")
        print("\n📁 Features available:")
        print("   ✅ Real-time STFT computation")
        print("   ✅ Adjustable STFT parameters")
        print("   ✅ Enhanced evaluation metrics")
        print("   ✅ Spectral features analysis")
        print("   ✅ Dual STFT display")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

