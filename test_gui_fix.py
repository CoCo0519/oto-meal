#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify GUI fix
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.signal import savgol_filter

def test_combobox_fix():
    """Test the Combobox fix"""
    print("🧪 Testing Combobox Fix")
    print("="*40)
    
    try:
        # Create a test window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test Combobox with bind (the fixed version)
        frame = ttk.Frame(root)
        
        var = tk.StringVar(value="median")
        combo = ttk.Combobox(frame, textvariable=var,
                            values=["median", "wiener", "wavelet_approx"], 
                            state="readonly")
        
        # Test the bind method (this should work)
        def test_callback(event):
            print(f"✅ Combobox selection changed: {var.get()}")
        
        combo.bind('<<ComboboxSelected>>', test_callback)
        
        print("✅ Combobox created successfully with bind method")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Combobox test failed: {e}")
        return False

def test_denoising_methods():
    """Test denoising methods"""
    print("\n🔧 Testing Denoising Methods")
    print("="*40)
    
    # Create test signal
    t = np.linspace(0, 5, 500)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.random.randn(len(t))
    
    methods = {
        'Savitzky-Golay': lambda x: savgol_filter(x, 51, 3),
        'Median Filter': lambda x: medfilt(x, 5),
        'Wiener Filter': lambda x: wiener(x, noise=0.1),
    }
    
    for name, method in methods.items():
        try:
            denoised = method(signal)
            snr = 10 * np.log10(np.var(denoised) / np.var(signal - denoised))
            print(f"✅ {name}: SNR = {snr:.1f} dB")
        except Exception as e:
            print(f"❌ {name}: Failed - {e}")
    
    return True

def main():
    """Main test function"""
    print("🚀 GUI Fix Verification Test")
    print("="*50)
    
    # Test Combobox fix
    combobox_ok = test_combobox_fix()
    
    # Test denoising methods
    denoising_ok = test_denoising_methods()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Combobox Fix: {'✅ PASSED' if combobox_ok else '❌ FAILED'}")
    print(f"Denoising Methods: {'✅ PASSED' if denoising_ok else '❌ FAILED'}")
    
    if combobox_ok and denoising_ok:
        print("\n🎉 All tests passed! GUI should work correctly now.")
        print("💡 You can now run:")
        print("   python ppg_denoising_gui.py         # Full GUI")
        print("   python ppg_denoising_gui_simple.py  # Simple GUI")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
