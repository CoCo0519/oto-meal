#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for PPG Denoising GUI
Quick test to verify GUI functionality without opening the full interface
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt, butter, filtfilt, wiener
import os
from pathlib import Path

def test_denoising_methods():
    """Test all denoising methods with sample data"""
    print("🧪 Testing PPG Denoising Methods")
    print("="*50)
    
    # Create synthetic PPG signal
    t = np.linspace(0, 10, 1000)  # 10 seconds
    fs = 100
    
    # Heart rate component (72 BPM)
    hr_freq = 1.2
    heart_component = np.sin(2 * np.pi * hr_freq * t)
    
    # Respiratory component (12 breaths per minute)
    resp_freq = 0.2
    resp_component = 0.3 * np.sin(2 * np.pi * resp_freq * t)
    
    # Noise
    noise = 0.2 * np.random.randn(len(t))
    
    # Combine components
    original_signal = heart_component + resp_component + noise
    
    print(f"📊 Signal created: {len(original_signal)} samples, {t[-1]:.1f}s duration")
    print(f"📈 Signal range: {np.min(original_signal):.3f} to {np.max(original_signal):.3f}")
    
    # Test different denoising methods
    methods = {
        'Savitzky-Golay': lambda x: savgol_filter(x, 51, 3),
        'Median Filter': lambda x: medfilt(x, 5),
        'Wiener Filter': lambda x: wiener(x, noise=0.1),
        'Wavelet Approx': lambda x: filtfilt(*butter(6, 12.5/(fs/2), btype='low'), x),
        'Bandpass Filter': lambda x: filtfilt(*butter(4, [0.8/(fs/2), 3.5/(fs/2)], btype='band'), x)
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        try:
            denoised = method_func(original_signal)
            
            # Calculate performance metrics
            noise_removed = original_signal - denoised
            signal_power = np.var(denoised)
            noise_power = np.var(noise_removed)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = float('inf')
            
            std_reduction = np.std(original_signal) - np.std(denoised)
            
            results[method_name] = {
                'snr_db': snr_db,
                'std_reduction': std_reduction,
                'denoised': denoised
            }
            
            print(f"✅ {method_name}: SNR={snr_db:.1f}dB, Std Reduction={std_reduction:.3f}")
            
        except Exception as e:
            print(f"❌ {method_name}: Failed - {e}")
    
    # Find best method
    if results:
        best_method = max(results.keys(), key=lambda x: results[x]['snr_db'])
        print(f"\n🏆 Best Method: {best_method}")
        print(f"   SNR: {results[best_method]['snr_db']:.1f} dB")
        print(f"   Std Reduction: {results[best_method]['std_reduction']:.3f}")
    
    return results

def test_gui_components():
    """Test GUI components without opening the interface"""
    print("\n🔧 Testing GUI Components")
    print("="*50)
    
    try:
        import tkinter as tk
        print("✅ tkinter available")
        
        # Test basic tkinter functionality
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test matplotlib integration
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        print("✅ matplotlib-tkinter integration available")
        
        # Test figure creation
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")
        
        # Test canvas creation (without displaying)
        canvas = FigureCanvasTkAgg(fig, root)
        print("✅ FigureCanvasTkAgg created successfully")
        
        root.destroy()
        
    except ImportError as e:
        print(f"❌ GUI components not available: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\n📁 Testing Data Loading")
    print("="*50)
    
    # Look for available data files
    data_dirs = ['hyx_data', 'lhr_data', 'lj_data']
    found_files = []
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            txt_files = list(Path(data_dir).glob("*.txt"))
            if txt_files:
                found_files.extend(txt_files[:2])  # Take first 2 files
                print(f"✅ Found {len(txt_files)} files in {data_dir}")
    
    if not found_files:
        print("⚠️ No data files found, will use synthetic data")
        return True
    
    # Test loading a sample file
    test_file = found_files[0]
    try:
        # Try different encodings
        try:
            data = np.loadtxt(test_file, skiprows=1, encoding='utf-8')
            print(f"✅ Loaded {test_file.name} (UTF-8): {data.shape}")
        except:
            data = np.loadtxt(test_file, skiprows=1, encoding='gbk')
            print(f"✅ Loaded {test_file.name} (GBK): {data.shape}")
        
        # Test signal extraction
        if len(data.shape) == 1:
            signal = data
        else:
            signal = data[:, 0]
        
        print(f"📊 Signal length: {len(signal)} samples")
        print(f"📈 Signal range: {np.min(signal):.1f} to {np.max(signal):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load {test_file.name}: {e}")
        return False

def create_demo_plot():
    """Create a demo plot showing denoising results"""
    print("\n📊 Creating Demo Plot")
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
    
    # Apply Savitzky-Golay filter
    denoised_signal = savgol_filter(original_signal, 51, 3)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original signal
    ax1.plot(t, original_signal, 'b-', linewidth=1, alpha=0.7, label='Original Signal')
    ax1.set_title('PPG Signal Denoising Demo', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Denoised signal
    ax2.plot(t, original_signal, 'b-', linewidth=1, alpha=0.5, label='Original')
    ax2.plot(t, denoised_signal, 'r-', linewidth=1.5, label='Denoised (Savitzky-Golay)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add performance metrics
    noise_removed = original_signal - denoised_signal
    signal_power = np.var(denoised_signal)
    noise_power = np.var(noise_removed)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    std_reduction = np.std(original_signal) - np.std(denoised_signal)
    
    metrics_text = f'SNR: {snr_db:.1f} dB\nStd Reduction: {std_reduction:.3f}'
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'ppg_denoising_demo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Demo plot saved: {output_path}")
    print(f"📊 Performance: SNR={snr_db:.1f}dB, Std Reduction={std_reduction:.3f}")

def main():
    """Main test function"""
    print("🚀 PPG Denoising GUI Test Suite")
    print("="*60)
    
    # Test denoising methods
    denoising_results = test_denoising_methods()
    
    # Test GUI components
    gui_available = test_gui_components()
    
    # Test data loading
    data_loading_ok = test_data_loading()
    
    # Create demo plot
    create_demo_plot()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Denoising Methods: {'✅ PASSED' if denoising_results else '❌ FAILED'}")
    print(f"GUI Components: {'✅ AVAILABLE' if gui_available else '❌ NOT AVAILABLE'}")
    print(f"Data Loading: {'✅ WORKING' if data_loading_ok else '❌ FAILED'}")
    
    if gui_available:
        print("\n🎉 GUI is ready to use!")
        print("💡 To start the GUI:")
        print("   python ppg_denoising_gui_simple.py  # Simple version")
        print("   python ppg_denoising_gui.py         # Full version")
    else:
        print("\n⚠️ GUI components not available.")
        print("💡 Install required packages:")
        print("   pip install tkinter matplotlib")
    
    print("\n📁 Generated files:")
    print("   - ppg_denoising_demo.png (Demo plot)")
    print("   - ppg_denoising_gui.py (Full GUI)")
    print("   - ppg_denoising_gui_simple.py (Simple GUI)")
    print("   - PPG_DENOISING_GUI_README.md (Documentation)")

if __name__ == "__main__":
    main()







