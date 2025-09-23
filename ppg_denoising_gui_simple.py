#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPG Signal Denoising GUI - Simplified Version
Quick demo version for testing the GUI functionality
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
import scipy.signal as signal
from scipy.signal import savgol_filter, medfilt, butter, filtfilt, wiener
from pathlib import Path

class SimplePPGDenoisingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG Signal Denoising Toolkit - Demo")
        self.root.geometry("1200x800")
        
        # Data variables
        self.original_signal = None
        self.time_axis = None
        self.fs = 100
        self.denoised_signal = None
        
        # Create GUI
        self.create_widgets()
        self.load_demo_signal()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - Plot
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(left_panel)
        self.create_plot_panel(right_panel)
    
    def create_control_panel(self, parent):
        """Create control panel"""
        # Method selection
        method_frame = ttk.LabelFrame(parent, text="Denoising Method", padding=10)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.method_var = tk.StringVar(value="savgol")
        methods = [
            ("Savitzky-Golay", "savgol"),
            ("Median Filter", "median"),
            ("Wiener Filter", "wiener"),
            ("Wavelet Approx", "wavelet_approx"),
            ("Bandpass Filter", "bandpass")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, 
                           value=value, command=self.apply_denoising).pack(anchor=tk.W)
        
        # Parameters
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Savitzky-Golay
        ttk.Label(param_frame, text="SG Window:").pack(anchor=tk.W)
        self.sg_window_var = tk.IntVar(value=51)
        ttk.Scale(param_frame, from_=5, to=101, variable=self.sg_window_var,
                 orient=tk.HORIZONTAL, command=self.apply_denoising).pack(fill=tk.X)
        
        ttk.Label(param_frame, text="SG Poly Order:").pack(anchor=tk.W)
        self.sg_poly_var = tk.IntVar(value=3)
        ttk.Scale(param_frame, from_=1, to=5, variable=self.sg_poly_var,
                 orient=tk.HORIZONTAL, command=self.apply_denoising).pack(fill=tk.X)
        
        # Median Filter
        ttk.Label(param_frame, text="Median Kernel:").pack(anchor=tk.W)
        self.median_kernel_var = tk.IntVar(value=5)
        ttk.Scale(param_frame, from_=3, to=21, variable=self.median_kernel_var,
                 orient=tk.HORIZONTAL, command=self.apply_denoising).pack(fill=tk.X)
        
        # Wiener Filter
        ttk.Label(param_frame, text="Wiener Noise:").pack(anchor=tk.W)
        self.wiener_noise_var = tk.DoubleVar(value=0.1)
        ttk.Scale(param_frame, from_=0.01, to=1.0, variable=self.wiener_noise_var,
                 orient=tk.HORIZONTAL, command=self.apply_denoising).pack(fill=tk.X)
        
        # Wavelet Approximation
        ttk.Label(param_frame, text="Wavelet Cutoff (Hz):").pack(anchor=tk.W)
        self.wavelet_cutoff_var = tk.DoubleVar(value=12.5)
        ttk.Scale(param_frame, from_=1.0, to=50.0, variable=self.wavelet_cutoff_var,
                 orient=tk.HORIZONTAL, command=self.apply_denoising).pack(fill=tk.X)
        
        # Bandpass Filter
        ttk.Label(param_frame, text="Bandpass Low (Hz):").pack(anchor=tk.W)
        self.bp_low_var = tk.DoubleVar(value=0.8)
        ttk.Scale(param_frame, from_=0.1, to=5.0, variable=self.bp_low_var,
                 orient=tk.HORIZONTAL, command=self.apply_denoising).pack(fill=tk.X)
        
        ttk.Label(param_frame, text="Bandpass High (Hz):").pack(anchor=tk.W)
        self.bp_high_var = tk.DoubleVar(value=3.5)
        ttk.Scale(param_frame, from_=1.0, to=10.0, variable=self.bp_high_var,
                 orient=tk.HORIZONTAL, command=self.apply_denoising).pack(fill=tk.X)
        
        # Display options
        display_frame = ttk.LabelFrame(parent, text="Display", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.show_original_var = tk.BooleanVar(value=True)
        self.show_denoised_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(display_frame, text="Show Original", 
                       variable=self.show_original_var, command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Denoised", 
                       variable=self.show_denoised_var, command=self.update_plot).pack(anchor=tk.W)
        
        # Metrics
        metrics_frame = ttk.LabelFrame(parent, text="Performance", padding=10)
        metrics_frame.pack(fill=tk.X)
        
        self.snr_label = ttk.Label(metrics_frame, text="SNR: -- dB")
        self.snr_label.pack(anchor=tk.W)
        
        self.std_label = ttk.Label(metrics_frame, text="Std Reduction: --")
        self.std_label.pack(anchor=tk.W)
    
    def create_plot_panel(self, parent):
        """Create plot panel"""
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("PPG Signal Analysis")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_demo_signal(self):
        """Load a demo signal"""
        # Look for available data files
        data_dirs = ['hyx_data', 'lhr_data', 'lj_data']
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                txt_files = list(Path(data_dir).glob("*.txt"))
                if txt_files:
                    self.load_signal_file(str(txt_files[0]))
                    break
        
        # If no data files found, create a synthetic signal
        if self.original_signal is None:
            self.create_synthetic_signal()
    
    def load_signal_file(self, file_path):
        """Load signal from file"""
        try:
            try:
                data = np.loadtxt(file_path, skiprows=1, encoding='utf-8')
            except:
                data = np.loadtxt(file_path, skiprows=1, encoding='gbk')
            
            if len(data.shape) == 1:
                self.original_signal = data
            else:
                self.original_signal = data[:, 0]
            
            self.time_axis = np.arange(len(self.original_signal)) / self.fs
            self.update_plot()
            
        except Exception as e:
            print(f"Failed to load signal: {e}")
            self.create_synthetic_signal()
    
    def create_synthetic_signal(self):
        """Create a synthetic PPG signal for demonstration"""
        t = np.linspace(0, 10, 1000)  # 10 seconds
        
        # Create synthetic PPG signal
        # Heart rate component
        hr_freq = 1.2  # 72 BPM
        heart_component = np.sin(2 * np.pi * hr_freq * t)
        
        # Respiratory component
        resp_freq = 0.2  # 12 breaths per minute
        resp_component = 0.3 * np.sin(2 * np.pi * resp_freq * t)
        
        # Noise
        noise = 0.2 * np.random.randn(len(t))
        
        # Combine components
        self.original_signal = heart_component + resp_component + noise
        self.time_axis = t
        self.fs = 100
        
        self.update_plot()
    
    def apply_denoising(self, *args):
        """Apply denoising with current parameters"""
        if self.original_signal is None:
            return
        
        try:
            method = self.method_var.get()
            
            if method == "savgol":
                window_length = self.sg_window_var.get()
                poly_order = self.sg_poly_var.get()
                if window_length % 2 == 0:
                    window_length += 1
                self.denoised_signal = savgol_filter(self.original_signal, window_length, poly_order)
            
            elif method == "median":
                kernel_size = self.median_kernel_var.get()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                self.denoised_signal = medfilt(self.original_signal, kernel_size)
            
            elif method == "wiener":
                noise_var = self.wiener_noise_var.get()
                self.denoised_signal = wiener(self.original_signal, noise=noise_var)
            
            elif method == "wavelet_approx":
                cutoff = self.wavelet_cutoff_var.get()
                b, a = butter(6, cutoff / (self.fs/2), btype='low')
                self.denoised_signal = filtfilt(b, a, self.original_signal)
            
            elif method == "bandpass":
                low_freq = self.bp_low_var.get()
                high_freq = self.bp_high_var.get()
                b, a = butter(4, [low_freq/(self.fs/2), high_freq/(self.fs/2)], btype='band')
                self.denoised_signal = filtfilt(b, a, self.original_signal)
            
            self.update_plot()
            self.update_metrics()
            
        except Exception as e:
            print(f"Denoising failed: {e}")
    
    def update_plot(self):
        """Update plot"""
        if self.original_signal is None:
            return
        
        self.ax.clear()
        self.ax.set_title("PPG Signal Analysis")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        
        if self.show_original_var.get():
            self.ax.plot(self.time_axis, self.original_signal, 'b-', 
                        linewidth=1, alpha=0.7, label='Original')
        
        if self.show_denoised_var.get() and self.denoised_signal is not None:
            self.ax.plot(self.time_axis, self.denoised_signal, 'r-', 
                        linewidth=1.5, label='Denoised')
        
        self.ax.legend()
        self.canvas.draw()
    
    def update_metrics(self):
        """Update performance metrics"""
        if self.denoised_signal is None:
            return
        
        # Calculate SNR
        noise = self.original_signal - self.denoised_signal
        signal_power = np.var(self.denoised_signal)
        noise_power = np.var(noise)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        # Calculate std reduction
        std_reduction = np.std(self.original_signal) - np.std(self.denoised_signal)
        
        # Update labels
        self.snr_label.config(text=f"SNR: {snr_db:.1f} dB")
        self.std_label.config(text=f"Std Reduction: {std_reduction:.2f}")

def main():
    """Main function"""
    root = tk.Tk()
    app = SimplePPGDenoisingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

