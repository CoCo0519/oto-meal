#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPG Signal Denoising GUI Toolkit
Interactive GUI for real-time parameter adjustment and denoising visualization
Similar to MATLAB's signal processing toolkit
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import scipy.signal as signal
from scipy.signal import savgol_filter, medfilt, butter, filtfilt, wiener
import os
from pathlib import Path
import threading
import time

class PPGDenoisingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG Signal Denoising Toolkit")
        self.root.geometry("1400x900")
        
        # Data variables
        self.original_signal = None
        self.time_axis = None
        self.fs = 100  # Default sampling rate
        self.current_signal = None
        self.denoised_signal = None
        
        # GUI variables
        self.setup_variables()
        
        # Create GUI
        self.create_widgets()
        
        # Initialize plots
        self.setup_plots()
        
        # Load default signal if available
        self.load_default_signal()
    
    def setup_variables(self):
        """Setup GUI variables"""
        # Denoising method
        self.method_var = tk.StringVar(value="savgol")
        
        # Savitzky-Golay parameters
        self.savgol_window_var = tk.IntVar(value=51)
        self.savgol_poly_var = tk.IntVar(value=3)
        self.savgol_deriv_var = tk.IntVar(value=0)  # Derivative order
        self.savgol_delta_var = tk.DoubleVar(value=1.0)  # Delta for derivative
        
        # Median filter parameters
        self.median_kernel_var = tk.IntVar(value=5)
        
        # Wiener filter parameters
        self.wiener_noise_var = tk.DoubleVar(value=0.1)
        self.wiener_mysize_var = tk.IntVar(value=3)  # Wiener filter size
        self.wiener_noise_var_auto = tk.BooleanVar(value=False)  # Auto noise estimation
        
        # Wavelet approximation parameters
        self.wavelet_cutoff_var = tk.DoubleVar(value=12.5)  # fs/8
        self.wavelet_order_var = tk.IntVar(value=6)
        self.wavelet_window_var = tk.IntVar(value=256)  # Window size for wavelet
        self.wavelet_overlap_var = tk.IntVar(value=128)  # Overlap for wavelet
        
        # Heart rate filtering parameters
        self.hr_low_var = tk.DoubleVar(value=0.8)  # Low frequency for heart rate
        self.hr_high_var = tk.DoubleVar(value=3.5)  # High frequency for heart rate
        self.hr_order_var = tk.IntVar(value=4)  # Filter order
        self.hr_smooth_var = tk.IntVar(value=3)  # Additional smoothing
        
        # Bandpass filter parameters
        self.bandpass_low_var = tk.DoubleVar(value=0.8)
        self.bandpass_high_var = tk.DoubleVar(value=3.5)
        self.bandpass_order_var = tk.IntVar(value=4)
        
        # Adaptive filter parameters
        self.adaptive_window_var = tk.IntVar(value=20)
        self.adaptive_method_var = tk.StringVar(value="moving_avg")  # Adaptive method
        self.adaptive_alpha_var = tk.DoubleVar(value=0.1)  # Learning rate for adaptive
        
        # Combined filter parameters
        self.combined_method1_var = tk.StringVar(value="median")
        self.combined_method2_var = tk.StringVar(value="savgol")
        
        # Display options
        self.show_original_var = tk.BooleanVar(value=True)
        self.show_denoised_var = tk.BooleanVar(value=True)
        self.show_spectrum_var = tk.BooleanVar(value=False)
        self.show_noise_var = tk.BooleanVar(value=False)
        self.show_metrics_var = tk.BooleanVar(value=True)
        self.show_stft_var = tk.BooleanVar(value=True)  # Always show STFT
        self.real_time_stft_var = tk.BooleanVar(value=True)  # Real-time STFT
        
        # Analysis parameters
        self.analysis_window_var = tk.IntVar(value=256)
        self.analysis_overlap_var = tk.IntVar(value=128)
        
        # STFT parameters
        self.stft_window_var = tk.IntVar(value=256)
        self.stft_overlap_var = tk.IntVar(value=128)
        self.stft_nfft_var = tk.IntVar(value=512)
        self.stft_window_type_var = tk.StringVar(value="hann")
        self.stft_freq_range_var = tk.DoubleVar(value=20.0)  # Max frequency to display
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Right panel - Plots
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(left_panel)
        self.create_plot_panel(right_panel)
    
    def create_control_panel(self, parent):
        """Create control panel"""
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Signal", command=self.load_signal).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save Denoised", command=self.save_denoised).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)
        
        # Denoising method selection
        method_frame = ttk.LabelFrame(parent, text="Denoising Method", padding=10)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        methods = [
            ("Savitzky-Golay", "savgol"),
            ("Median Filter", "median"),
            ("Wiener Filter", "wiener"),
            ("Wavelet Approx", "wavelet_approx"),
            ("Heart Rate Filter", "heart_rate"),
            ("Bandpass Filter", "bandpass"),
            ("Adaptive Filter", "adaptive"),
            ("Combined Filter", "combined")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, 
                           value=value, command=self.on_method_change).pack(anchor=tk.W)
        
        # Parameter controls
        self.create_parameter_controls(parent)
        
        # Display options
        display_frame = ttk.LabelFrame(parent, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(display_frame, text="Show Original", variable=self.show_original_var,
                       command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Denoised", variable=self.show_denoised_var,
                       command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Spectrum", variable=self.show_spectrum_var,
                       command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Noise", variable=self.show_noise_var,
                       command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Metrics", variable=self.show_metrics_var,
                       command=self.update_plot).pack(anchor=tk.W)
        
        # STFT controls
        stft_frame = ttk.LabelFrame(parent, text="STFT Parameters", padding=10)
        stft_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(stft_frame, text="Window Size:").pack(anchor=tk.W)
        ttk.Scale(stft_frame, from_=64, to=1024, variable=self.stft_window_var,
                 orient=tk.HORIZONTAL, command=self.on_stft_change).pack(fill=tk.X)
        
        ttk.Label(stft_frame, text="Overlap Size:").pack(anchor=tk.W)
        ttk.Scale(stft_frame, from_=32, to=512, variable=self.stft_overlap_var,
                 orient=tk.HORIZONTAL, command=self.on_stft_change).pack(fill=tk.X)
        
        ttk.Label(stft_frame, text="NFFT Size:").pack(anchor=tk.W)
        ttk.Scale(stft_frame, from_=128, to=2048, variable=self.stft_nfft_var,
                 orient=tk.HORIZONTAL, command=self.on_stft_change).pack(fill=tk.X)
        
        ttk.Label(stft_frame, text="Window Type:").pack(anchor=tk.W)
        self.stft_window_combo = ttk.Combobox(stft_frame, textvariable=self.stft_window_type_var,
                    values=["hann", "hamming", "blackman", "bartlett", "boxcar"], state="readonly")
        self.stft_window_combo.pack(fill=tk.X)
        self.stft_window_combo.bind('<<ComboboxSelected>>', self.on_stft_change)
        
        ttk.Label(stft_frame, text="Max Frequency (Hz):").pack(anchor=tk.W)
        ttk.Scale(stft_frame, from_=5.0, to=50.0, variable=self.stft_freq_range_var,
                 orient=tk.HORIZONTAL, command=self.on_stft_change).pack(fill=tk.X)
        
        ttk.Checkbutton(stft_frame, text="Real-time STFT", variable=self.real_time_stft_var,
                       command=self.on_stft_change).pack(anchor=tk.W)
        
        # Real-time processing
        process_frame = ttk.LabelFrame(parent, text="Processing", padding=10)
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(process_frame, text="Apply Denoising", command=self.apply_denoising).pack(fill=tk.X, pady=2)
        ttk.Button(process_frame, text="Reset Signal", command=self.reset_signal).pack(fill=tk.X, pady=2)
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X)
        
        self.snr_label = ttk.Label(metrics_frame, text="SNR: -- dB")
        self.snr_label.pack(anchor=tk.W)
        
        self.std_label = ttk.Label(metrics_frame, text="Std Reduction: --")
        self.std_label.pack(anchor=tk.W)
        
        self.peak_label = ttk.Label(metrics_frame, text="Peak-to-Peak: --")
        self.peak_label.pack(anchor=tk.W)
        
        self.rmse_label = ttk.Label(metrics_frame, text="RMSE: --")
        self.rmse_label.pack(anchor=tk.W)
        
        self.correlation_label = ttk.Label(metrics_frame, text="Correlation: --")
        self.correlation_label.pack(anchor=tk.W)
        
        self.processing_time_label = ttk.Label(metrics_frame, text="Processing Time: -- ms")
        self.processing_time_label.pack(anchor=tk.W)
        
        # Additional quality metrics
        self.mae_label = ttk.Label(metrics_frame, text="MAE: --")
        self.mae_label.pack(anchor=tk.W)
        
        self.psnr_label = ttk.Label(metrics_frame, text="PSNR: -- dB")
        self.psnr_label.pack(anchor=tk.W)
        
        self.ssim_label = ttk.Label(metrics_frame, text="SSIM: --")
        self.ssim_label.pack(anchor=tk.W)
        
        self.spectral_centroid_label = ttk.Label(metrics_frame, text="Spectral Centroid: -- Hz")
        self.spectral_centroid_label.pack(anchor=tk.W)
        
        self.spectral_rolloff_label = ttk.Label(metrics_frame, text="Spectral Rolloff: -- Hz")
        self.spectral_rolloff_label.pack(anchor=tk.W)
    
    def create_parameter_controls(self, parent):
        """Create parameter control panels"""
        # Savitzky-Golay parameters
        self.savgol_frame = ttk.LabelFrame(parent, text="Savitzky-Golay Parameters", padding=10)
        self.savgol_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.savgol_frame, text="Window Length:").pack(anchor=tk.W)
        ttk.Scale(self.savgol_frame, from_=5, to=101, variable=self.savgol_window_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.savgol_frame, text="Polynomial Order:").pack(anchor=tk.W)
        ttk.Scale(self.savgol_frame, from_=1, to=5, variable=self.savgol_poly_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.savgol_frame, text="Derivative Order:").pack(anchor=tk.W)
        ttk.Scale(self.savgol_frame, from_=0, to=3, variable=self.savgol_deriv_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.savgol_frame, text="Delta (Derivative):").pack(anchor=tk.W)
        ttk.Scale(self.savgol_frame, from_=0.1, to=5.0, variable=self.savgol_delta_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        # Median filter parameters
        self.median_frame = ttk.LabelFrame(parent, text="Median Filter Parameters", padding=10)
        self.median_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.median_frame, text="Kernel Size:").pack(anchor=tk.W)
        ttk.Scale(self.median_frame, from_=3, to=21, variable=self.median_kernel_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        # Wiener filter parameters
        self.wiener_frame = ttk.LabelFrame(parent, text="Wiener Filter Parameters", padding=10)
        self.wiener_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.wiener_frame, text="Noise Variance:").pack(anchor=tk.W)
        ttk.Scale(self.wiener_frame, from_=0.01, to=1.0, variable=self.wiener_noise_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.wiener_frame, text="Filter Size:").pack(anchor=tk.W)
        ttk.Scale(self.wiener_frame, from_=3, to=15, variable=self.wiener_mysize_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Checkbutton(self.wiener_frame, text="Auto Noise Estimation", 
                       variable=self.wiener_noise_var_auto, command=self.on_parameter_change).pack(anchor=tk.W)
        
        # Wavelet approximation parameters
        self.wavelet_frame = ttk.LabelFrame(parent, text="Wavelet Approximation Parameters", padding=10)
        self.wavelet_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.wavelet_frame, text="Cutoff Frequency (Hz):").pack(anchor=tk.W)
        ttk.Scale(self.wavelet_frame, from_=1.0, to=50.0, variable=self.wavelet_cutoff_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.wavelet_frame, text="Filter Order:").pack(anchor=tk.W)
        ttk.Scale(self.wavelet_frame, from_=2, to=10, variable=self.wavelet_order_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.wavelet_frame, text="Window Size:").pack(anchor=tk.W)
        ttk.Scale(self.wavelet_frame, from_=64, to=512, variable=self.wavelet_window_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.wavelet_frame, text="Overlap Size:").pack(anchor=tk.W)
        ttk.Scale(self.wavelet_frame, from_=32, to=256, variable=self.wavelet_overlap_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        # Heart rate filter parameters
        self.heart_rate_frame = ttk.LabelFrame(parent, text="Heart Rate Filter Parameters", padding=10)
        self.heart_rate_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.heart_rate_frame, text="Low Frequency (Hz):").pack(anchor=tk.W)
        ttk.Scale(self.heart_rate_frame, from_=0.5, to=2.0, variable=self.hr_low_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.heart_rate_frame, text="High Frequency (Hz):").pack(anchor=tk.W)
        ttk.Scale(self.heart_rate_frame, from_=2.0, to=5.0, variable=self.hr_high_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.heart_rate_frame, text="Filter Order:").pack(anchor=tk.W)
        ttk.Scale(self.heart_rate_frame, from_=2, to=8, variable=self.hr_order_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.heart_rate_frame, text="Smoothing Window:").pack(anchor=tk.W)
        ttk.Scale(self.heart_rate_frame, from_=1, to=10, variable=self.hr_smooth_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        # Bandpass filter parameters
        self.bandpass_frame = ttk.LabelFrame(parent, text="Bandpass Filter Parameters", padding=10)
        self.bandpass_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.bandpass_frame, text="Low Frequency (Hz):").pack(anchor=tk.W)
        ttk.Scale(self.bandpass_frame, from_=0.1, to=5.0, variable=self.bandpass_low_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.bandpass_frame, text="High Frequency (Hz):").pack(anchor=tk.W)
        ttk.Scale(self.bandpass_frame, from_=1.0, to=10.0, variable=self.bandpass_high_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.bandpass_frame, text="Filter Order:").pack(anchor=tk.W)
        ttk.Scale(self.bandpass_frame, from_=2, to=8, variable=self.bandpass_order_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        # Adaptive filter parameters
        self.adaptive_frame = ttk.LabelFrame(parent, text="Adaptive Filter Parameters", padding=10)
        self.adaptive_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.adaptive_frame, text="Window Size:").pack(anchor=tk.W)
        ttk.Scale(self.adaptive_frame, from_=5, to=50, variable=self.adaptive_window_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        ttk.Label(self.adaptive_frame, text="Method:").pack(anchor=tk.W)
        self.adaptive_method_combo = ttk.Combobox(self.adaptive_frame, textvariable=self.adaptive_method_var,
                    values=["moving_avg", "exponential", "gaussian"], state="readonly")
        self.adaptive_method_combo.pack(fill=tk.X)
        self.adaptive_method_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)
        
        ttk.Label(self.adaptive_frame, text="Learning Rate (α):").pack(anchor=tk.W)
        ttk.Scale(self.adaptive_frame, from_=0.01, to=1.0, variable=self.adaptive_alpha_var,
                 orient=tk.HORIZONTAL, command=self.on_parameter_change).pack(fill=tk.X)
        
        # Combined filter parameters
        self.combined_frame = ttk.LabelFrame(parent, text="Combined Filter Parameters", padding=10)
        self.combined_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.combined_frame, text="Method 1:").pack(anchor=tk.W)
        self.combined_method1_combo = ttk.Combobox(self.combined_frame, textvariable=self.combined_method1_var,
                    values=["median", "wiener", "wavelet_approx"], state="readonly")
        self.combined_method1_combo.pack(fill=tk.X)
        self.combined_method1_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)
        
        ttk.Label(self.combined_frame, text="Method 2:").pack(anchor=tk.W)
        self.combined_method2_combo = ttk.Combobox(self.combined_frame, textvariable=self.combined_method2_var,
                    values=["savgol", "bandpass", "adaptive"], state="readonly")
        self.combined_method2_combo.pack(fill=tk.X)
        self.combined_method2_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)
        
        # Initially hide all parameter frames except the selected method
        self.update_parameter_visibility()
    
    def create_plot_panel(self, parent):
        """Create plot panel"""
        # Create figure with subplots (2x3 layout for more space)
        self.fig = Figure(figsize=(16, 10), dpi=100)
        
        # Time domain plot
        self.ax_time = self.fig.add_subplot(2, 3, 1)
        self.ax_time.set_title("Time Domain Signal")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.grid(True, alpha=0.3)
        
        # Frequency domain plot
        self.ax_freq = self.fig.add_subplot(2, 3, 2)
        self.ax_freq.set_title("Frequency Domain")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        self.ax_freq.grid(True, alpha=0.3)
        
        # STFT spectrogram - Original
        self.ax_stft_orig = self.fig.add_subplot(2, 3, 3)
        self.ax_stft_orig.set_title("STFT - Original Signal")
        self.ax_stft_orig.set_xlabel("Time (s)")
        self.ax_stft_orig.set_ylabel("Frequency (Hz)")
        
        # STFT spectrogram - Denoised
        self.ax_stft_denoised = self.fig.add_subplot(2, 3, 4)
        self.ax_stft_denoised.set_title("STFT - Denoised Signal")
        self.ax_stft_denoised.set_xlabel("Time (s)")
        self.ax_stft_denoised.set_ylabel("Frequency (Hz)")
        
        # Difference plot
        self.ax_diff = self.fig.add_subplot(2, 3, 5)
        self.ax_diff.set_title("Signal Difference (Original - Denoised)")
        self.ax_diff.set_xlabel("Time (s)")
        self.ax_diff.set_ylabel("Difference")
        self.ax_diff.grid(True, alpha=0.3)
        
        # Spectral features plot
        self.ax_spectral = self.fig.add_subplot(2, 3, 6)
        self.ax_spectral.set_title("Spectral Features")
        self.ax_spectral.set_xlabel("Time (s)")
        self.ax_spectral.set_ylabel("Frequency (Hz)")
        self.ax_spectral.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
    def setup_plots(self):
        """Setup initial plots"""
        # Initialize empty plots
        self.ax_time.clear()
        self.ax_time.set_title("Time Domain Signal")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.grid(True, alpha=0.3)
        
        self.ax_freq.clear()
        self.ax_freq.set_title("Frequency Domain")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        self.ax_freq.grid(True, alpha=0.3)
        
        self.ax_stft_orig.clear()
        self.ax_stft_orig.set_title("STFT - Original Signal")
        self.ax_stft_orig.set_xlabel("Time (s)")
        self.ax_stft_orig.set_ylabel("Frequency (Hz)")
        
        self.ax_stft_denoised.clear()
        self.ax_stft_denoised.set_title("STFT - Denoised Signal")
        self.ax_stft_denoised.set_xlabel("Time (s)")
        self.ax_stft_denoised.set_ylabel("Frequency (Hz)")
        
        self.ax_diff.clear()
        self.ax_diff.set_title("Signal Difference (Original - Denoised)")
        self.ax_diff.set_xlabel("Time (s)")
        self.ax_diff.set_ylabel("Difference")
        self.ax_diff.grid(True, alpha=0.3)
        
        self.ax_spectral.clear()
        self.ax_spectral.set_title("Spectral Features")
        self.ax_spectral.set_xlabel("Time (s)")
        self.ax_spectral.set_ylabel("Frequency (Hz)")
        self.ax_spectral.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def load_default_signal(self):
        """Load a default signal for demonstration"""
        # Look for available data files
        data_dirs = ['hyx_data', 'lhr_data', 'lj_data']
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                txt_files = list(Path(data_dir).glob("*.txt"))
                if txt_files:
                    self.load_signal_file(str(txt_files[0]))
                    break
    
    def load_signal(self):
        """Load signal from file"""
        file_path = filedialog.askopenfilename(
            title="Select PPG Signal File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_signal_file(file_path)
    
    def load_signal_file(self, file_path):
        """Load signal from specific file"""
        try:
            # Try different encodings
            try:
                data = np.loadtxt(file_path, skiprows=1, encoding='utf-8')
            except:
                data = np.loadtxt(file_path, skiprows=1, encoding='gbk')
            
            if len(data.shape) == 1:
                self.original_signal = data
                # If single column, assume it's signal data only
                self.time_axis = np.arange(len(self.original_signal)) / self.fs
            else:
                # If multiple columns, first column might be time
                if data.shape[1] >= 2:
                    # Check if first column looks like time data
                    time_col = data[:, 0]
                    signal_col = data[:, 1]
                    
                    # If time column starts from 0 and increases, use it
                    if np.all(np.diff(time_col) > 0) and time_col[0] >= 0:
                        self.time_axis = time_col
                        self.original_signal = signal_col
                        print(f"Using time column from file: {time_col[0]:.3f} - {time_col[-1]:.3f} s")
                    else:
                        # Use signal column and generate time axis
                        self.original_signal = signal_col
                        self.time_axis = np.arange(len(self.original_signal)) / self.fs
                        print(f"Generated time axis: 0 - {self.time_axis[-1]:.3f} s")
                else:
                    self.original_signal = data[:, 0]
                    self.time_axis = np.arange(len(self.original_signal)) / self.fs
            
            self.current_signal = self.original_signal.copy()
            
            # Debug info
            print(f"Signal loaded: {len(self.original_signal)} samples")
            print(f"Time range: {self.time_axis[0]:.3f} - {self.time_axis[-1]:.3f} s")
            print(f"Duration: {self.time_axis[-1] - self.time_axis[0]:.3f} s")
            print(f"Sampling rate: {self.fs} Hz")
            
            # Update plots
            self.update_plot()
            
            messagebox.showinfo("Success", f"Signal loaded successfully!\nSamples: {len(self.original_signal)}\nDuration: {self.time_axis[-1]:.1f}s")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load signal: {e}")
    
    def apply_denoising(self):
        """Apply denoising with current parameters"""
        if self.original_signal is None:
            messagebox.showwarning("Warning", "Please load a signal first!")
            return
        
        try:
            import time
            start_time = time.time()
            method = self.method_var.get()
            
            if method == "savgol":
                window_length = self.savgol_window_var.get()
                poly_order = self.savgol_poly_var.get()
                deriv = self.savgol_deriv_var.get()
                delta = self.savgol_delta_var.get()
                if window_length % 2 == 0:
                    window_length += 1
                self.denoised_signal = savgol_filter(self.current_signal, window_length, poly_order, 
                                                   deriv=deriv, delta=delta)
            
            elif method == "median":
                kernel_size = self.median_kernel_var.get()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                self.denoised_signal = medfilt(self.current_signal, kernel_size)
            
            elif method == "wiener":
                noise_var = self.wiener_noise_var.get()
                mysize = self.wiener_mysize_var.get()
                auto_noise = self.wiener_noise_var_auto.get()
                
                if auto_noise:
                    # Auto-estimate noise variance
                    noise_var = np.var(self.current_signal - savgol_filter(self.current_signal, 51, 3))
                
                self.denoised_signal = wiener(self.current_signal, noise=noise_var, mysize=mysize)
            
            elif method == "wavelet_approx":
                cutoff = self.wavelet_cutoff_var.get()
                order = self.wavelet_order_var.get()
                window_size = self.wavelet_window_var.get()
                overlap = self.wavelet_overlap_var.get()
                
                # Apply wavelet approximation with windowing
                b, a = butter(order, cutoff / (self.fs/2), btype='low')
                
                # Apply windowed filtering for better edge handling
                signal_len = len(self.current_signal)
                if signal_len > window_size:
                    # Use overlapping windows
                    step_size = window_size - overlap
                    denoised_parts = []
                    
                    for start in range(0, signal_len - window_size + 1, step_size):
                        end = min(start + window_size, signal_len)
                        window_signal = self.current_signal[start:end]
                        window_denoised = filtfilt(b, a, window_signal)
                        denoised_parts.append(window_denoised)
                    
                    # Combine overlapping windows
                    self.denoised_signal = np.zeros_like(self.current_signal)
                    weights = np.zeros_like(self.current_signal)
                    
                    for i, denoised_part in enumerate(denoised_parts):
                        start = i * step_size
                        end = start + len(denoised_part)
                        self.denoised_signal[start:end] += denoised_part
                        weights[start:end] += 1
                    
                    # Normalize by weights
                    self.denoised_signal = np.divide(self.denoised_signal, weights, 
                                                   out=np.zeros_like(self.denoised_signal), 
                                                   where=weights!=0)
                else:
                    # Signal too short for windowing
                    self.denoised_signal = filtfilt(b, a, self.current_signal)
            
            elif method == "heart_rate":
                # Heart rate specific filtering to remove spikes/artifacts
                hr_low = self.hr_low_var.get()
                hr_high = self.hr_high_var.get()
                hr_order = self.hr_order_var.get()
                hr_smooth = self.hr_smooth_var.get()
                
                # Step 1: Apply heart rate bandpass filter
                b, a = butter(hr_order, [hr_low/(self.fs/2), hr_high/(self.fs/2)], btype='band')
                filtered_signal = filtfilt(b, a, self.current_signal)
                
                # Step 2: Apply additional smoothing to remove spikes
                if hr_smooth > 1:
                    # Use moving average for additional smoothing
                    kernel = np.ones(hr_smooth) / hr_smooth
                    filtered_signal = np.convolve(filtered_signal, kernel, mode='same')
                
                # Step 3: Apply Savitzky-Golay filter for final smoothing
                window_length = min(51, len(filtered_signal) // 4)
                if window_length % 2 == 0:
                    window_length += 1
                
                self.denoised_signal = savgol_filter(filtered_signal, window_length, 3)
            
            elif method == "bandpass":
                low_freq = self.bandpass_low_var.get()
                high_freq = self.bandpass_high_var.get()
                order = self.bandpass_order_var.get()
                b, a = butter(order, [low_freq/(self.fs/2), high_freq/(self.fs/2)], btype='band')
                self.denoised_signal = filtfilt(b, a, self.current_signal)
            
            elif method == "adaptive":
                window_size = self.adaptive_window_var.get()
                adaptive_method = self.adaptive_method_var.get()
                alpha = self.adaptive_alpha_var.get()
                
                if adaptive_method == "moving_avg":
                    self.denoised_signal = np.convolve(
                        self.current_signal, 
                        np.ones(window_size)/window_size, 
                        mode='same'
                    )
                elif adaptive_method == "exponential":
                    # Exponential moving average
                    self.denoised_signal = np.zeros_like(self.current_signal)
                    self.denoised_signal[0] = self.current_signal[0]
                    for i in range(1, len(self.current_signal)):
                        self.denoised_signal[i] = alpha * self.current_signal[i] + (1 - alpha) * self.denoised_signal[i-1]
                elif adaptive_method == "gaussian":
                    # Gaussian-weighted moving average
                    sigma = window_size / 6.0  # 3-sigma rule
                    x = np.arange(-window_size//2, window_size//2 + 1)
                    weights = np.exp(-(x**2) / (2 * sigma**2))
                    weights = weights / np.sum(weights)
                    self.denoised_signal = np.convolve(self.current_signal, weights, mode='same')
            
            elif method == "combined":
                # Apply first method
                method1 = self.combined_method1_var.get()
                method2 = self.combined_method2_var.get()
                
                # Apply method1
                if method1 == "median":
                    kernel_size = self.median_kernel_var.get()
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    temp_signal = medfilt(self.current_signal, kernel_size)
                elif method1 == "wiener":
                    noise_var = self.wiener_noise_var.get()
                    temp_signal = wiener(self.current_signal, noise=noise_var)
                elif method1 == "wavelet_approx":
                    cutoff = self.wavelet_cutoff_var.get()
                    order = self.wavelet_order_var.get()
                    b, a = butter(order, cutoff / (self.fs/2), btype='low')
                    temp_signal = filtfilt(b, a, self.current_signal)
                
                # Apply method2
                if method2 == "savgol":
                    window_length = self.savgol_window_var.get()
                    poly_order = self.savgol_poly_var.get()
                    if window_length % 2 == 0:
                        window_length += 1
                    self.denoised_signal = savgol_filter(temp_signal, window_length, poly_order)
                elif method2 == "bandpass":
                    low_freq = self.bandpass_low_var.get()
                    high_freq = self.bandpass_high_var.get()
                    order = self.bandpass_order_var.get()
                    b, a = butter(order, [low_freq/(self.fs/2), high_freq/(self.fs/2)], btype='band')
                    self.denoised_signal = filtfilt(b, a, temp_signal)
                elif method2 == "adaptive":
                    window_size = self.adaptive_window_var.get()
                    self.denoised_signal = np.convolve(
                        temp_signal, 
                        np.ones(window_size)/window_size, 
                        mode='same'
                    )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update plots and metrics
            self.update_plot()
            self.update_metrics()
            
            # Update processing time label
            self.processing_time_label.config(text=f"Processing Time: {processing_time:.1f} ms")
            
        except Exception as e:
            messagebox.showerror("Error", f"Denoising failed: {e}")
    
    def compute_stft(self, signal_data):
        """Compute STFT for a signal"""
        nperseg = self.stft_window_var.get()
        noverlap = self.stft_overlap_var.get()
        nfft = self.stft_nfft_var.get()
        window_type = self.stft_window_type_var.get()
        
        try:
            # Ensure signal is valid
            if signal_data is None or len(signal_data) == 0:
                print("STFT: Invalid signal data")
                return None, None, None
            
            # Ensure parameters are reasonable
            if nperseg > len(signal_data):
                nperseg = len(signal_data) // 2
                if nperseg % 2 == 0:
                    nperseg += 1
            
            if noverlap >= nperseg:
                noverlap = nperseg // 2
            
            # Ensure nfft is reasonable
            if nfft < nperseg:
                nfft = nperseg
            
            f, t, Zxx = signal.stft(signal_data, fs=self.fs, window=window_type, 
                                   nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            
            # Debug info
            print(f"STFT computed: f={len(f)}, t={len(t)}, Zxx={Zxx.shape}")
            print(f"  Frequency range: {f[0]:.1f} - {f[-1]:.1f} Hz")
            print(f"  Time range: {t[0]:.1f} - {t[-1]:.1f} s")
            print(f"  Magnitude range: {np.min(np.abs(Zxx)):.6f} - {np.max(np.abs(Zxx)):.6f}")
            print(f"  Parameters: nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}, fs={self.fs}")
            
            return f, t, Zxx
        except Exception as e:
            print(f"STFT computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def compute_spectral_features(self, signal):
        """Compute spectral features"""
        try:
            # Compute STFT
            f, t, Zxx = self.compute_stft(signal)
            if f is None:
                return None, None
            
            # Spectral centroid
            magnitude = np.abs(Zxx)
            spectral_centroid = np.sum(f[:, np.newaxis] * magnitude, axis=0) / np.sum(magnitude, axis=0)
            
            # Spectral rolloff (95% energy)
            cumsum_magnitude = np.cumsum(magnitude, axis=0)
            total_energy = cumsum_magnitude[-1, :]
            rolloff_threshold = 0.95 * total_energy
            
            spectral_rolloff = np.zeros(len(t))
            for i in range(len(t)):
                idx = np.where(cumsum_magnitude[:, i] >= rolloff_threshold[i])[0]
                if len(idx) > 0:
                    spectral_rolloff[i] = f[idx[0]]
            
            return spectral_centroid, spectral_rolloff
        except Exception as e:
            print(f"Spectral features computation failed: {e}")
            return None, None
    
    def update_plot(self):
        """Update all plots"""
        if self.original_signal is None:
            return
        
        # Clear all axes
        self.ax_time.clear()
        self.ax_freq.clear()
        self.ax_stft_orig.clear()
        self.ax_stft_denoised.clear()
        self.ax_diff.clear()
        self.ax_spectral.clear()
        
        # Time domain plot
        self.ax_time.set_title("Time Domain Signal")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.grid(True, alpha=0.3)
        
        if self.show_original_var.get():
            self.ax_time.plot(self.time_axis, self.original_signal, 'b-', 
                           linewidth=1, alpha=0.7, label='Original')
        
        if self.show_denoised_var.get() and self.denoised_signal is not None:
            self.ax_time.plot(self.time_axis, self.denoised_signal, 'r-', 
                           linewidth=1.5, label='Denoised')
        
        if self.show_noise_var.get() and self.denoised_signal is not None:
            noise = self.original_signal - self.denoised_signal
            self.ax_time.plot(self.time_axis, noise, 'g--', 
                           linewidth=1, alpha=0.7, label='Noise')
        
        self.ax_time.legend()
        
        # Add metrics text if enabled
        if self.show_metrics_var.get() and self.denoised_signal is not None:
            noise = self.original_signal - self.denoised_signal
            signal_power = np.var(self.denoised_signal)
            noise_power = np.var(noise)
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            std_reduction = np.std(self.original_signal) - np.std(self.denoised_signal)
            
            metrics_text = f'SNR: {snr_db:.1f} dB\nStd Reduction: {std_reduction:.3f}'
            self.ax_time.text(0.02, 0.98, metrics_text, transform=self.ax_time.transAxes,
                             verticalalignment='top', fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Frequency domain plot
        self.ax_freq.set_title("Frequency Domain")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        self.ax_freq.grid(True, alpha=0.3)
        
        if self.show_original_var.get():
            fft_orig = np.fft.fft(self.original_signal)
            freqs = np.fft.fftfreq(len(self.original_signal), 1/self.fs)
            mask = freqs >= 0
            self.ax_freq.plot(freqs[mask], np.abs(fft_orig[mask]), 'b-', 
                           linewidth=1, alpha=0.7, label='Original')
        
        if self.show_denoised_var.get() and self.denoised_signal is not None:
            fft_denoised = np.fft.fft(self.denoised_signal)
            freqs = np.fft.fftfreq(len(self.denoised_signal), 1/self.fs)
            mask = freqs >= 0
            self.ax_freq.plot(freqs[mask], np.abs(fft_denoised[mask]), 'r-', 
                           linewidth=1.5, label='Denoised')
        
        self.ax_freq.set_xlim(0, 50)  # Show up to 50 Hz
        self.ax_freq.legend()
        
        # STFT spectrograms
        if self.show_stft_var.get() and self.real_time_stft_var.get():
            # Original signal STFT
            self.ax_stft_orig.set_title("STFT - Original Signal")
            self.ax_stft_orig.set_xlabel("Time (s)")
            self.ax_stft_orig.set_ylabel("Frequency (Hz)")
            
            f_orig, t_orig, Zxx_orig = self.compute_stft(self.original_signal)
            if f_orig is not None and t_orig is not None and Zxx_orig is not None:
                max_freq = self.stft_freq_range_var.get()
                freq_mask = f_orig <= max_freq
                
                # Check if we have valid data
                if np.sum(freq_mask) > 0 and Zxx_orig[freq_mask, :].size > 0:
                    magnitude_orig = np.abs(Zxx_orig[freq_mask, :])
                    
                    # Ensure data is finite
                    if np.all(np.isfinite(magnitude_orig)) and np.max(magnitude_orig) > 0:
                        # Use imshow instead of pcolormesh for better display
                        extent = [t_orig[0], t_orig[-1], f_orig[freq_mask][0], f_orig[freq_mask][-1]]
                        im_orig = self.ax_stft_orig.imshow(magnitude_orig, 
                                                          aspect='auto', origin='lower',
                                                          extent=extent, cmap='jet')
                        
                        # Add single colorbar
                        if not hasattr(self, 'cbar_orig'):
                            self.cbar_orig = self.fig.colorbar(im_orig, ax=self.ax_stft_orig, label='Magnitude')
                        else:
                            self.cbar_orig.update_normal(im_orig)
                        
                        # Set proper axis limits
                        self.ax_stft_orig.set_ylim(f_orig[freq_mask][0], f_orig[freq_mask][-1])
                        self.ax_stft_orig.set_xlim(t_orig[0], t_orig[-1])
                    else:
                        self.ax_stft_orig.text(0.5, 0.5, 'No valid STFT data', 
                                             transform=self.ax_stft_orig.transAxes,
                                             ha='center', va='center')
                else:
                    self.ax_stft_orig.text(0.5, 0.5, 'No frequencies in range', 
                                         transform=self.ax_stft_orig.transAxes,
                                         ha='center', va='center')
            
            # Denoised signal STFT
            if self.denoised_signal is not None:
                self.ax_stft_denoised.set_title("STFT - Denoised Signal")
                self.ax_stft_denoised.set_xlabel("Time (s)")
                self.ax_stft_denoised.set_ylabel("Frequency (Hz)")
                
                f_denoised, t_denoised, Zxx_denoised = self.compute_stft(self.denoised_signal)
                if f_denoised is not None and t_denoised is not None and Zxx_denoised is not None:
                    freq_mask = f_denoised <= max_freq
                    
                    # Check if we have valid data
                    if np.sum(freq_mask) > 0 and Zxx_denoised[freq_mask, :].size > 0:
                        magnitude_denoised = np.abs(Zxx_denoised[freq_mask, :])
                        
                        # Ensure data is finite
                        if np.all(np.isfinite(magnitude_denoised)) and np.max(magnitude_denoised) > 0:
                            # Use imshow instead of pcolormesh for better display
                            extent = [t_denoised[0], t_denoised[-1], f_denoised[freq_mask][0], f_denoised[freq_mask][-1]]
                            im_denoised = self.ax_stft_denoised.imshow(magnitude_denoised, 
                                                                      aspect='auto', origin='lower',
                                                                      extent=extent, cmap='jet')
                            
                            # Add single colorbar
                            if not hasattr(self, 'cbar_denoised'):
                                self.cbar_denoised = self.fig.colorbar(im_denoised, ax=self.ax_stft_denoised, label='Magnitude')
                            else:
                                self.cbar_denoised.update_normal(im_denoised)
                            
                            # Set proper axis limits
                            self.ax_stft_denoised.set_ylim(f_denoised[freq_mask][0], f_denoised[freq_mask][-1])
                            self.ax_stft_denoised.set_xlim(t_denoised[0], t_denoised[-1])
                        else:
                            self.ax_stft_denoised.text(0.5, 0.5, 'No valid STFT data', 
                                                     transform=self.ax_stft_denoised.transAxes,
                                                     ha='center', va='center')
                    else:
                        self.ax_stft_denoised.text(0.5, 0.5, 'No frequencies in range', 
                                                 transform=self.ax_stft_denoised.transAxes,
                                                 ha='center', va='center')
            
            # Spectral features
            self.ax_spectral.set_title("Spectral Features")
            self.ax_spectral.set_xlabel("Time (s)")
            self.ax_spectral.set_ylabel("Frequency (Hz)")
            
            if self.denoised_signal is not None:
                # Compute spectral features for both signals
                centroid_orig, rolloff_orig = self.compute_spectral_features(self.original_signal)
                centroid_denoised, rolloff_denoised = self.compute_spectral_features(self.denoised_signal)
                
                if (centroid_orig is not None and centroid_denoised is not None and 
                    rolloff_orig is not None and rolloff_denoised is not None):
                    
                    # Get time axis from STFT
                    if f_orig is not None and t_orig is not None:
                        # Plot spectral centroid
                        self.ax_spectral.plot(t_orig, centroid_orig, 'b-', linewidth=2, label='Centroid (Orig)')
                        self.ax_spectral.plot(t_orig, centroid_denoised, 'r-', linewidth=2, label='Centroid (Denoised)')
                        
                        # Plot spectral rolloff
                        self.ax_spectral.plot(t_orig, rolloff_orig, 'b--', linewidth=1, alpha=0.7, label='Rolloff (Orig)')
                        self.ax_spectral.plot(t_orig, rolloff_denoised, 'r--', linewidth=1, alpha=0.7, label='Rolloff (Denoised)')
                        
                        self.ax_spectral.legend()
                        self.ax_spectral.grid(True, alpha=0.3)
                    else:
                        self.ax_spectral.text(0.5, 0.5, 'No STFT data for spectral features', 
                                            transform=self.ax_spectral.transAxes,
                                            ha='center', va='center')
                else:
                    self.ax_spectral.text(0.5, 0.5, 'Failed to compute spectral features', 
                                        transform=self.ax_spectral.transAxes,
                                        ha='center', va='center')
        
        # Difference plot
        if self.denoised_signal is not None:
            self.ax_diff.set_title("Signal Difference (Original - Denoised)")
            self.ax_diff.set_xlabel("Time (s)")
            self.ax_diff.set_ylabel("Difference")
            self.ax_diff.grid(True, alpha=0.3)
            
            diff = self.original_signal - self.denoised_signal
            self.ax_diff.plot(self.time_axis, diff, 'g-', linewidth=1, label='Noise')
            self.ax_diff.legend()
        
        self.fig.tight_layout()
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
        
        # Calculate other metrics
        std_reduction = np.std(self.original_signal) - np.std(self.denoised_signal)
        peak_to_peak = np.ptp(self.denoised_signal)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((self.original_signal - self.denoised_signal)**2))
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(self.original_signal, self.denoised_signal)[0, 1]
        
        # Calculate additional metrics
        mae = np.mean(np.abs(self.original_signal - self.denoised_signal))
        
        # PSNR calculation
        max_val = np.max(self.original_signal)
        if rmse > 0:
            psnr_db = 20 * np.log10(max_val / rmse)
        else:
            psnr_db = float('inf')
        
        # SSIM approximation (simplified version)
        mu1 = np.mean(self.original_signal)
        mu2 = np.mean(self.denoised_signal)
        sigma1 = np.var(self.original_signal)
        sigma2 = np.var(self.denoised_signal)
        sigma12 = np.mean((self.original_signal - mu1) * (self.denoised_signal - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        # Spectral features
        centroid_orig, rolloff_orig = self.compute_spectral_features(self.original_signal)
        centroid_denoised, rolloff_denoised = self.compute_spectral_features(self.denoised_signal)
        
        avg_centroid_orig = np.mean(centroid_orig) if centroid_orig is not None else 0
        avg_centroid_denoised = np.mean(centroid_denoised) if centroid_denoised is not None else 0
        avg_rolloff_orig = np.mean(rolloff_orig) if rolloff_orig is not None else 0
        avg_rolloff_denoised = np.mean(rolloff_denoised) if rolloff_denoised is not None else 0
        
        # Update labels
        self.snr_label.config(text=f"SNR: {snr_db:.1f} dB")
        self.std_label.config(text=f"Std Reduction: {std_reduction:.3f}")
        self.peak_label.config(text=f"Peak-to-Peak: {peak_to_peak:.1f}")
        self.rmse_label.config(text=f"RMSE: {rmse:.3f}")
        self.correlation_label.config(text=f"Correlation: {correlation:.3f}")
        self.mae_label.config(text=f"MAE: {mae:.3f}")
        self.psnr_label.config(text=f"PSNR: {psnr_db:.1f} dB")
        self.ssim_label.config(text=f"SSIM: {ssim:.3f}")
        self.spectral_centroid_label.config(text=f"Spectral Centroid: {avg_centroid_denoised:.1f} Hz")
        self.spectral_rolloff_label.config(text=f"Spectral Rolloff: {avg_rolloff_denoised:.1f} Hz")
    
    def update_parameter_visibility(self):
        """Update visibility of parameter frames based on selected method"""
        method = self.method_var.get()
        
        # Hide all frames
        frames = [self.savgol_frame, self.median_frame, self.wiener_frame, 
                 self.wavelet_frame, self.heart_rate_frame, self.bandpass_frame, 
                 self.adaptive_frame, self.combined_frame]
        
        for frame in frames:
            frame.pack_forget()
        
        # Show relevant frame
        if method == "savgol":
            self.savgol_frame.pack(fill=tk.X, pady=(0, 5))
        elif method == "median":
            self.median_frame.pack(fill=tk.X, pady=(0, 5))
        elif method == "wiener":
            self.wiener_frame.pack(fill=tk.X, pady=(0, 5))
        elif method == "wavelet_approx":
            self.wavelet_frame.pack(fill=tk.X, pady=(0, 5))
        elif method == "heart_rate":
            self.heart_rate_frame.pack(fill=tk.X, pady=(0, 5))
        elif method == "bandpass":
            self.bandpass_frame.pack(fill=tk.X, pady=(0, 5))
        elif method == "adaptive":
            self.adaptive_frame.pack(fill=tk.X, pady=(0, 5))
        elif method == "combined":
            self.combined_frame.pack(fill=tk.X, pady=(0, 5))
    
    def on_method_change(self):
        """Handle method change"""
        self.update_parameter_visibility()
        self.apply_denoising()
    
    def on_parameter_change(self, *args):
        """Handle parameter change"""
        self.apply_denoising()
    
    def on_analysis_change(self, *args):
        """Handle analysis parameter change"""
        self.update_plot()
    
    def on_stft_change(self, *args):
        """Handle STFT parameter change"""
        if self.real_time_stft_var.get():
            self.update_plot()
    
    def reset_signal(self):
        """Reset to original signal"""
        if self.original_signal is not None:
            self.current_signal = self.original_signal.copy()
            self.denoised_signal = None
            self.update_plot()
            self.update_metrics()
    
    def save_denoised(self):
        """Save denoised signal"""
        if self.denoised_signal is None:
            messagebox.showwarning("Warning", "Please apply denoising first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Denoised Signal",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Prepare data: time and denoised signal
                time_column = self.time_axis.reshape(-1, 1)
                signal_column = self.denoised_signal.reshape(-1, 1)
                data_to_save = np.hstack([time_column, signal_column])
                
                np.savetxt(file_path, data_to_save, 
                          fmt='%.6f', delimiter='\t',
                          header=f'Time(s)\tDenoised_Signal_{self.method_var.get()}',
                          comments='')
                
                messagebox.showinfo("Success", f"Denoised signal saved to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save signal: {e}")
    
    def export_results(self):
        """Export analysis results"""
        if self.denoised_signal is None:
            messagebox.showwarning("Warning", "Please apply denoising first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Analysis Results",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Save the current figure
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Analysis results exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")

def main():
    """Main function"""
    root = tk.Tk()
    app = PPGDenoisingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

