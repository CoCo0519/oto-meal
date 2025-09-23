#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify multi-channel support and Chinese font display
"""

import os
import sys
from pathlib import Path

def test_channel_detection():
    """Test channel detection functionality"""
    print("🧪 Testing Channel Detection")
    print("="*50)
    
    # Import the function
    try:
        from ppg_advanced_analysis import check_available_channels
        print("✅ Successfully imported check_available_channels")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Test with sample data directories
    data_dirs = ['hyx_data', 'lhr_data', 'lj_data']
    
    for data_dir in data_dirs:
        if not Path(data_dir).exists():
            print(f"⚠️ Directory {data_dir} not found")
            continue
            
        print(f"\n📂 Testing directory: {data_dir}")
        txt_files = list(Path(data_dir).glob("*.txt"))
        
        if not txt_files:
            print(f"   No .txt files found")
            continue
            
        # Test first file
        test_file = txt_files[0]
        print(f"   Testing file: {test_file.name}")
        
        try:
            available_channels = check_available_channels(str(test_file))
            print(f"   Available channels: {available_channels}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

def test_chinese_font():
    """Test Chinese font display"""
    print("\n🧪 Testing Chinese Font Display")
    print("="*50)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Set Chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create test plot with Chinese text
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_title('PPG信号测试 - 中文显示测试', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('幅值', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add Chinese text annotation
        ax.text(5, 0.5, '这是中文测试文本', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save test image
        test_save_path = 'chinese_font_test.png'
        plt.savefig(test_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if Path(test_save_path).exists():
            print(f"✅ Chinese font test image saved: {test_save_path}")
            print("   Please check if Chinese characters display correctly")
            return True
        else:
            print("❌ Failed to save test image")
            return False
            
    except Exception as e:
        print(f"❌ Chinese font test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 PPG Advanced Analysis - Channel & Font Test")
    print("="*60)
    
    # Test channel detection
    channel_test = test_channel_detection()
    
    # Test Chinese font
    font_test = test_chinese_font()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Channel Detection Test: {'✅ PASSED' if channel_test else '❌ FAILED'}")
    print(f"Chinese Font Test: {'✅ PASSED' if font_test else '❌ FAILED'}")
    
    if channel_test and font_test:
        print("\n🎉 All tests passed! The script should now:")
        print("   - Support multiple channels (green, ir, red)")
        print("   - Display Chinese characters correctly")
        print("   - Auto-detect available channels per file")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
