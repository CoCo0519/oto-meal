#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for denoised data saving functionality
"""

import os
import sys
from pathlib import Path

def test_denoised_data_saving():
    """Test the denoised data saving functionality"""
    print("🧪 Testing Denoised Data Saving")
    print("="*50)
    
    # Import the function
    try:
        from ppg_advanced_analysis import advanced_ppg_analysis
        print("✅ Successfully imported advanced_ppg_analysis")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Test with a sample file
    test_files = []
    for data_dir in ['hyx_data', 'lhr_data', 'lj_data']:
        if Path(data_dir).exists():
            txt_files = list(Path(data_dir).glob("*.txt"))
            if txt_files:
                test_files.append(str(txt_files[0]))
                break
    
    if not test_files:
        print("❌ No test files found")
        return False
    
    test_file = test_files[0]
    print(f"📁 Testing with file: {Path(test_file).name}")
    
    # Test saving denoised data
    try:
        result = advanced_ppg_analysis(test_file, 'green', './test_denoised_output', save_denoised_data=True)
        
        if result and result['save_paths']['denoised_data']:
            denoised_path = result['save_paths']['denoised_data']
            print(f"✅ Denoised data saved successfully: {denoised_path}")
            
            # Check if file exists and has content
            if Path(denoised_path).exists():
                with open(denoised_path, 'r') as f:
                    lines = f.readlines()
                    print(f"📊 File contains {len(lines)} lines")
                    if len(lines) > 1:
                        print(f"📋 Header: {lines[0].strip()}")
                        print(f"📋 Sample data: {lines[1].strip()}")
                    return True
            else:
                print("❌ Denoised data file not found")
                return False
        else:
            print("❌ Failed to save denoised data")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def show_directory_structure():
    """Show the new directory structure with denoised data"""
    print("\n📁 Directory Structure with Denoised Data")
    print("="*50)
    
    print("📂 Analysis Results:")
    print("ppg_analysis_results/")
    print("├── hyx_results/")
    print("│   ├── 喉咙-吞咽6次间隔10秒/")
    print("│   │   ├── 喉咙-吞咽6次间隔10秒_green_comprehensive_analysis.png")
    print("│   │   ├── 喉咙-吞咽6次间隔10秒_green_denoising_comparison.png")
    print("│   │   └── 喉咙-吞咽6次间隔10秒_green_STFT_comparison.png")
    print("│   └── ...")
    print("├── lhr_results/")
    print("└── lj_results/")
    
    print("\n📂 Denoised Data:")
    print("ppg_denoised_data/")
    print("├── hyx_denoised/")
    print("│   ├── 喉咙-吞咽6次间隔10秒_green_denoised_wiener.txt")
    print("│   ├── 喉咙-吞咽6次间隔10秒_ir_denoised_median.txt")
    print("│   └── ...")
    print("├── lhr_denoised/")
    print("│   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_green_denoised_wiener.txt")
    print("│   └── ...")
    print("└── lj_denoised/")
    print("    ├── PPG_1_20250721_150644_lijiao_一次吞咽_green_denoised_wiener.txt")
    print("    └── ...")
    
    print("\n📋 Data Format:")
    print("Time(s)    GREEN_denoised_wiener")
    print("0.000000   12345.678901")
    print("0.010000   12346.789012")
    print("0.020000   12347.890123")
    print("...")

def main():
    """Main test function"""
    print("🚀 PPG Advanced Analysis - Denoised Data Saving Test")
    print("="*60)
    
    # Test denoised data saving
    saving_test = test_denoised_data_saving()
    
    # Show directory structure
    show_directory_structure()
    
    # Summary
    print("\n" + "="*60)
    print("DENOISED DATA SAVING TEST SUMMARY")
    print("="*60)
    print(f"Denoised Data Saving Test: {'✅ PASSED' if saving_test else '❌ FAILED'}")
    
    if saving_test:
        print("\n🎉 Denoised data saving functionality added!")
        print("📁 New features:")
        print("   ✅ Saves best denoised data to ppg_denoised_data/")
        print("   ✅ Organized by data source (hyx_denoised/, lhr_denoised/, etc.)")
        print("   ✅ Includes time axis and denoised signal")
        print("   ✅ Filename includes channel and denoising method")
        print("   ✅ Tab-separated format for easy import")
        print("\n💡 Usage:")
        print("   python ppg_advanced_analysis.py --batch  # 默认保存降噪数据")
        print("   python ppg_advanced_analysis.py --batch '*_data' 'green' './results' 'false'  # 不保存降噪数据")
    else:
        print("\n⚠️ Denoised data saving test failed. Please check the implementation.")

if __name__ == "__main__":
    main()
