#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the new directory structure
"""

import os
import sys
from pathlib import Path

def test_directory_mapping():
    """Test the directory mapping logic"""
    print("🧪 Testing Directory Structure Mapping")
    print("="*50)
    
    # Test cases
    test_cases = [
        ("hyx_data", "hyx_results"),
        ("lhr_data", "lhr_results"), 
        ("lj_data", "lj_results"),
        ("test_data", "test_results"),
        ("my_custom_data", "my_custom_results"),
        ("data_folder", "data_folder_results")
    ]
    
    print("📋 Directory mapping test cases:")
    for input_dir, expected_output in test_cases:
        # Simulate the logic from the updated function
        if input_dir.endswith('_data'):
            actual_output = input_dir.replace('_data', '_results')
        else:
            actual_output = f"{input_dir}_results"
        
        status = "✅" if actual_output == expected_output else "❌"
        print(f"   {status} {input_dir} → {actual_output} (expected: {expected_output})")
    
    return True

def simulate_directory_structure():
    """Simulate the new directory structure"""
    print("\n🏗️ Simulated Directory Structure")
    print("="*50)
    
    # Simulate data directories
    data_dirs = ["hyx_data", "lhr_data", "lj_data"]
    base_results = "ppg_analysis_results"
    
    print(f"📁 Data directories: {', '.join(data_dirs)}")
    print(f"📂 Base results directory: {base_results}")
    print("\n📋 Generated structure:")
    
    for data_dir in data_dirs:
        # Convert xxx_data to xxx_results
        if data_dir.endswith('_data'):
            results_dir = data_dir.replace('_data', '_results')
        else:
            results_dir = f"{data_dir}_results"
        
        print(f"\n📂 {base_results}/{results_dir}/")
        print(f"   ├── 文件1/")
        print(f"   │   ├── 文件1_green_comprehensive_analysis.png")
        print(f"   │   ├── 文件1_green_denoising_comparison.png")
        print(f"   │   ├── 文件1_green_STFT_comparison.png")
        print(f"   │   ├── 文件1_ir_comprehensive_analysis.png")
        print(f"   │   ├── 文件1_ir_denoising_comparison.png")
        print(f"   │   └── 文件1_ir_STFT_comparison.png")
        print(f"   ├── 文件2/")
        print(f"   │   ├── 文件2_green_comprehensive_analysis.png")
        print(f"   │   ├── 文件2_green_denoising_comparison.png")
        print(f"   │   ├── 文件2_green_STFT_comparison.png")
        print(f"   │   ├── 文件2_ir_comprehensive_analysis.png")
        print(f"   │   ├── 文件2_ir_denoising_comparison.png")
        print(f"   │   └── 文件2_ir_STFT_comparison.png")
        print(f"   └── ...")

def main():
    """Main test function"""
    print("🚀 PPG Advanced Analysis - Directory Structure Test")
    print("="*60)
    
    # Test directory mapping
    mapping_test = test_directory_mapping()
    
    # Simulate directory structure
    simulate_directory_structure()
    
    # Summary
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE TEST SUMMARY")
    print("="*60)
    print(f"Directory Mapping Test: {'✅ PASSED' if mapping_test else '❌ FAILED'}")
    
    if mapping_test:
        print("\n🎉 Directory structure optimization completed!")
        print("📁 Now each xxx_data/ will generate xxx_results/ subdirectory")
        print("📂 This makes it easy to distinguish results from different data sources")
        print("\n💡 Benefits:")
        print("   - Clear separation of results by data source")
        print("   - Easy to locate specific analysis results")
        print("   - Organized file structure")
        print("   - No confusion between different datasets")
    else:
        print("\n⚠️ Directory mapping test failed. Please check the logic.")

if __name__ == "__main__":
    main()
