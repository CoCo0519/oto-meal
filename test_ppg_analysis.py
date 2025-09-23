#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for PPG advanced analysis
"""

import os
import sys
from pathlib import Path

def test_ppg_analysis():
    """Test the PPG analysis script"""
    print("🧪 Testing PPG Advanced Analysis Script")
    print("="*50)
    
    # Check if the script exists
    script_path = Path("ppg_advanced_analysis.py")
    if not script_path.exists():
        print("❌ ppg_advanced_analysis.py not found")
        return False
    
    print("✅ Script file found")
    
    # Check for data directories
    data_dirs = []
    for pattern in ["*_data", "hyx_data", "lhr_data", "lj_data"]:
        dirs = list(Path(".").glob(pattern))
        data_dirs.extend(dirs)
    
    if not data_dirs:
        print("❌ No data directories found")
        return False
    
    print(f"✅ Found {len(data_dirs)} data directories:")
    for dir_path in data_dirs:
        txt_files = list(dir_path.glob("*.txt"))
        print(f"   - {dir_path}: {len(txt_files)} .txt files")
    
    # Test single file processing
    test_file = None
    for dir_path in data_dirs:
        txt_files = list(dir_path.glob("*.txt"))
        if txt_files:
            test_file = txt_files[0]
            break
    
    if test_file:
        print(f"\n🔍 Testing single file processing: {test_file.name}")
        try:
            # Import and test the function
            import ppg_advanced_analysis
            result = ppg_advanced_analysis.advanced_ppg_analysis(str(test_file), 'green', './test_output')
            if result:
                print("✅ Single file processing test passed")
            else:
                print("❌ Single file processing test failed")
        except Exception as e:
            print(f"❌ Error during single file test: {e}")
    else:
        print("❌ No test files found")
    
    print("\n📋 Usage Examples:")
    print("  python ppg_advanced_analysis.py")
    print("  python ppg_advanced_analysis.py --batch")
    print("  python ppg_advanced_analysis.py --batch '*_data' 'green,ir,red'")
    print("  python ppg_advanced_analysis.py single_file.txt green")
    
    return True

if __name__ == "__main__":
    test_ppg_analysis()

