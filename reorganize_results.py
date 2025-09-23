#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reorganize existing results into the new directory structure
"""

import os
import shutil
from pathlib import Path

def reorganize_existing_results():
    """Reorganize existing results into the new structure"""
    print("🔄 Reorganizing Existing Results")
    print("="*50)
    
    base_results = Path("ppg_analysis_results")
    if not base_results.exists():
        print("❌ No existing results directory found")
        return
    
    # Define data source mappings
    data_mappings = {
        "hyx_data": "hyx_results",
        "lhr_data": "lhr_results", 
        "lj_data": "lj_results"
    }
    
    # Files that belong to each data source
    hyx_files = [
        "喉咙- 咳嗽共6次间隔10秒",
        "喉咙-吞咽6次间隔10秒", 
        "喉咙-咀嚼5下共6次间隔10秒",
        "喉咙-喝水共4次间隔10秒",
        "喉咙-说话共6次持续5秒间隔10秒",
        "耳道- 咳嗽共6次间隔10秒",
        "耳道-吞咽6次间隔10秒",
        "耳道-咀嚼5下共6次间隔10秒", 
        "耳道-喝水共4次间隔10秒",
        "耳道-说话共6次持续5秒间隔10秒"
    ]
    
    lhr_files = [
        "PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽",
        "PPG_1_20250721_161413_双耳佩戴_饮水三次吞咽",
        "PPG_1_20250721_161608_不同位置_饮水三次吞咽",
        "PPG_1_20250721_161845_不同位置_饮水三次吞咽",
        "PPG_1_20250721_162030_不同位置_唾液四次吞咽",
        "PPG_2_20250721_152548",
        "PPG_2_20250721_161413",
        "PPG_2_20250721_161608",
        "PPG_2_20250721_161845",
        "PPG_2_20250721_162030"
    ]
    
    lj_files = [
        "PPG_1_20250721_150644_lijiao_一次吞咽",
        "PPG_2_20250721_150644_lijiao_一次吞咽"
    ]
    
    file_mappings = {
        "hyx_results": hyx_files,
        "lhr_results": lhr_files,
        "lj_results": lj_files
    }
    
    # Create new directory structure
    for results_dir, files in file_mappings.items():
        target_dir = base_results / results_dir
        target_dir.mkdir(exist_ok=True)
        print(f"📂 Created: {target_dir}")
        
        # Move files to appropriate directories
        for file_pattern in files:
            source_path = base_results / file_pattern
            if source_path.exists():
                target_path = target_dir / file_pattern
                if source_path.is_dir():
                    shutil.move(str(source_path), str(target_path))
                    print(f"   📁 Moved: {file_pattern}/")
                else:
                    # Handle individual files
                    target_path.parent.mkdir(exist_ok=True)
                    shutil.move(str(source_path), str(target_path))
                    print(f"   📄 Moved: {file_pattern}")
    
    # Clean up any remaining files
    remaining_files = [f for f in base_results.iterdir() if f.is_dir() and f.name not in ["hyx_results", "lhr_results", "lj_results"]]
    for remaining_file in remaining_files:
        if remaining_file.name not in ["hyx_results", "lhr_results", "lj_results"]:
            print(f"⚠️ Remaining file: {remaining_file.name}")
    
    print("\n✅ Reorganization completed!")

def show_new_structure():
    """Show the new directory structure"""
    print("\n📁 New Directory Structure")
    print("="*50)
    
    base_results = Path("ppg_analysis_results")
    if not base_results.exists():
        print("❌ No results directory found")
        return
    
    for results_dir in ["hyx_results", "lhr_results", "lj_results"]:
        target_dir = base_results / results_dir
        if target_dir.exists():
            file_count = len(list(target_dir.rglob("*.png")))
            print(f"\n📂 {results_dir}/ ({file_count} images)")
            
            # Show first few files as examples
            subdirs = [d for d in target_dir.iterdir() if d.is_dir()][:3]
            for subdir in subdirs:
                png_files = list(subdir.glob("*.png"))
                print(f"   ├── {subdir.name}/ ({len(png_files)} images)")
                for png_file in png_files[:2]:  # Show first 2 images
                    print(f"   │   ├── {png_file.name}")
                if len(png_files) > 2:
                    print(f"   │   └── ... ({len(png_files)-2} more)")
            
            if len(list(target_dir.iterdir())) > 3:
                print(f"   └── ... ({len(list(target_dir.iterdir()))-3} more files)")

def main():
    """Main function"""
    print("🚀 PPG Results Reorganization")
    print("="*60)
    
    # Reorganize existing results
    reorganize_existing_results()
    
    # Show new structure
    show_new_structure()
    
    print("\n" + "="*60)
    print("REORGANIZATION SUMMARY")
    print("="*60)
    print("✅ Results have been reorganized into:")
    print("   📂 hyx_results/ - Results from hyx_data/")
    print("   📂 lhr_results/ - Results from lhr_data/") 
    print("   📂 lj_results/ - Results from lj_data/")
    print("\n💡 Benefits:")
    print("   - Clear separation by data source")
    print("   - Easy to find specific results")
    print("   - Organized file structure")
    print("   - Future runs will use this structure automatically")

if __name__ == "__main__":
    main()
