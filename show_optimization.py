#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Show the optimization results
"""

from pathlib import Path

def show_optimization():
    """Show the directory structure optimization"""
    print("🎯 PPG Advanced Analysis - Directory Structure Optimization")
    print("="*70)
    
    print("\n📋 BEFORE (Old Structure):")
    print("ppg_analysis_results/")
    print("├── 喉咙-吞咽6次间隔10秒/")
    print("│   ├── 喉咙-吞咽6次间隔10秒_green_comprehensive_analysis.png")
    print("│   ├── 喉咙-吞咽6次间隔10秒_green_denoising_comparison.png")
    print("│   └── 喉咙-吞咽6次间隔10秒_green_STFT_comparison.png")
    print("├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽/")
    print("│   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_green_comprehensive_analysis.png")
    print("│   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_green_denoising_comparison.png")
    print("│   └── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_green_STFT_comparison.png")
    print("└── ... (all files mixed together)")
    
    print("\n📋 AFTER (New Structure):")
    print("ppg_analysis_results/")
    print("├── hyx_results/")
    print("│   ├── 喉咙-吞咽6次间隔10秒/")
    print("│   │   ├── 喉咙-吞咽6次间隔10秒_green_comprehensive_analysis.png")
    print("│   │   ├── 喉咙-吞咽6次间隔10秒_green_denoising_comparison.png")
    print("│   │   ├── 喉咙-吞咽6次间隔10秒_green_STFT_comparison.png")
    print("│   │   ├── 喉咙-吞咽6次间隔10秒_ir_comprehensive_analysis.png")
    print("│   │   ├── 喉咙-吞咽6次间隔10秒_ir_denoising_comparison.png")
    print("│   │   └── 喉咙-吞咽6次间隔10秒_ir_STFT_comparison.png")
    print("│   ├── 喉咙-咀嚼5下共6次间隔10秒/")
    print("│   └── ...")
    print("├── lhr_results/")
    print("│   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽/")
    print("│   │   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_green_comprehensive_analysis.png")
    print("│   │   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_green_denoising_comparison.png")
    print("│   │   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_green_STFT_comparison.png")
    print("│   │   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_ir_comprehensive_analysis.png")
    print("│   │   ├── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_ir_denoising_comparison.png")
    print("│   │   └── PPG_1_20250721_152548_双耳佩戴_饮水3次吞咽_ir_STFT_comparison.png")
    print("│   └── ...")
    print("└── lj_results/")
    print("    ├── PPG_1_20250721_150644_lijiao_一次吞咽/")
    print("    └── ...")
    
    print("\n🎯 Key Improvements:")
    print("✅ Clear separation by data source (hyx_data → hyx_results)")
    print("✅ Easy to distinguish different datasets")
    print("✅ Organized file structure")
    print("✅ Support for multiple channels (green, ir, red)")
    print("✅ Chinese font support")
    print("✅ Automatic directory creation")
    
    print("\n📋 Usage Examples:")
    print("# Process all data directories with new structure")
    print("python ppg_advanced_analysis.py")
    print("")
    print("# Process specific data directory")
    print("python ppg_advanced_analysis.py --batch 'hyx_data' 'green,ir,red'")
    print("")
    print("# Process with custom output directory")
    print("python ppg_advanced_analysis.py --batch '*_data' 'green' './my_analysis_results'")
    
    print("\n📊 Directory Mapping:")
    print("hyx_data/     → hyx_results/")
    print("lhr_data/     → lhr_results/")
    print("lj_data/      → lj_results/")
    print("custom_data/  → custom_results/")
    
    print("\n🎉 Optimization Complete!")
    print("="*70)

if __name__ == "__main__":
    show_optimization()
