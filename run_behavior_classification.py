#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行脚本：基于耳道PPG&IMU的行为分类系统
提供简化的命令行接口，方便快速开始训练和评估

使用示例：
    python run_behavior_classification.py --quick-start
    python run_behavior_classification.py --full-pipeline --models fusion
    python run_behavior_classification.py --data-only
"""

import os
import sys
import argparse
import json
from pathlib import Path

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'torch', 'numpy', 'scipy', 'matplotlib', 'seaborn', 
        'sklearn', 'pandas', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        # 将sklearn显示名称转换回scikit-learn
        display_packages = []
        for pkg in missing_packages:
            if pkg == 'sklearn':
                display_packages.append('scikit-learn')
            else:
                display_packages.append(pkg)
        
        for pkg in display_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(display_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_data_directory(data_dir):
    """检查数据目录"""
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    # 检查必要的数据文件
    required_patterns = ['耳道', '喉咙']
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"❌ 数据目录中未找到.txt文件: {data_dir}")
        return False
    
    ear_files = [f for f in txt_files if '耳道' in f]
    throat_files = [f for f in txt_files if '喉咙' in f]
    
    if not ear_files:
        print("❌ 未找到耳道数据文件（文件名应包含'耳道'）")
        return False
    
    if not throat_files:
        print("❌ 未找到喉咙数据文件（文件名应包含'喉咙'）")
        return False
    
    print(f"✅ 数据目录检查通过: {data_dir}")
    print(f"   - 耳道文件: {len(ear_files)}个")
    print(f"   - 喉咙文件: {len(throat_files)}个")
    
    return True

def run_data_labeling_only(data_dir):
    """仅运行数据标注"""
    print("\n" + "="*50)
    print("运行数据标注模块")
    print("="*50)
    
    try:
        from data_labeling_system import main as labeling_main
        
        # 临时修改sys.argv以传递参数
        original_argv = sys.argv.copy()
        sys.argv = ['data_labeling_system.py']
        
        labeling_main()
        
        # 恢复原始argv
        sys.argv = original_argv
        
        print("✅ 数据标注完成")
        
    except Exception as e:
        print(f"❌ 数据标注失败: {e}")
        return False
    
    return True

def run_feature_extraction_only(data_dir):
    """仅运行特征提取和基础分类"""
    print("\n" + "="*50)
    print("运行特征提取和基础分类")
    print("="*50)
    
    try:
        from behavior_classification_system import main as classification_main
        
        # 临时修改sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ['behavior_classification_system.py']
        
        classification_main()
        
        # 恢复原始argv
        sys.argv = original_argv
        
        print("✅ 特征提取和基础分类完成")
        
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return False
    
    return True

def run_full_pipeline(data_dir, models, config):
    """运行完整训练流水线"""
    print("\n" + "="*50)
    print("运行完整训练流水线")
    print("="*50)
    
    try:
        from complete_training_pipeline import TrainingPipeline
        
        # 创建临时配置文件
        temp_config = {
            'data_dir': data_dir,
            'model_types': models,
            **config
        }
        
        config_path = 'temp_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(temp_config, f, ensure_ascii=False, indent=2)
        
        # 初始化并运行流水线
        pipeline = TrainingPipeline(config_path)
        results = pipeline.run_complete_pipeline()
        
        # 清理临时文件
        if os.path.exists(config_path):
            os.remove(config_path)
        
        print("✅ 完整训练流水线执行完成")
        print(f"   结果保存在: {pipeline.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练流水线失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_quick_start_config():
    """创建快速开始配置"""
    return {
        'batch_size': 16,  # 较小的批次以适应更多设备
        'learning_rate': 1e-4,
        'num_epochs': 50,  # 较少的轮数用于快速验证
        'patience': 10,
        'test_size': 0.2,
        'use_class_weights': True,
        'augmentation': False  # 关闭数据增强以加快速度
    }

def main():
    parser = argparse.ArgumentParser(
        description='基于耳道PPG&IMU的行为分类系统 - 一键运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --quick-start                    # 快速开始（推荐新手）
  %(prog)s --full-pipeline                  # 运行完整流水线
  %(prog)s --data-only                      # 仅数据标注
  %(prog)s --feature-only                   # 仅特征提取
  %(prog)s --models cnn transformer         # 指定模型类型
  %(prog)s --config config_example.json     # 使用配置文件
        """
    )
    
    # 基本参数
    parser.add_argument('--data-dir', type=str, default='./hyx_data',
                       help='数据目录路径 (默认: ./hyx_data)')
    
    # 运行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--quick-start', action='store_true',
                           help='快速开始模式（推荐新手使用）')
    mode_group.add_argument('--full-pipeline', action='store_true',
                           help='运行完整训练流水线')
    mode_group.add_argument('--data-only', action='store_true',
                           help='仅运行数据标注')
    mode_group.add_argument('--feature-only', action='store_true',
                           help='仅运行特征提取和基础分类')
    
    # 模型选择
    parser.add_argument('--models', nargs='+', 
                       choices=['cnn', 'transformer', 'fusion'],
                       default=['fusion'],
                       help='要训练的模型类型 (默认: fusion)')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    
    # 其他选项
    parser.add_argument('--no-gpu', action='store_true',
                       help='禁用GPU加速')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("🎯 基于耳道PPG&IMU的行为分类系统")
    print("   目标：静息/咀嚼/咳嗽/吞咽 四分类")
    print("   技术：深度学习 + 多模态特征融合")
    print()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 检查数据目录
    if not check_data_directory(args.data_dir):
        print("\n💡 提示:")
        print("   1. 确保数据目录存在且包含.txt文件")
        print("   2. 文件名应包含'耳道'和'喉咙'关键词")
        print("   3. 数据格式：首行为中文表头，6列数值数据")
        return 1
    
    # 设置GPU
    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("🚫 已禁用GPU加速")
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"📋 使用配置文件: {args.config}")
    else:
        config = create_quick_start_config()
        if args.quick_start:
            print("🚀 使用快速开始配置")
    
    # 覆盖配置参数
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    
    # 运行对应模式
    success = False
    
    if args.data_only:
        success = run_data_labeling_only(args.data_dir)
        
    elif args.feature_only:
        success = run_feature_extraction_only(args.data_dir)
        
    elif args.full_pipeline or args.quick_start:
        if args.quick_start:
            print("🎯 快速开始模式：将训练融合模型（推荐）")
            models = ['fusion']
        else:
            models = args.models
            
        success = run_full_pipeline(args.data_dir, models, config)
    
    # 总结
    print("\n" + "="*60)
    if success:
        print("🎉 执行完成！")
        if args.full_pipeline or args.quick_start:
            print("📊 查看结果：")
            print("   - 训练日志：training.log")
            print("   - 可视化结果：comprehensive_results.png") 
            print("   - 模型文件：best_*_model.pth")
            print("   - 详细报告：classification_report.txt")
    else:
        print("❌ 执行失败，请查看错误信息")
        return 1
    
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
