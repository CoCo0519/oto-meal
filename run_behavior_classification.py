#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_behavior_classification.py
统一入口（修复+优化版）
- 先解析 CLI/设置 CUDA_VISIBLE_DEVICES，再做后端优化与日志
- 自检：GPU/AMP 正常；未装 Triton 时自动跳过 torch.compile 测试
- DataLoader 参数更稳健（仅在 num_workers>0 时设置 prefetch_factor）
- 兼容老旧模块：可选注入 torch/optim/nn/F（行为模块现在已修复为显式 import，不再依赖注入）
"""

from __future__ import annotations

import os
import sys
import json
import glob
import shutil
import argparse
import platform
from datetime import datetime
import traceback
from pathlib import Path
import importlib
import warnings

# --------------------------- CLI 构建 ---------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="行为分类训练/推理统一入口（支持 *_data 多目录）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例：\n"
            "  %(prog)s --quick-start                   # 快速跑一遍（等价于特征提取）\n"
            "  %(prog)s --full-pipeline                 # 运行完整训练流水线\n"
            "  %(prog)s --data-only                     # 仅数据标注/准备\n"
            "  %(prog)s --feature-only                  # 仅特征提取\n"
            "  %(prog)s --models cnn transformer        # 指定模型类型\n"
            "  %(prog)s --config config_example.json    # 使用配置文件\n"
            "  %(prog)s --data-glob \"./*_data\"          # 使用所有 *_data 目录\n"
            "\n"
            "GPU优化示例：\n"
            "  %(prog)s --test-gpu                           # 测试GPU功能\n"
            "  %(prog)s --full-pipeline --mixed-precision    # 启用混合精度训练\n"
            "  %(prog)s --full-pipeline --batch-size 64       # 自定义批次大小\n"
            "  %(prog)s --full-pipeline --gpu-id 0            # 指定GPU设备\n"
            "  %(prog)s --full-pipeline --gpu-memory-fraction 0.8  # 限制GPU内存使用\n"
            "  %(prog)s --full-pipeline --no-gpu              # 强制使用CPU\n"
        )
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick-start', action='store_true', help='快速运行（默认走特征提取）')
    mode.add_argument('--full-pipeline', action='store_true', help='运行完整训练流水线')
    mode.add_argument('--data-only', action='store_true', help='仅数据标注/准备')
    mode.add_argument('--feature-only', action='store_true', help='仅特征提取')

    parser.add_argument('--data-dir', type=str, default='./hyx_data', help='数据目录路径 (默认: ./hyx_data)')
    parser.add_argument('--data-glob', type=str, default=None, help='数据目录通配符，例如 \"./*_data\"（将自动合并其下所有 .txt）')
    parser.add_argument('--merge-strategy', type=str, default='copy', choices=['copy', 'link'], help='合并数据时策略：copy/硬链接(link)。Windows 上推荐 copy。')
    parser.add_argument('--models', nargs='+', default=['fusion'], help='指定模型类型列表，例如: --models cnn transformer fusion')
    parser.add_argument('--config', type=str, default=None, help='配置 JSON 文件路径（可包含 data_glob/dataloader/gpu 等参数）')

    # GPU 相关
    parser.add_argument('--no-gpu', action='store_true', help='禁用GPU加速')
    parser.add_argument('--gpu-id', type=int, default=None, help='指定使用的GPU设备ID')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小（GPU模式下会自动优化）')
    parser.add_argument('--mixed-precision', action='store_true', help='启用混合精度训练（自动检测GPU支持）')
    parser.add_argument('--no-mixed-precision', action='store_true', help='禁用混合精度训练')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.9, help='GPU内存使用比例 (0.1-1.0, 默认0.9)')
    parser.add_argument('--test-gpu', action='store_true', help='仅测试GPU功能，不运行训练')

    # 开关：是否测试 torch.compile（Triton 缺失会失败）
    parser.add_argument('--skip-compile-test', action='store_true', help='跳过 torch.compile 测试（未装 Triton 时建议开启）')

    return parser

# --------------------------- GPU/环境工具 ---------------------------

def ensure_torch_cuda_build():
    """确保为 CUDA 构建；若检测到 +cpu 版本，给出修复建议"""
    try:
        import torch
        built_cuda = getattr(torch.version, 'cuda', None)
        print(f"  Torch: {getattr(torch, '__version__', 'unknown')}  built cuda: {built_cuda or '[cpu build]'}")
        if (not built_cuda) or ('cpu' in str(getattr(torch, '__version__', '')).lower()):
            print("  ⚠️ 检测到 CPU 构建的 PyTorch。若期望使用 GPU，请卸载并安装 CUDA 版：")
            print("     pip uninstall -y torch torchvision torchaudio")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            return False
        return True
    except Exception as e:
        print(f"  ⚠️ 无法检测 torch 构建类型：{e}")
        return False

def setup_gpu_performance(precision: str = "high", mem_fraction: float | None = None):
    """优化 PyTorch 后端以提升 GPU 利用率（若可用）"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(precision)
            torch.cuda.empty_cache()
            if mem_fraction is not None and hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                try:
                    torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
                except Exception as e:
                    print(f"  ⚠️ 设置每进程显存比例失败：{e}")
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                vis = os.environ['CUDA_VISIBLE_DEVICES']
                if vis:
                    try:
                        torch.cuda.set_device(int(vis.split(',')[0]))
                    except Exception:
                        pass
            print("✅ GPU 性能优化已启用（TF32/CuDNN/benchmarks）")
        else:
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 2)//2))
                print("⚠️ 使用 CPU 模式，已优化线程数")
            except Exception:
                pass
    except ImportError:
        print("⚠️ PyTorch 未安装，跳过 GPU 优化设置")
    except Exception as e:
        print(f"⚠️ GPU 性能优化设置失败：{e}")

def get_gpu_info():
    """获取详细的GPU信息"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            info = {'available': True, 'device_count': device_count, 'current_device': current_device, 'devices': []}
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                cc = float(f"{props.major}.{props.minor}")
                info['devices'].append({
                    'id': i,
                    'name': props.name,
                    'memory_total': props.total_memory / (1024**3),
                    'memory_allocated': torch.cuda.memory_allocated(i) / (1024**3),
                    'memory_reserved': torch.cuda.memory_reserved(i) / (1024**3),
                    'compute_capability': cc,
                    'multiprocessor_count': props.multi_processor_count
                })
            return info
        else:
            return {'available': False}
    except Exception as e:
        print(f"⚠️ 获取GPU信息失败：{e}")
        return {'available': False}

def optimize_data_loading_for_gpu():
    """为GPU训练优化数据加载参数"""
    cpu_count = os.cpu_count() or 2
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'num_workers': min(cpu_count, 8),
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 4,  # 仅当 num_workers>0 时有效
                'drop_last': True
            }
        else:
            return {
                'num_workers': max(1, cpu_count // 2),
                'pin_memory': False,
                'persistent_workers': False,
                'prefetch_factor': 2,
                'drop_last': False
            }
    except Exception:
        return {
            'num_workers': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'drop_last': False
        }

def log_gpu_environment():
    """打印与 GPU/CPU 环境相关的信息"""
    print("\n[环境信息]")
    try:
        import torch
        print(f"  Python: {platform.python_version()}  Torch: {getattr(torch, '__version__', 'unknown')}")
        ensure_torch_cuda_build()

        gpu_info = get_gpu_info()
        if gpu_info['available']:
            d = gpu_info['devices'][gpu_info['current_device']]
            cudnn_ver = getattr(torch.backends, 'cudnn', None)
            cudnn_ver = getattr(cudnn_ver, 'version', lambda: None)()
            print(f"  CUDA 可用 ✔  设备: {d['name']}")
            print(f"  Compute Capability: {d['compute_capability']}")
            print(f"  显存总量: {d['memory_total']:.1f} GiB  已分配: {d['memory_allocated']:.2f} GiB  已保留: {d['memory_reserved']:.2f} GiB")
            print(f"  cuDNN 版本: {cudnn_ver}")
            print(f"  cudnn.benchmark: {getattr(torch.backends.cudnn, 'benchmark', None)}")
            print(f"  allow_tf32: {getattr(torch.backends.cuda.matmul, 'allow_tf32', None)}")
            if hasattr(torch.cuda, 'amp'):
                print("  ✅ 支持 AMP (自动混合精度)")
        else:
            print("  CUDA 不可用，使用 CPU")
            print(f"  CPU 核心数: {os.cpu_count()}")
    except ImportError:
        print(f"  Python: {platform.python_version()}  Torch: 未安装")
        print("  ⚠️ PyTorch 未安装，将使用 CPU 模式")
        print(f"  CPU 核心数: {os.cpu_count()}")
    except Exception as e:
        print(f"⚠️ 环境信息打印失败：{e}")

def create_gpu_optimized_config(user_args=None):
    """创建GPU优化的训练配置（结合设备信息与 CLI 覆盖）"""
    gpu_info = get_gpu_info()
    base = {
        'use_gpu': gpu_info['available'],
        'mixed_precision': False,
        'gradient_accumulation_steps': 1,
        'gradient_clip_norm': 1.0,
        'dataloader_params': optimize_data_loading_for_gpu(),
        'device_id': 0,
        'use_amp': False,
        'compile_model': False
    }

    if gpu_info['available']:
        d = gpu_info['devices'][gpu_info['current_device']]
        vram = d['memory_total']
        cc = d['compute_capability']

        if vram >= 16:
            base['batch_size'] = 64; base['mixed_precision'] = True; base['use_amp'] = True
        elif vram >= 8:
            base['batch_size'] = 32; base['mixed_precision'] = True; base['use_amp'] = True
        elif vram >= 4:
            base['batch_size'] = 16; base['mixed_precision'] = True; base['use_amp'] = True
        else:
            base['batch_size'] = 8;  base['mixed_precision'] = False; base['use_amp'] = False

        base['use_tensor_cores'] = cc >= 7.0
        base['compile_model'] = cc >= 7.0

        if gpu_info['device_count'] > 1:
            base['multi_gpu'] = True
            base['device_count'] = gpu_info['device_count']
            base['batch_size'] *= gpu_info['device_count']
        else:
            base['multi_gpu'] = False
            base['device_count'] = 1
    else:
        base.update({
            'batch_size': 8,
            'mixed_precision': False,
            'use_tensor_cores': False,
            'multi_gpu': False,
            'device_count': 1,
            'use_amp': False,
            'compile_model': False
        })

    if user_args is not None:
        if user_args.batch_size is not None:
            base['batch_size'] = user_args.batch_size
        if user_args.mixed_precision:
            base['mixed_precision'] = True; base['use_amp'] = True
        if user_args.no_mixed_precision:
            base['mixed_precision'] = False; base['use_amp'] = False
        if user_args.gpu_id is not None:
            base['device_id'] = user_args.gpu_id

    dl = base['dataloader_params']
    if not dl or dl.get('num_workers', 0) <= 0:
        dl.pop('prefetch_factor', None)

    return base

def monitor_gpu_memory():
    """监控GPU内存使用情况"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU内存: {allocated:.2f}GB / {reserved:.2f}GB / {total:.2f}GB (已用/保留/总量)")
            if allocated / total > 0.9:
                print("  ⚠️ GPU内存使用率过高，建议清理缓存")
                import torch as _t; _t.cuda.empty_cache()
                return True
        return False
    except Exception:
        return False

def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False

def test_gpu_functionality(skip_compile_test: bool = False):
    """测试GPU功能是否正常工作"""
    print("\n🧪 测试GPU功能...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法进行GPU测试")
            return False

        device = torch.device('cuda')
        print(f"✅ 使用GPU设备: {torch.cuda.get_device_name()}")

        x = torch.randn(1024, 1024, device=device)
        y = torch.randn(1024, 1024, device=device)
        _ = torch.matmul(x, y)
        print("✅ GPU张量运算正常")

        try:
            if hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    x = torch.randn(512, 512, device=device)
                    y = torch.randn(512, 512, device=device)
                    _ = torch.matmul(x, y)
                print("✅ 混合精度 (AMP) 正常")
        except Exception as e:
            print(f"⚠️ 混合精度测试失败: {e}")

        try:
            if not skip_compile_test and hasattr(torch, 'compile'):
                if _has_triton():
                    mdl = torch.nn.Linear(256, 128).to(device)
                    cmpl = torch.compile(mdl)
                    _ = cmpl(torch.randn(4, 256, device=device))
                    print("✅ 模型编译 (torch.compile) 正常")
                else:
                    print("ℹ️ 未检测到 Triton，已跳过 torch.compile 测试（如需启用请安装 triton）")
            elif skip_compile_test:
                print("ℹ️ 按参数要求跳过 torch.compile 测试")
            else:
                print("ℹ️ 当前 PyTorch 不支持 torch.compile")
        except Exception as e:
            print(f"⚠️ 模型编译测试失败: {e}")

        import torch as _t; _t.cuda.empty_cache()
        print("✅ GPU功能测试完成")
        return True
    except ImportError:
        print("❌ PyTorch 未安装，无法进行GPU测试")
        return False
    except Exception as e:
        print(f"❌ GPU功能测试失败: {e}")
        return False

def setup_plot_fonts():
    """为 matplotlib 配置中文字体与负号显示（若 matplotlib 可用）。"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt  # noqa: F401
        for name in ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]:
            try:
                matplotlib.rcParams['font.sans-serif'] = [name]
                break
            except Exception:
                continue
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

# --------------------------- 数据工具 ---------------------------

def check_data_directory(data_dir: str) -> bool:
    p = Path(data_dir)
    if not p.exists() or not p.is_dir():
        print(f"❌ 数据目录不存在：{p.resolve()}")
        return False
    txts = list(p.rglob("*.txt"))
    if not txts:
        print(f"❌ 数据目录中未找到 .txt 数据文件：{p.resolve()}")
        return False
    print(f"✅ 数据目录检查通过：{p.resolve()}（{len(txts)} 个 .txt）")
    return True

def gather_data_to_merged_dir(data_glob: str, merge_strategy: str = "copy") -> str:
    merged = Path("./_merged_data")
    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(parents=True, exist_ok=True)

    folders = [Path(p) for p in glob.glob(data_glob) if Path(p).is_dir()]
    if not folders:
        print(f"❌ 未找到匹配的数据文件夹：{data_glob}")
        return str(merged)

    count = 0
    for folder in folders:
        for txt in folder.rglob("*.txt"):
            dst = merged / f"{folder.name}__{txt.name}"
            if merge_strategy == "link" and hasattr(os, 'link'):
                try:
                    os.link(txt, dst)
                except Exception:
                    shutil.copy2(txt, dst)
            else:
                shutil.copy2(txt, dst)
            count += 1

    print(f"✅ 已收集 {count} 个 .txt 文件到 {merged.resolve()}")
    return str(merged)

# --------------------------- （可选）行为模块注入 ---------------------------

def patch_behavior_module_symbols(mod):
    """老代码兜底：把 optim/nn/F/torch 注入到模块命名空间（新版本已显式 import，不再依赖）"""
    try:
        import torch
        import torch.optim as optim
        import torch.nn as nn
        import torch.nn.functional as F
        for name, obj in [('optim', optim), ('nn', nn), ('F', F), ('torch', torch)]:
            if not hasattr(mod, name):
                setattr(mod, name, obj)
    except Exception as e:
        print(f"⚠️ 注入 torch 符号失败：{e}")

# --------------------------- 子流程 ---------------------------

def run_data_labeling_only(data_dir: str) -> bool:
    print("\n" + "="*50)
    print("仅运行数据标注 / 基础处理")
    print("="*50)

    if not check_data_directory(data_dir):
        return False

    try:
        mod = importlib.import_module('behavior_classification_system')
        patch_behavior_module_symbols(mod)
        classification_main = getattr(mod, 'main')

        original_argv = sys.argv.copy()
        try:
            sys.argv = ['behavior_classification_system.py', '--data-dir', data_dir, '--feature-only']
            classification_main()
        except SystemExit:
            try:
                sys.argv = ['behavior_classification_system.py', '--data_dir', data_dir, '--feature-only']
                classification_main()
            except SystemExit:
                sys.argv = ['behavior_classification_system.py', '--data-dir', data_dir]
                classification_main()
        finally:
            sys.argv = original_argv
        print("✅ 数据标注/基础处理完成")
        return True
    except Exception as e:
        print(f"❌ 数据标注/基础处理失败: {e}")
        traceback.print_exc()
        return False

def run_feature_extraction_only(data_dir: str) -> bool:
    print("\n" + "="*50)
    print("仅运行特征提取")
    print("="*50)

    if not check_data_directory(data_dir):
        return False

    try:
        mod = importlib.import_module('behavior_classification_system')
        patch_behavior_module_symbols(mod)
        classification_main = getattr(mod, 'main')

        original_argv = sys.argv.copy()
        try:
            sys.argv = ['behavior_classification_system.py', '--data-dir', data_dir, '--data-only']
            classification_main()
        except SystemExit:
            try:
                sys.argv = ['behavior_classification_system.py', '--data_dir', data_dir, '--data-only']
                classification_main()
            except SystemExit:
                sys.argv = ['behavior_classification_system.py', '--data-dir', data_dir]
                classification_main()
        finally:
            sys.argv = original_argv
        print("✅ 特征提取完成")
        return True
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        traceback.print_exc()
        return False

def run_simple_cpu_training(data_dir: str, models: list[str], config: dict, args=None) -> bool:
    print("\n" + "="*50)
    print("运行简化CPU训练模式")
    print("="*50)

    if not check_data_directory(data_dir):
        return False

    try:
        print("📊 开始数据预处理...")
        data_files = list(Path(data_dir).rglob("*.txt"))
        print(f"找到 {len(data_files)} 个数据文件")

        total_samples = 0
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_samples += len(lines)
                    print(f"  {file_path.name}: {len(lines)} 行数据")
            except Exception as e:
                print(f"  ⚠️ 读取 {file_path.name} 失败: {e}")

        print(f"📈 数据统计:")
        print(f"  - 总文件数: {len(data_files)}")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 平均每文件: {total_samples/len(data_files):.1f} 样本")

        report_path = f"simple_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("简化CPU训练分析报告\n")
            f.write("="*50 + "\n")
            f.write(f"数据目录: {data_dir}\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总文件数: {len(data_files)}\n")
            f.write(f"总样本数: {total_samples}\n")
            f.write(f"平均每文件: {total_samples/len(data_files):.1f} 样本\n\n")
            f.write("文件列表:\n")
            for file_path in data_files:
                f.write(f"  - {file_path.name}\n")

        print(f"📄 分析报告已保存: {report_path}")
        print("✅ 简化CPU训练完成")
        print("\n💡 提示: 要使用完整的GPU训练功能，请安装 PyTorch CUDA 构建：")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return True

    except Exception as e:
        print(f"❌ 简化CPU训练失败: {e}")
        traceback.print_exc()
        return False

def run_full_pipeline(data_dir: str, models: list[str], config: dict, args=None) -> bool:
    print("\n" + "="*50)
    print("运行完整训练流水线")
    print("="*50)

    if not check_data_directory(data_dir):
        return False

    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
    except ImportError:
        print("⚠️ PyTorch 未安装，跳过GPU优化")
    except Exception:
        pass

    try:
        from complete_training_pipeline import TrainingPipeline  # 按项目结构调整
    except ImportError:
        print("⚠️ complete_training_pipeline 模块未找到")
        print("尝试使用简化的CPU模式...")
        return run_simple_cpu_training(data_dir, models, config, args)

    gpu_cfg = create_gpu_optimized_config(args)

    tmp = {
        'data_dir': data_dir,
        'model_types': models,
        **gpu_cfg,
        **{k: v for k, v in config.items() if k not in ('data_dir', 'model_types')}
    }
    if 'dataloader_params' in config:
        tmp['dataloader_params'].update(config['dataloader_params'])

    print(f"🚀 GPU优化配置:")
    print(f"   - 使用GPU: {tmp['use_gpu']}")
    print(f"   - 混合精度: {tmp['mixed_precision']}  (AMP: {tmp.get('use_amp', False)})")
    print(f"   - 批次大小: {tmp['batch_size']}")
    print(f"   - 设备ID: {tmp.get('device_id', 0)}")
    print(f"   - 数据加载器workers: {tmp['dataloader_params']['num_workers']}")
    pf = tmp['dataloader_params'].get('prefetch_factor', None)
    print(f"   - 预取因子: {pf if pf is not None else '(未设置或无效)'}")
    print(f"   - 模型编译优化: {tmp.get('compile_model', False)}")
    if tmp.get('multi_gpu'):
        print(f"   - 多GPU训练: {tmp['device_count']} 个GPU")

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = f"./_auto_config_{ts}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(tmp, f, ensure_ascii=False, indent=2)

        pipeline = TrainingPipeline(config_path)

        if tmp['use_gpu']:
            print("\n📊 训练前GPU状态:")
            monitor_gpu_memory()

        results = pipeline.run_complete_pipeline()

        if tmp['use_gpu']:
            print("\n🧹 训练后GPU清理:")
            monitor_gpu_memory()

        if os.path.exists(config_path):
            os.remove(config_path)

        print("✅ 完整训练流水线执行完成")
        print(f"   结果保存在: {getattr(pipeline, 'output_dir', '[pipeline.output_dir 不可用]')}")
        return True

    except Exception as e:
        print(f"❌ 训练流水线失败: {e}")
        traceback.print_exc()
        print("============================================================")
        print("❌ 执行失败，请查看错误信息")
        return False

# --------------------------- 主入口 ---------------------------

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = build_arg_parser()
    args = parser.parse_args()

    # 先处理 GPU 开关与可见设备，再进行后端优化
    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("🚫 已禁用GPU加速")
    elif args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"🎯 使用GPU设备: {args.gpu_id}")

    setup_gpu_performance(precision="high", mem_fraction=None if args.no_gpu else args.gpu_memory_fraction)
    log_gpu_environment()

    # 自检（可选跳过 compile 测试）
    gpu_ok = True
    if not args.no_gpu:
        gpu_ok = test_gpu_functionality(skip_compile_test=args.skip_compile_test)
        if not gpu_ok:
            print("⚠️ GPU不可用，将使用CPU模式")
            args.no_gpu = True

    setup_plot_fonts()

    # 加载配置文件（若提供）
    config: dict = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"⚠️ 读取配置文件失败（将继续使用命令行参数）：{e}")

    # 数据合并：优先 data_glob
    actual_data_dir = args.data_dir
    data_glob = args.data_glob or config.get("data_glob")
    merge_strategy = args.merge_strategy or config.get("merge_strategy", "copy")
    if data_glob:
        print(f"🔎 通过通配符收集数据: {data_glob}")
        actual_data_dir = gather_data_to_merged_dir(data_glob, merge_strategy)

    # 根据模式执行
    ok = False
    if args.test_gpu:
        print("\n🧪 仅进行GPU功能测试...")
        ok = test_gpu_functionality(skip_compile_test=args.skip_compile_test)
    elif args.quick_start or args.feature_only:
        ok = run_feature_extraction_only(actual_data_dir)
    elif args.data_only:
        ok = run_data_labeling_only(actual_data_dir)
    elif args.full_pipeline:
        ok = run_full_pipeline(actual_data_dir, args.models, config, args)
    else:
        print("\n⚠️ 未选择具体模式，建议使用 --quick-start 或 --full-pipeline。")
        parser.print_help()
        return

    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
