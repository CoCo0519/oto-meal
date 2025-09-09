# -*- coding: utf-8 -*-
"""
数据标注系统：基于喉咙PPG&IMU信号标注耳道数据中的行为时间段
通过分析喉咙信号的特征变化，自动识别并标注对应时间段的行为类型

核心思路：
1. 喉咙信号作为"真值标注器" - 行为发生时喉咙信号有明显特征变化
2. 时间对齐 - 确保喉咙和耳道数据在时间上同步
3. 行为检测 - 基于喉咙信号的能量/频谱变化检测行为发生时刻
4. 标签传播 - 将检测到的行为时间段标签应用到对应的耳道数据

作者：基于Project-Swallow项目扩展
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from datetime import datetime, timedelta

# 导入现有预处理模块
from anc_template_pipeline_100Hz import (
    notch_filter, butter_highpass, butter_bandpass, short_time_energy
)

class BehaviorDetector:
    """行为检测器：基于喉咙信号检测行为发生时间"""
    
    def __init__(self, fs=100):
        self.fs = fs
        self.window_size = 1.0  # 1秒窗口用于特征计算
        self.overlap = 0.5  # 50%重叠
        
    def detect_behavior_events(self, throat_ppg, throat_imu, behavior_type):
        """
        检测行为事件
        Args:
            throat_ppg: 喉咙PPG信号 (N,3) - 绿光/红外/红光
            throat_imu: 喉咙IMU信号 (N,3) - X/Y/Z轴加速度
            behavior_type: 行为类型 ('cough', 'swallow', 'chew')
        Returns:
            events: [(start_time, end_time, confidence), ...]
        """
        print(f"正在检测 {behavior_type} 行为事件...")
        
        # 预处理信号
        ppg_processed = self._preprocess_ppg(throat_ppg)
        imu_processed = self._preprocess_imu(throat_imu)
        
        # 提取特征
        features = self._extract_behavior_features(ppg_processed, imu_processed)
        
        # 基于行为类型的特定检测策略
        if behavior_type == 'cough':
            events = self._detect_cough_events(features)
        elif behavior_type == 'swallow':
            events = self._detect_swallow_events(features)
        elif behavior_type == 'chew':
            events = self._detect_chew_events(features)
        else:
            events = []
            
        print(f"检测到 {len(events)} 个 {behavior_type} 事件")
        return events
    
    def _preprocess_ppg(self, ppg_data):
        """PPG预处理"""
        processed = []
        for i in range(ppg_data.shape[1]):
            # 工频陷波
            ppg_nf = notch_filter(ppg_data[:, i], self.fs, mains=50, Q=30)
            # 高通去漂移
            ppg_hp = butter_highpass(ppg_nf, self.fs, fc=0.1, order=2)
            processed.append(ppg_hp)
        return np.column_stack(processed)
    
    def _preprocess_imu(self, imu_data):
        """IMU预处理"""
        # 高通去重力
        imu_hp = butter_highpass(imu_data, self.fs, fc=0.3, order=2)
        # 带通滤波
        imu_bp = butter_bandpass(imu_hp, self.fs, 0.5, 15.0, order=3)
        return imu_bp
    
    def _extract_behavior_features(self, ppg_data, imu_data):
        """提取行为相关特征"""
        window_samples = int(self.window_size * self.fs)
        stride_samples = int(window_samples * (1 - self.overlap))
        
        features = []
        time_stamps = []
        
        for start in range(0, len(ppg_data) - window_samples + 1, stride_samples):
            end = start + window_samples
            
            ppg_window = ppg_data[start:end]
            imu_window = imu_data[start:end]
            
            # 时间戳（窗口中心点）
            time_center = (start + end) / 2 / self.fs
            time_stamps.append(time_center)
            
            # PPG特征
            ppg_features = self._compute_ppg_features(ppg_window)
            
            # IMU特征
            imu_features = self._compute_imu_features(imu_window)
            
            # 合并特征
            window_features = np.concatenate([ppg_features, imu_features])
            features.append(window_features)
        
        return {
            'features': np.array(features),
            'time_stamps': np.array(time_stamps),
            'window_size': self.window_size
        }
    
    def _compute_ppg_features(self, ppg_window):
        """计算PPG特征"""
        features = []
        
        for channel in range(ppg_window.shape[1]):
            signal_ch = ppg_window[:, channel]
            
            # 统计特征
            features.extend([
                np.mean(signal_ch),
                np.std(signal_ch),
                np.var(signal_ch),
                np.ptp(signal_ch),  # peak-to-peak
                np.percentile(signal_ch, 25),
                np.percentile(signal_ch, 75)
            ])
            
            # 频域特征
            fft_vals = np.fft.fft(signal_ch)
            freqs = np.fft.fftfreq(len(signal_ch), 1/self.fs)
            
            # 主要频率成分的能量
            power_spectrum = np.abs(fft_vals)**2
            
            # 不同频带的能量
            low_band = np.sum(power_spectrum[(freqs >= 0.5) & (freqs <= 2.0)])
            mid_band = np.sum(power_spectrum[(freqs >= 2.0) & (freqs <= 8.0)])
            high_band = np.sum(power_spectrum[(freqs >= 8.0) & (freqs <= 15.0)])
            
            features.extend([low_band, mid_band, high_band])
            
            # 峰检测特征
            peaks, _ = find_peaks(signal_ch, height=np.std(signal_ch)*0.5)
            features.append(len(peaks))  # 峰数量
            
        return np.array(features)
    
    def _compute_imu_features(self, imu_window):
        """计算IMU特征"""
        features = []
        
        # 各轴独立特征
        for axis in range(imu_window.shape[1]):
            signal_axis = imu_window[:, axis]
            
            # 统计特征
            features.extend([
                np.mean(signal_axis),
                np.std(signal_axis),
                np.var(signal_axis),
                np.ptp(signal_axis)
            ])
            
            # 零交叉率
            zero_crossings = np.sum(np.diff(np.signbit(signal_axis)))
            features.append(zero_crossings)
        
        # 合成特征
        magnitude = np.linalg.norm(imu_window, axis=1)
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude)
        ])
        
        # 短时能量
        energy = short_time_energy(magnitude, self.fs, win_sec=0.1)
        features.extend([
            np.mean(energy),
            np.max(energy),
            np.std(energy)
        ])
        
        return np.array(features)
    
    def _detect_cough_events(self, features):
        """检测咳嗽事件 - 特征：短时高能量爆发"""
        feature_data = features['features']
        time_stamps = features['time_stamps']
        
        # 使用IMU能量峰值检测咳嗽
        # 咳嗽特征：IMU能量突然增大，持续时间短
        imu_energy_idx = -3  # IMU能量均值特征的索引
        energy_values = feature_data[:, imu_energy_idx]
        
        # 动态阈值：能量均值 + 2*标准差
        threshold = np.mean(energy_values) + 2 * np.std(energy_values)
        
        # 寻找超过阈值的峰值
        peaks, properties = find_peaks(
            energy_values, 
            height=threshold,
            distance=int(2.0 / features['window_size']),  # 最小间隔2秒
            width=1  # 最小宽度
        )
        
        events = []
        for i, peak_idx in enumerate(peaks):
            start_time = max(0, time_stamps[peak_idx] - 0.5)  # 峰值前0.5秒
            end_time = min(time_stamps[-1], time_stamps[peak_idx] + 1.0)  # 峰值后1秒
            confidence = min(1.0, energy_values[peak_idx] / threshold)
            
            events.append((start_time, end_time, confidence))
            
        return events
    
    def _detect_swallow_events(self, features):
        """检测吞咽事件 - 特征：PPG和IMU协同变化"""
        feature_data = features['features']
        time_stamps = features['time_stamps']
        
        # 吞咽特征：PPG幅度变化 + IMU运动模式
        # 使用PPG标准差和IMU能量的组合
        ppg_std_idx = 1  # PPG标准差特征索引（第一个通道）
        imu_energy_idx = -3  # IMU能量均值索引
        
        ppg_std = feature_data[:, ppg_std_idx]
        imu_energy = feature_data[:, imu_energy_idx]
        
        # 标准化特征
        ppg_std_norm = (ppg_std - np.mean(ppg_std)) / np.std(ppg_std)
        imu_energy_norm = (imu_energy - np.mean(imu_energy)) / np.std(imu_energy)
        
        # 组合特征：PPG变化 + IMU能量
        combined_feature = ppg_std_norm + 0.7 * imu_energy_norm
        
        # 动态阈值
        threshold = np.mean(combined_feature) + 1.5 * np.std(combined_feature)
        
        # 寻找吞咽事件
        peaks, _ = find_peaks(
            combined_feature,
            height=threshold,
            distance=int(3.0 / features['window_size']),  # 最小间隔3秒
            width=2  # 持续时间稍长
        )
        
        events = []
        for peak_idx in peaks:
            start_time = max(0, time_stamps[peak_idx] - 1.0)
            end_time = min(time_stamps[-1], time_stamps[peak_idx] + 2.0)
            confidence = min(1.0, combined_feature[peak_idx] / threshold)
            
            events.append((start_time, end_time, confidence))
            
        return events
    
    def _detect_chew_events(self, features):
        """检测咀嚼事件 - 特征：周期性重复运动"""
        feature_data = features['features']
        time_stamps = features['time_stamps']
        
        # 咀嚼特征：IMU的周期性变化
        imu_std_features = feature_data[:, -6:-3]  # IMU各轴标准差
        imu_combined_std = np.mean(imu_std_features, axis=1)
        
        # 使用滑动窗口检测周期性活动
        window_size = int(5.0 / features['window_size'])  # 5秒窗口
        
        events = []
        for i in range(len(imu_combined_std) - window_size):
            window_data = imu_combined_std[i:i+window_size]
            
            # 检测是否有持续的活动（标准差高于阈值）
            activity_threshold = np.mean(imu_combined_std) + 0.8 * np.std(imu_combined_std)
            active_ratio = np.sum(window_data > activity_threshold) / len(window_data)
            
            # 如果窗口内60%以上时间有活动，认为是咀嚼
            if active_ratio > 0.6:
                start_time = time_stamps[i]
                end_time = time_stamps[i + window_size - 1]
                confidence = active_ratio
                
                # 避免重复检测
                if not events or start_time > events[-1][1]:
                    events.append((start_time, end_time, confidence))
        
        return events

class DataLabeler:
    """数据标注器：将喉咙检测结果应用到耳道数据"""
    
    def __init__(self, data_dir='./hyx_data'):
        self.data_dir = data_dir
        self.detector = BehaviorDetector()
        
    def create_labeled_dataset(self):
        """创建标注数据集"""
        print("正在创建标注数据集...")
        
        labeled_data = []
        
        # 获取所有匹配的喉咙-耳道数据对
        data_pairs = self._find_data_pairs()
        
        for throat_file, ear_file, behavior in data_pairs:
            print(f"\n处理数据对: {behavior}")
            print(f"  喉咙文件: {throat_file}")
            print(f"  耳道文件: {ear_file}")
            
            # 加载数据
            throat_data = self._load_data(throat_file)
            ear_data = self._load_data(ear_file)
            
            if throat_data is None or ear_data is None:
                continue
            
            # 时间对齐（简化处理：假设同时开始记录）
            min_length = min(len(throat_data), len(ear_data))
            throat_data = throat_data[:min_length]
            ear_data = ear_data[:min_length]
            
            # 基于喉咙数据检测行为事件
            throat_ppg = throat_data[:, :3]  # 前3列为PPG
            throat_imu = throat_data[:, 3:6]  # 后3列为IMU
            
            events = self.detector.detect_behavior_events(
                throat_ppg, throat_imu, behavior
            )
            
            # 为耳道数据创建标签
            labels = self._create_labels(ear_data, events, behavior)
            
            # 保存标注结果
            labeled_sample = {
                'ear_data': ear_data,
                'labels': labels,
                'behavior': behavior,
                'events': events,
                'throat_file': throat_file,
                'ear_file': ear_file
            }
            
            labeled_data.append(labeled_sample)
            
            # 可视化检测结果
            self._visualize_detection_results(
                throat_data, ear_data, events, behavior, 
                os.path.basename(throat_file)
            )
        
        print(f"\n总共处理了 {len(labeled_data)} 个数据对")
        return labeled_data
    
    def _find_data_pairs(self):
        """找到匹配的喉咙-耳道数据对"""
        pairs = []
        
        # 扫描所有文件
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        # 行为映射
        behavior_keywords = {
            'cough': '咳嗽',
            'swallow': '吞咽',
            'chew': '咀嚼',
            'drink': '喝水'
        }
        
        for behavior_en, behavior_cn in behavior_keywords.items():
            throat_files = [f for f in files if '喉咙' in f and behavior_cn in f]
            ear_files = [f for f in files if '耳道' in f and behavior_cn in f]
            
            # 简单配对：按文件名相似度
            for throat_file in throat_files:
                for ear_file in ear_files:
                    # 检查是否为同一实验的数据
                    if self._is_matching_pair(throat_file, ear_file):
                        pairs.append((throat_file, ear_file, behavior_en))
                        break
        
        return pairs
    
    def _is_matching_pair(self, throat_file, ear_file):
        """判断是否为匹配的数据对"""
        # 简化判断：去除"喉咙"和"耳道"后的文件名是否相似
        throat_clean = throat_file.replace('喉咙', '').replace('-', '').replace(' ', '')
        ear_clean = ear_file.replace('耳道', '').replace('-', '').replace(' ', '')
        
        # 检查关键词是否匹配
        return throat_clean == ear_clean
    
    def _load_data(self, filename):
        """加载数据文件"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            data = np.loadtxt(filepath, skiprows=1, encoding='utf-8')
            return data
        except:
            try:
                data = np.loadtxt(filepath, skiprows=1, encoding='gbk')
                return data
            except Exception as e:
                print(f"无法加载文件 {filename}: {e}")
                return None
    
    def _create_labels(self, ear_data, events, behavior):
        """为耳道数据创建标签"""
        # 初始化为静息状态（标签0）
        labels = np.zeros(len(ear_data), dtype=int)
        
        # 行为标签映射
        behavior_map = {
            'cough': 2,    # 咳嗽
            'swallow': 3,  # 吞咽
            'chew': 1,     # 咀嚼
            'drink': 3     # 喝水归类为吞咽
        }
        
        behavior_label = behavior_map.get(behavior, 0)
        fs = 100  # 采样率
        
        # 根据检测到的事件设置标签
        for start_time, end_time, confidence in events:
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            
            # 确保索引在有效范围内
            start_idx = max(0, min(start_idx, len(labels)-1))
            end_idx = max(start_idx, min(end_idx, len(labels)))
            
            # 设置行为标签
            labels[start_idx:end_idx] = behavior_label
        
        return labels
    
    def _visualize_detection_results(self, throat_data, ear_data, events, behavior, filename):
        """可视化检测结果"""
        fs = 100
        time_axis = np.arange(len(throat_data)) / fs
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'行为检测结果: {behavior} - {filename}', fontsize=14)
        
        # 喉咙PPG信号
        axes[0].plot(time_axis, throat_data[:, 0], label='喉咙PPG(绿光)', alpha=0.7)
        axes[0].set_title('喉咙PPG信号')
        axes[0].set_ylabel('幅值')
        axes[0].legend()
        
        # 喉咙IMU信号
        throat_imu_mag = np.linalg.norm(throat_data[:, 3:6], axis=1)
        axes[1].plot(time_axis, throat_imu_mag, label='喉咙IMU幅值', color='orange', alpha=0.7)
        axes[1].set_title('喉咙IMU信号')
        axes[1].set_ylabel('幅值')
        axes[1].legend()
        
        # 耳道PPG信号
        axes[2].plot(time_axis, ear_data[:, 0], label='耳道PPG(绿光)', color='green', alpha=0.7)
        axes[2].set_title('耳道PPG信号')
        axes[2].set_ylabel('幅值')
        axes[2].legend()
        
        # 耳道IMU信号
        ear_imu_mag = np.linalg.norm(ear_data[:, 3:6], axis=1)
        axes[3].plot(time_axis, ear_imu_mag, label='耳道IMU幅值', color='red', alpha=0.7)
        axes[3].set_title('耳道IMU信号')
        axes[3].set_ylabel('幅值')
        axes[3].set_xlabel('时间 (秒)')
        axes[3].legend()
        
        # 标注检测到的事件
        for start_time, end_time, confidence in events:
            for ax in axes:
                ax.axvspan(start_time, end_time, alpha=0.3, color='yellow', 
                          label=f'{behavior}事件 (conf={confidence:.2f})' if ax == axes[0] else "")
        
        # 只在第一个子图显示事件标签
        if events:
            axes[0].legend()
        
        plt.tight_layout()
        
        # 保存图片
        save_dir = 'labeled_data_visualization'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{behavior}_{filename.replace(".txt", ".png")}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  可视化结果已保存: {save_path}")
    
    def save_labeled_dataset(self, labeled_data, output_file='labeled_dataset.npz'):
        """保存标注数据集"""
        print(f"\n正在保存标注数据集到 {output_file}...")
        
        all_ear_data = []
        all_labels = []
        all_behaviors = []
        all_events = []
        file_info = []
        
        for sample in labeled_data:
            all_ear_data.append(sample['ear_data'])
            all_labels.append(sample['labels'])
            all_behaviors.append(sample['behavior'])
            all_events.append(sample['events'])
            file_info.append({
                'throat_file': sample['throat_file'],
                'ear_file': sample['ear_file'],
                'behavior': sample['behavior']
            })
        
        # 保存为npz格式
        np.savez_compressed(
            output_file,
            ear_data=all_ear_data,
            labels=all_labels,
            behaviors=all_behaviors,
            events=all_events,
            file_info=file_info
        )
        
        print(f"标注数据集已保存: {output_file}")
        
        # 保存统计信息
        stats = self._compute_dataset_statistics(labeled_data)
        stats_file = output_file.replace('.npz', '_stats.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("标注数据集统计信息\n")
            f.write("="*50 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"统计信息已保存: {stats_file}")
        return output_file
    
    def _compute_dataset_statistics(self, labeled_data):
        """计算数据集统计信息"""
        stats = {}
        
        # 基本统计
        stats['数据对数量'] = len(labeled_data)
        
        # 行为类型统计
        behavior_counts = {}
        total_samples = 0
        label_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 静息/咀嚼/咳嗽/吞咽
        
        for sample in labeled_data:
            behavior = sample['behavior']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            
            labels = sample['labels']
            total_samples += len(labels)
            
            for label in range(4):
                label_counts[label] += np.sum(labels == label)
        
        stats['行为类型分布'] = behavior_counts
        stats['总样本点数'] = total_samples
        
        # 标签分布
        label_names = ['静息', '咀嚼', '咳嗽', '吞咽']
        for i, name in enumerate(label_names):
            count = label_counts[i]
            percentage = count / total_samples * 100
            stats[f'{name}标签数量'] = f"{count} ({percentage:.1f}%)"
        
        # 事件统计
        total_events = sum(len(sample['events']) for sample in labeled_data)
        stats['检测到的事件总数'] = total_events
        
        return stats

def main():
    """主函数"""
    print("=== 基于喉咙PPG&IMU的耳道数据标注系统 ===")
    
    # 初始化标注器
    labeler = DataLabeler()
    
    # 创建标注数据集
    labeled_data = labeler.create_labeled_dataset()
    
    if not labeled_data:
        print("未找到匹配的数据对，请检查数据文件")
        return
    
    # 保存标注数据集
    output_file = labeler.save_labeled_dataset(labeled_data)
    
    print(f"\n=== 标注完成 ===")
    print(f"标注数据集已保存: {output_file}")
    print("可视化结果保存在 labeled_data_visualization/ 目录下")

if __name__ == "__main__":
    main()
