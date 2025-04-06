import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ExperimentAnalyzer:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.results = {
            'teacher': {'baseline': None, 'improved': None},
            'student': {'baseline': None, 'improved': None}
        }
    
    def load_results(self):
        """載入最新的實驗結果文件"""
        for speaker in ['teacher', 'student']:
            for method in ['baseline', 'improved']:
                # 找到最新的結果文件
                pattern = f"{method}_results_*.json"
                result_files = list(self.experiment_dir.glob(pattern))
                if result_files:
                    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        self.results[speaker][method] = json.load(f)
    
    def calculate_metrics(self, data: List[Dict]) -> Dict:
        """計算評估指標"""
        metrics = {
            'processing_time_mean': np.mean([r['processing_time'] for r in data]),
            'processing_time_std': np.std([r['processing_time'] for r in data]),
            'f0_std_mean': np.mean([np.std(r['f0']) for r in data]),
            'mfcc_std_mean': np.mean([np.std(np.array(r['mfcc']).flatten()) for r in data]),
            'energy_std_mean': np.mean([np.std(r['rms']) for r in data])
        }
        
        # 計算對齊相關指標（僅改進版本有這些數據）
        if data[0].get('alignments'):
            metrics.update({
                'num_segments_mean': np.mean([len(r['alignments']) for r in data]),
                'segment_duration_mean': np.mean([
                    np.mean([end - start for start, end in zip(r['start_points'], r['end_points'])])
                    for r in data if r['start_points'] and r['end_points']
                ])
            })
        
        return metrics
    
    def generate_summary_csv(self) -> str:
        """生成總體比較結果的CSV文件"""
        comparison_data = []
        
        for speaker in ['teacher', 'student']:
            for method in ['baseline', 'improved']:
                if self.results[speaker][method]:
                    metrics = self.calculate_metrics(self.results[speaker][method])
                    row = {
                        '說話者類型': speaker,
                        '方法': method,
                        '平均處理時間(秒)': metrics['processing_time_mean'],
                        '處理時間標準差': metrics['processing_time_std'],
                        'F0標準差均值': metrics['f0_std_mean'],
                        'MFCC標準差均值': metrics['mfcc_std_mean'],
                        '能量標準差均值': metrics['energy_std_mean']
                    }
                    
                    # 添加僅改進版本有的指標
                    if method == 'improved':
                        row.update({
                            '平均語音段數': metrics.get('num_segments_mean', 0),
                            '平均段落持續時間(秒)': metrics.get('segment_duration_mean', 0)
                        })
                    
                    comparison_data.append(row)
        
        # 創建DataFrame並保存為CSV
        df = pd.DataFrame(comparison_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.experiment_dir / f'1_summary_comparison_{timestamp}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        return str(csv_path)
    
    def generate_detailed_csv(self) -> str:
        """生成詳細分析的CSV文件"""
        analysis_data = []
        
        for speaker in ['teacher', 'student']:
            for method in ['baseline', 'improved']:
                if self.results[speaker][method]:
                    for result in self.results[speaker][method]:
                        row = {
                            '文件名': result['file_name'],
                            '說話者類型': speaker,
                            '方法': method,
                            '處理時間(秒)': result['processing_time'],
                            'F0均值': np.mean(result['f0']),
                            'F0標準差': np.std(result['f0']),
                            'MFCC均值': np.mean(np.array(result['mfcc']).flatten()),
                            'MFCC標準差': np.std(np.array(result['mfcc']).flatten()),
                            '能量均值': np.mean(result['rms']),
                            '能量標準差': np.std(result['rms']),
                            '過零率均值': np.mean(result['zcr']),
                            'RMSE': 0.0,  # 預設值
                            '對齊準確率': 0.0,  # 預設值
                            '異常切點數': 0  # 預設值
                        }
                        
                        # 添加改進版本特有的指標
                        if method == 'improved':
                            if 'start_points' in result and result['start_points']:
                                row.update({
                                    '語音段數': len(result['alignments']),
                                    '平均段落持續時間': np.mean([
                                        end - start 
                                        for start, end in zip(result['start_points'], result['end_points'])
                                    ])
                                })
                        
                        analysis_data.append(row)
        
        # 創建DataFrame並保存為CSV
        df = pd.DataFrame(analysis_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.experiment_dir / f'2_detailed_analysis_{timestamp}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        return str(csv_path)
    
    def generate_metrics_csv(self) -> str:
        """生成評估指標的CSV文件"""
        metrics_data = []
        
        # 處理時間統計
        processing_times = {'baseline': [], 'improved': []}
        for speaker in ['teacher', 'student']:
            for method in ['baseline', 'improved']:
                if self.results[speaker][method]:
                    processing_times[method].extend([
                        r['processing_time'] for r in self.results[speaker][method]
                    ])
        
        # 計算每個方法的評估指標
        for method in ['baseline', 'improved']:
            times = processing_times[method]
            metrics_data.append({
                '指標類型': '處理時間統計',
                '方法': method,
                '最小值': np.min(times),
                '最大值': np.max(times),
                '平均值': np.mean(times),
                '中位數': np.median(times),
                '標準差': np.std(times),
                '樣本數': len(times)
            })
        
        # 特徵穩定性統計
        for speaker in ['teacher', 'student']:
            for method in ['baseline', 'improved']:
                if self.results[speaker][method]:
                    data = self.results[speaker][method]
                    metrics = self.calculate_metrics(data)
                    
                    metrics_data.append({
                        '指標類型': f'{speaker}_特徵穩定性',
                        '方法': method,
                        '最小值': metrics['f0_std_mean'],
                        '最大值': metrics['mfcc_std_mean'],
                        '平均值': metrics['energy_std_mean'],
                        '中位數': np.median([metrics['f0_std_mean'], 
                                        metrics['mfcc_std_mean'], 
                                        metrics['energy_std_mean']]),
                        '標準差': np.std([metrics['f0_std_mean'], 
                                     metrics['mfcc_std_mean'], 
                                     metrics['energy_std_mean']]),
                        '樣本數': len(data)
                    })
        
        # 對齊效果統計（僅改進版本）
        for speaker in ['teacher', 'student']:
            if self.results[speaker]['improved']:
                data = self.results[speaker]['improved']
                segments = [len(r['alignments']) for r in data]
                durations = []
                for result in data:
                    if result['start_points'] and result['end_points']:
                        durations.extend([
                            end - start 
                            for start, end in zip(result['start_points'], result['end_points'])
                        ])
                
                metrics_data.append({
                    '指標類型': f'{speaker}_對齊效果',
                    '方法': 'improved',
                    '最小值': np.min(segments),
                    '最大值': np.max(segments),
                    '平均值': np.mean(segments),
                    '中位數': np.median(segments),
                    '標準差': np.std(segments),
                    '樣本數': len(segments)
                })
        
        df = pd.DataFrame(metrics_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.experiment_dir / f'3_evaluation_metrics_{timestamp}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        return str(csv_path)
    
    def generate_visualization(self):
        """生成可視化圖表"""
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 處理時間比較
        processing_times = {
            'baseline': [],
            'improved': []
        }
        for speaker in ['teacher', 'student']:
            for method in ['baseline', 'improved']:
                if self.results[speaker][method]:
                    processing_times[method].extend([
                        r['processing_time'] for r in self.results[speaker][method]
                    ])
        
        sns.boxplot(data=pd.DataFrame(processing_times), ax=axes[0, 0])
        axes[0, 0].set_title('處理時間比較')
        axes[0, 0].set_ylabel('時間 (秒)')
        
        # 2. 特徵穩定性比較
        stability_data = []
        for speaker in ['teacher', 'student']:
            for method in ['baseline', 'improved']:
                if self.results[speaker][method]:
                    metrics = self.calculate_metrics(self.results[speaker][method])
                    stability_data.append({
                        '方法': method,
                        'F0標準差': metrics['f0_std_mean'],
                        'MFCC標準差': metrics['mfcc_std_mean'],
                        '能量標準差': metrics['energy_std_mean']
                    })
        
        stability_df = pd.DataFrame(stability_data)
        stability_df.plot(x='方法', kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('特徵穩定性比較')
        axes[0, 1].set_ylabel('標準差')
        
        # 3. 語音段檢測比較（僅改進版本）
        segments_data = []
        for speaker in ['teacher', 'student']:
            if self.results[speaker]['improved']:
                segments_data.extend([
                    len(r['alignments']) for r in self.results[speaker]['improved']
                ])
        
        sns.histplot(segments_data, ax=axes[1, 0])
        axes[1, 0].set_title('語音段數分布（改進版本）')
        axes[1, 0].set_xlabel('語音段數')
        
        # 4. 段落持續時間分布（僅改進版本）
        durations = []
        for speaker in ['teacher', 'student']:
            if self.results[speaker]['improved']:
                for result in self.results[speaker]['improved']:
                    if result['start_points'] and result['end_points']:
                        durations.extend([
                            end - start 
                            for start, end in zip(result['start_points'], result['end_points'])
                        ])
        
        sns.histplot(durations, ax=axes[1, 1])
        axes[1, 1].set_title('段落持續時間分布（改進版本）')
        axes[1, 1].set_xlabel('持續時間 (秒)')
        
        # 保存圖表
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.experiment_dir / f'analysis_plots_{timestamp}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)

def main():
    # 設置實驗目錄
    experiment_dir = "/Users/chris/Desktop/數據紀錄日誌-處理/05_dtw_alignment_experiment/experiment_for_baseline_and_improved"
    
    # 創建分析器
    analyzer = ExperimentAnalyzer(experiment_dir)
    
    # 載入結果
    print("載入實驗結果...")
    analyzer.load_results()
    
    # 生成總體比較CSV
    print("\n生成總體比較CSV...")
    summary_csv = analyzer.generate_summary_csv()
    print(f"總體比較已保存到：{summary_csv}")
    
    # 生成詳細分析CSV
    print("\n生成詳細分析CSV...")
    detailed_csv = analyzer.generate_detailed_csv()
    print(f"詳細分析已保存到：{detailed_csv}")
    
    # 生成評估指標CSV
    print("\n生成評估指標CSV...")
    metrics_csv = analyzer.generate_metrics_csv()
    print(f"評估指標已保存到：{metrics_csv}")
    
    # 生成可視化
    print("\n生成可視化圖表...")
    plot_path = analyzer.generate_visualization()
    print(f"可視化圖表已保存到：{plot_path}")
    
    print("\n分析完成！生成了三個CSV文件：")
    print("1. 總體比較（summary_comparison）")
    print("2. 詳細分析（detailed_analysis）")
    print("3. 評估指標（evaluation_metrics）")

if __name__ == "__main__":
    main() 