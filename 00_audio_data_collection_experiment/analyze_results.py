import os
import json
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

def load_results(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                result = json.load(f)
                results.append(result)
    return results

def calculate_statistics(results):
    stats = {
        '準確率': [],
        '召回率': [],
        'F1分數': [],
        '處理延遲(秒)': [],
        '分段準確率': [],
        'RMSE': [],
        'SNR': []
    }
    
    for result in results:
        stats['準確率'].append(result.get('accuracy', 0))
        stats['召回率'].append(result.get('recall', 0))
        stats['F1分數'].append(result.get('f1_score', 0))
        stats['處理延遲(秒)'].append(result.get('processing_delay', 0))
        stats['分段準確率'].append(result.get('segmentation_accuracy', 0))
        stats['RMSE'].append(result.get('rmse', 0))
        stats['SNR'].append(result.get('snr', 0))
    
    return {k: np.mean(v) for k, v in stats.items()}

def compare_results():
    baseline_results = load_results('vad_experiment_results/baseline')
    improved_results = load_results('vad_experiment_results/improved')
    
    baseline_stats = calculate_statistics(baseline_results)
    improved_stats = calculate_statistics(improved_results)
    
    # 準備表格數據
    table_data = []
    for metric in baseline_stats.keys():
        baseline_value = baseline_stats[metric]
        improved_value = improved_stats[metric]
        improvement = ((improved_value - baseline_value) / baseline_value) * 100
        table_data.append([
            metric,
            f"{baseline_value:.3f}",
            f"{improved_value:.3f}",
            f"{improvement:+.1f}%"
        ])
    
    print("\n性能比較:")
    print(tabulate(table_data, headers=['指標', '基準版本', '改進版本', '改進百分比'], tablefmt='grid'))
    
    # 檢查性能要求
    requirements = {
        '靜音檢測通過率': 25.9,
        '音量穩定性': 0.2,
        '信噪比通過率': 24.8
    }
    
    print("\n性能要求檢查:")
    for req, threshold in requirements.items():
        if req == '靜音檢測通過率':
            actual = improved_stats['準確率'] * 100
        elif req == '音量穩定性':
            actual = (1 - improved_stats['RMSE']) * 100
        else:  # 信噪比通過率
            actual = improved_stats['SNR']
            
        status = "✓" if actual >= threshold else "✗"
        print(f"{status} {req}: {actual:.1f}% (目標: {threshold}%)")

if __name__ == '__main__':
    compare_results() 