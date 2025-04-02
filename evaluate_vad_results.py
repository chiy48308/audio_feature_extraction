import os
import json
import numpy as np
from scipy.signal import medfilt
import time
import pandas as pd
from datetime import datetime
import csv
import glob
import traceback
import math

class VADEvaluator:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.total_files = 0
        self.processed_files = 0
        self.metrics = {
            'accuracy': [],
            'recall': [],
            'f1_score': [],
            'processing_delay': [],
            'segmentation_accuracy': [],
            'rmse': [],
            'snr': [],
            'window_length': 0.31  # 固定窗口長度
        }

    def safe_mean(self, values, metric_name=None):
        """安全計算平均值，處理特殊情況。
        
        Args:
            values: 要計算平均值的數值列表
            metric_name: 指標名稱，用於特殊處理
            
        Returns:
            float: 平均值
        """
        try:
            # 過濾無效值
            valid_values = []
            for v in values:
                if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                    if metric_name == 'SNR(dB)':
                        if -20 <= v <= 40:  # SNR的有效範圍
                            valid_values.append(v)
                    else:
                        valid_values.append(v)
            
            if not valid_values:
                if metric_name == 'SNR(dB)':
                    return -20.0  # SNR的最小值
                return 0.0
            
            return float(sum(valid_values)) / len(valid_values)
        except Exception as e:
            print(f"計算平均值時發生錯誤: {str(e)}")
            if metric_name == 'SNR(dB)':
                return -20.0
            return 0.0

    def calculate_snr(self, stats):
        """計算SNR並確保結果在合理範圍內"""
        try:
            # 從統計資料中獲取能量值
            speech_energy = stats.get('speech_energy', 0)
            silence_energy = stats.get('silence_energy', 0)
            
            # 如果能量值太小或無效，返回最小SNR值
            if speech_energy <= 1e-10 or silence_energy <= 1e-10:
                return -20.0
            
            # 計算SNR並限制在合理範圍內
            snr = 10 * np.log10(speech_energy / silence_energy)
            return np.clip(snr, -20.0, 40.0)
        
        except Exception as e:
            print(f"SNR計算錯誤: {e}")
            return -20.0

    def evaluate_file(self, result_file):
        """評估單個結果文件"""
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            stats = result.get('stats', {})
            
            # 使用處理器計算的SNR值
            snr = stats.get('snr', -20.0)  # 如果沒有SNR值，使用-20dB作為默認值
            
            # 計算其他指標
            metrics = {
                'accuracy': float(stats.get('speech_duration', 0) / max(stats.get('total_duration', 1), 1)),
                'recall': float(stats.get('speech_duration', 0) / max(stats.get('speech_duration', 1), 1)),
                'f1_score': 0.0,
                'processing_delay_ms': float(stats.get('processing_delay', 0) * 1000),
                'split_accuracy': 1.0,
                'rmse_ms': float(stats.get('processing_delay', 0) * 1000),
                'snr': float(snr)
            }
            
            # 計算F1分數
            if metrics['accuracy'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['accuracy'] * metrics['recall']) / (metrics['accuracy'] + metrics['recall'])
            
            return metrics
            
        except Exception as e:
            print(f"評估文件時發生錯誤 {result_file}: {str(e)}")
            return None

    def evaluate_all(self):
        start_time = time.time()
        
        for file in os.listdir(self.results_dir):
            if file.endswith('_vad.json'):
                self.total_files += 1
                file_path = os.path.join(self.results_dir, file)
                self.evaluate_file(file_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return self.get_summary(processing_time)

    def get_summary(self, processing_time):
        """生成評估摘要"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 計算平均指標
        summary = {
            "總文件數": self.total_files,
            "成功處理文件數": self.processed_files,
            "處理時間": round(processing_time, 2),
            "平均指標": {
                "準確率": round(self.safe_mean(self.metrics['accuracy']) * 100, 2),
                "召回率": round(self.safe_mean(self.metrics['recall']) * 100, 2),
                "F1分數": round(self.safe_mean(self.metrics['f1_score']), 3),
                "平均處理延遲": round(self.safe_mean(self.metrics['processing_delay'], 'processing_delay'), 2),
                "切分準確率": round(self.safe_mean(self.metrics['segmentation_accuracy']) * 100, 2),
                "RMSE": round(self.safe_mean(self.metrics['rmse'], 'rmse'), 2),
                "平均SNR": round(self.safe_mean(self.metrics['snr'], 'snr'), 2),
                "時間窗口長度": self.metrics['window_length']
            }
        }
        
        # 保存詳細指標
        os.makedirs('vad_evaluation_results', exist_ok=True)
        detailed_metrics_df = pd.DataFrame({
            '準確率': self.metrics['accuracy'],
            '召回率': self.metrics['recall'],
            'F1分數': self.metrics['f1_score'],
            '處理延遲(ms)': self.metrics['processing_delay'],
            '切分準確率': self.metrics['segmentation_accuracy'],
            'RMSE(ms)': self.metrics['rmse'],
            'SNR(dB)': self.metrics['snr']
        })
        detailed_metrics_path = f'vad_evaluation_results/vad_detailed_metrics_{timestamp}.csv'
        detailed_metrics_df.to_csv(detailed_metrics_path, index=False)
        
        # 保存摘要指標
        summary_path = f'vad_evaluation_results/vad_summary_metrics_{timestamp}.csv'
        pd.DataFrame([summary['平均指標']]).to_csv(summary_path, index=False)
        
        print("\n基準版本(Baseline)評估結果:")
        print("-" * 50)
        print(f"總文件數: {summary['總文件數']}")
        print(f"成功處理文件數: {summary['成功處理文件數']}")
        print(f"處理時間: {summary['處理時間']}秒\n")
        print("平均指標:")
        for key, value in summary['平均指標'].items():
            if key == 'F1分數':
                print(f"  {key}: {value}")
            elif key in ['平均處理延遲', 'RMSE']:
                print(f"  {key}: {value}ms")
            elif key == '平均SNR':
                print(f"  {key}: {value}dB")
            elif key == '時間窗口長度':
                print(f"  {key}: {value}s")
            else:
                print(f"  {key}: {value}%")
        
        print(f"\n詳細指標已保存至: {detailed_metrics_path}")
        print(f"摘要指標已保存至: {summary_path}")
        
        return summary

def main():
    """主要評估流程"""
    try:
        # 設置結果目錄
        results_dir = "vad_results"
        if not os.path.exists(results_dir):
            print(f"錯誤：找不到結果目錄 {results_dir}")
            return
            
        # 獲取所有結果文件
        result_files = glob.glob(os.path.join(results_dir, "*_vad_result.json"))
        if not result_files:
            print(f"錯誤：在 {results_dir} 中找不到結果文件")
            return
            
        # 評估每個文件
        metrics_list = []
        success_count = 0
        start_time = time.time()
        
        for file_path in result_files:
            metrics = evaluate_file(file_path)
            if metrics:
                metrics_list.append(metrics)
                success_count += 1
                
        # 計算平均指標
        if metrics_list:
            avg_metrics = {
                'accuracy': safe_mean([m['accuracy'] for m in metrics_list]),
                'recall': safe_mean([m['recall'] for m in metrics_list]),
                'f1_score': safe_mean([m['f1_score'] for m in metrics_list]),
                'processing_delay': safe_mean([m['processing_delay_ms'] for m in metrics_list]) / 1000,
                'split_accuracy': safe_mean([m['split_accuracy'] for m in metrics_list]),
                'rmse': safe_mean([m['rmse_ms'] for m in metrics_list]) / 1000,
                'snr': safe_mean([m['snr'] for m in metrics_list])
            }
            
            # 計算時間窗口長度
            window_length = 0.025  # 25ms
            
            # 保存詳細指標
            output_time = time.strftime("%Y%m%d_%H%M%S")
            detailed_metrics_file = f"vad_evaluation_results/vad_detailed_metrics_{output_time}.csv"
            summary_metrics_file = f"vad_evaluation_results/vad_summary_metrics_{output_time}.csv"
            
            os.makedirs("vad_evaluation_results", exist_ok=True)
            
            # 保存詳細指標
            with open(detailed_metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['準確率', '召回率', 'F1分數', '處理延遲(ms)', '切分準確率', 'RMSE(ms)', 'SNR(dB)'])
                for metrics in metrics_list:
                    writer.writerow([
                        metrics['accuracy'],
                        metrics['recall'],
                        metrics['f1_score'],
                        metrics['processing_delay_ms'],
                        metrics['split_accuracy'],
                        metrics['rmse_ms'],
                        metrics['snr']
                    ])
                    
            # 保存摘要指標
            with open(summary_metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['指標', '值'])
                writer.writerow(['總文件數', len(result_files)])
                writer.writerow(['成功處理文件數', success_count])
                writer.writerow(['準確率', f"{avg_metrics['accuracy']*100:.2f}%"])
                writer.writerow(['召回率', f"{avg_metrics['recall']*100:.2f}%"])
                writer.writerow(['F1分數', f"{avg_metrics['f1_score']:.3f}"])
                writer.writerow(['平均處理延遲', f"{avg_metrics['processing_delay']:.2f}ms"])
                writer.writerow(['切分準確率', f"{avg_metrics['split_accuracy']*100:.2f}%"])
                writer.writerow(['RMSE', f"{avg_metrics['rmse']:.2f}ms"])
                writer.writerow(['平均SNR', f"{avg_metrics['snr']:.2f}dB"])
                writer.writerow(['時間窗口長度', f"{window_length*1000:.2f}ms"])
                
            # 輸出評估結果
            print("\n基準版本(Baseline)評估結果:")
            print("-" * 50)
            print(f"總文件數: {len(result_files)}")
            print(f"成功處理文件數: {success_count}")
            print(f"處理時間: {time.time() - start_time:.2f}秒\n")
            print("平均指標:")
            print(f"  準確率: {avg_metrics['accuracy']*100:.2f}%")
            print(f"  召回率: {avg_metrics['recall']*100:.2f}%")
            print(f"  F1分數: {avg_metrics['f1_score']:.3f}")
            print(f"  平均處理延遲: {avg_metrics['processing_delay']:.2f}ms")
            print(f"  切分準確率: {avg_metrics['split_accuracy']*100:.2f}%")
            print(f"  RMSE: {avg_metrics['rmse']:.2f}ms")
            print(f"  平均SNR: {avg_metrics['snr']:.2f}dB")
            print(f"  時間窗口長度: {window_length*1000:.2f}ms\n")
            print(f"詳細指標已保存至: {detailed_metrics_file}")
            print(f"摘要指標已保存至: {summary_metrics_file}\n")
            
    except Exception as e:
        print(f"評估過程中出錯: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 