import pandas as pd
import numpy as np
from pathlib import Path
import json
import librosa
from typing import Dict, List, Tuple
import csv
from datetime import datetime

class AlignmentEvaluator:
    def __init__(self):
        self.results = {
            'baseline': [],
            'improved': []
        }
    
    def calculate_rmse(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """計算對齊的RMSE（均方根誤差）"""
        return np.sqrt(np.mean((predicted - ground_truth) ** 2)) * 1000  # 轉換為毫秒
    
    def calculate_alignment_consistency(self, 
                                     predicted_starts: np.ndarray,
                                     predicted_ends: np.ndarray,
                                     ground_truth_starts: np.ndarray,
                                     ground_truth_ends: np.ndarray) -> Tuple[float, int]:
        """計算對齊一致性和異常切點數量"""
        start_diffs = np.abs(predicted_starts - ground_truth_starts) * 1000  # 轉換為毫秒
        end_diffs = np.abs(predicted_ends - ground_truth_ends) * 1000
        
        # 計算符合250ms閾值的比例
        valid_alignments = np.logical_and(start_diffs < 250, end_diffs < 250)
        consistency_rate = np.mean(valid_alignments) * 100
        
        # 檢測異常切點（過早或過晚）
        abnormal_points = np.sum(np.logical_or(start_diffs >= 250, end_diffs >= 250))
        
        return consistency_rate, abnormal_points
    
    def calculate_accuracy_rate(self, 
                              predicted_alignments: List[Tuple[float, float]],
                              ground_truth_alignments: List[Tuple[float, float]]) -> float:
        """計算語音-文本對應準確率"""
        total_segments = len(ground_truth_alignments)
        correct_alignments = 0
        
        for pred, truth in zip(predicted_alignments, ground_truth_alignments):
            # 如果預測的起訖點與真實值的偏差都在閾值內，視為正確對齊
            if (abs(pred[0] - truth[0]) * 1000 < 250 and 
                abs(pred[1] - truth[1]) * 1000 < 250):
                correct_alignments += 1
        
        return (correct_alignments / total_segments) * 100
    
    def evaluate_alignment(self, 
                         method: str,
                         audio_file: str,
                         predicted_data: Dict,
                         ground_truth_data: Dict) -> Dict:
        """評估單個音頻文件的對齊效果"""
        
        # 計算RMSE
        rmse = self.calculate_rmse(
            np.array(predicted_data['alignments']),
            np.array(ground_truth_data['alignments'])
        )
        
        # 計算對齊一致性
        consistency_rate, abnormal_points = self.calculate_alignment_consistency(
            np.array(predicted_data['start_points']),
            np.array(predicted_data['end_points']),
            np.array(ground_truth_data['start_points']),
            np.array(ground_truth_data['end_points'])
        )
        
        # 計算對應準確率
        accuracy_rate = self.calculate_accuracy_rate(
            predicted_data['segment_alignments'],
            ground_truth_data['segment_alignments']
        )
        
        result = {
            'method': method,
            'audio_file': Path(audio_file).name,
            'rmse_ms': rmse,
            'rmse_within_threshold': rmse <= 200,
            'consistency_rate': consistency_rate,
            'abnormal_points': abnormal_points,
            'accuracy_rate': accuracy_rate,
            'meets_accuracy_threshold': accuracy_rate >= 95
        }
        
        self.results[method].append(result)
        return result
    
    def save_results_csv(self, output_path: str):
        """將評估結果保存為CSV文件"""
        all_results = []
        for method in ['baseline', 'improved']:
            all_results.extend(self.results[method])
        
        df = pd.DataFrame(all_results)
        
        # 添加時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(output_path) / f'alignment_evaluation_{timestamp}.csv')
        
        # 保存CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 計算並保存統計摘要
        summary = {
            'method': ['baseline', 'improved'],
            'avg_rmse': [
                np.mean([r['rmse_ms'] for r in self.results['baseline']]),
                np.mean([r['rmse_ms'] for r in self.results['improved']])
            ],
            'avg_consistency': [
                np.mean([r['consistency_rate'] for r in self.results['baseline']]),
                np.mean([r['consistency_rate'] for r in self.results['improved']])
            ],
            'avg_accuracy': [
                np.mean([r['accuracy_rate'] for r in self.results['baseline']]),
                np.mean([r['accuracy_rate'] for r in self.results['improved']])
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_path = str(Path(output_path).parent / f'alignment_summary_{timestamp}.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        return output_path, summary_path

def main():
    # 設置路徑
    base_dir = Path("/Users/chris/Desktop/數據紀錄日誌-處理/05_dtw_alignment_experiment")
    experiment_dir = base_dir / "experiment_for_baseline_and_improved"
    
    # 創建評估器
    evaluator = AlignmentEvaluator()
    
    # 讀取基準版本和改進版本的結果
    for method in ['baseline', 'improved']:
        # 讀取教師音頻結果
        teacher_results_path = experiment_dir / f"teacher_{method}_results.json"
        with open(teacher_results_path, 'r', encoding='utf-8') as f:
            teacher_results = json.load(f)
        
        # 讀取學生音頻結果
        student_results_path = experiment_dir / f"student_{method}_results.json"
        with open(student_results_path, 'r', encoding='utf-8') as f:
            student_results = json.load(f)
        
        # 評估每個音頻文件的對齊效果
        for teacher_result, student_result in zip(teacher_results, student_results):
            evaluator.evaluate_alignment(
                method,
                teacher_result['file_name'],
                teacher_result['alignments'],
                student_result['alignments']
            )
    
    # 保存評估結果
    csv_path, summary_path = evaluator.save_results_csv(str(experiment_dir))
    print(f"評估結果已保存到：{csv_path}")
    print(f"統計摘要已保存到：{summary_path}")

if __name__ == "__main__":
    main() 