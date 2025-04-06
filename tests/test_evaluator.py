import unittest
import numpy as np
from pathlib import Path
import shutil
from audio_feature_extraction_toolkit import FeatureEvaluator

class TestFeatureEvaluator(unittest.TestCase):
    """測試特徵評估器"""
    
    def setUp(self):
        """設置測試環境"""
        self.evaluator = FeatureEvaluator()
        
        # 創建測試用的特徵數據
        self.test_features = [
            {
                'file_path': 'test1.wav',
                'f0_mean': 440.0,
                'f0_std': 1.0,
                'f0_missing_rate': 0.1,
                'f0_quality': 0.9,
                'mfcc_mean': [1.0, 2.0, 3.0],
                'mfcc_std': [0.1, 0.2, 0.3],
                'energy_mean': 0.8,
                'energy_std': 0.05
            },
            {
                'file_path': 'test2.wav',
                'f0_mean': 880.0,
                'f0_std': 2.0,
                'f0_missing_rate': 0.2,
                'f0_quality': 0.8,
                'mfcc_mean': [2.0, 3.0, 4.0],
                'mfcc_std': [0.2, 0.3, 0.4],
                'energy_mean': 0.9,
                'energy_std': 0.06
            }
        ]
    
    def test_calculate_feature_statistics(self):
        """測試特徵統計計算"""
        statistics = self.evaluator.calculate_feature_statistics(self.test_features)
        
        self.assertIsInstance(statistics, dict)
        self.assertIn('f0_mean_mean', statistics)
        self.assertIn('mfcc_mean_mean', statistics)
        self.assertIn('energy_mean_mean', statistics)
        
        # 檢查F0平均值計算
        self.assertAlmostEqual(statistics['f0_mean_mean'], 660.0)
    
    def test_evaluate_feature_quality(self):
        """測試特徵質量評估"""
        quality_metrics = self.evaluator.evaluate_feature_quality(self.test_features)
        
        self.assertIsInstance(quality_metrics, dict)
        self.assertIn('total_files', quality_metrics)
        self.assertIn('feature_integrity_rate', quality_metrics)
        self.assertIn('f0_quality_rate', quality_metrics)
        self.assertIn('mfcc_stability_rate', quality_metrics)
        self.assertIn('energy_stability_rate', quality_metrics)
        
        # 檢查文件總數
        self.assertEqual(quality_metrics['total_files'], 2)
    
    def test_generate_evaluation_report(self):
        """測試評估報告生成"""
        # 創建臨時輸出目錄
        output_dir = Path('test_output')
        
        try:
            report = self.evaluator.generate_evaluation_report(
                self.test_features,
                output_dir=str(output_dir)
            )
            
            self.assertIsInstance(report, dict)
            self.assertIn('statistics', report)
            self.assertIn('quality_metrics', report)
            self.assertIn('features_list', report)
            
            # 檢查輸出文件是否存在
            self.assertTrue((output_dir / 'evaluation_detailed.json').exists())
            self.assertTrue((output_dir / 'evaluation_summary.csv').exists())
            
        finally:
            # 清理臨時文件
            if output_dir.exists():
                shutil.rmtree(output_dir)
    
    def test_analyze_feature_distribution(self):
        """測試特徵分佈分析"""
        distribution = self.evaluator.analyze_feature_distribution(self.test_features)
        
        self.assertIsInstance(distribution, dict)
        self.assertIn('f0_distribution', distribution)
        self.assertIn('mfcc_distribution', distribution)
        self.assertIn('energy_distribution', distribution)
        
        # 檢查分佈統計值
        f0_dist = distribution['f0_distribution']
        self.assertIn('mean', f0_dist)
        self.assertIn('std', f0_dist)
        self.assertIn('percentiles', f0_dist)

if __name__ == '__main__':
    unittest.main() 