import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any
from pathlib import Path
import logging

class FeatureEvaluator:
    """特徵評估器類"""
    
    def __init__(self):
        """初始化評估器"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_feature_statistics(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        計算特徵統計信息
        
        參數:
            features_list: 特徵列表
            
        返回:
            統計信息字典
        """
        if not features_list:
            return {}
            
        # 提取所有特徵名稱
        feature_names = set()
        for features in features_list:
            feature_names.update(features.keys())
        feature_names.discard('file_path')
        
        statistics = {}
        
        # 計算每個特徵的統計信息
        for name in feature_names:
            values = []
            for features in features_list:
                if name in features:
                    value = features[name]
                    if isinstance(value, list):
                        values.extend(value)
                    else:
                        values.append(value)
            
            if values:
                values = np.array(values)
                statistics[f"{name}_min"] = float(np.min(values))
                statistics[f"{name}_max"] = float(np.max(values))
                statistics[f"{name}_mean"] = float(np.mean(values))
                statistics[f"{name}_std"] = float(np.std(values))
        
        return statistics
    
    def evaluate_feature_quality(self, features_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        評估特徵質量
        
        參數:
            features_list: 特徵列表
            
        返回:
            質量評估指標字典
        """
        if not features_list:
            return {}
            
        total_files = len(features_list)
        quality_metrics = {
            'total_files': total_files,
            'feature_integrity_rate': 100.0,  # 默認值
            'f0_quality_rate': 0.0,
            'mfcc_stability_rate': 0.0,
            'energy_stability_rate': 0.0
        }
        
        # 計算F0質量率
        f0_quality_sum = sum(f.get('f0_quality', 0) for f in features_list)
        quality_metrics['f0_quality_rate'] = (f0_quality_sum / total_files) * 100
        
        # 計算MFCC穩定性
        mfcc_std_threshold = 0.5
        mfcc_stable_count = sum(
            1 for f in features_list 
            if np.mean(f.get('mfcc_std', [1.0])) < mfcc_std_threshold
        )
        quality_metrics['mfcc_stability_rate'] = (mfcc_stable_count / total_files) * 100
        
        # 計算能量穩定性
        energy_std_threshold = 0.1
        energy_stable_count = sum(
            1 for f in features_list 
            if f.get('energy_std', 1.0) < energy_std_threshold
        )
        quality_metrics['energy_stability_rate'] = (energy_stable_count / total_files) * 100
        
        return quality_metrics
    
    def generate_evaluation_report(self, 
                                 features_list: List[Dict[str, Any]], 
                                 output_dir: str = "feature_evaluation") -> Dict[str, Any]:
        """
        生成評估報告
        
        參數:
            features_list: 特徵列表
            output_dir: 輸出目錄
            
        返回:
            評估報告字典
        """
        try:
            # 創建輸出目錄
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 計算統計信息和質量指標
            statistics = self.calculate_feature_statistics(features_list)
            quality_metrics = self.evaluate_feature_quality(features_list)
            
            # 生成詳細報告
            detailed_report = {
                'statistics': statistics,
                'quality_metrics': quality_metrics,
                'features_list': features_list
            }
            
            # 生成摘要報告
            summary_report = pd.DataFrame({
                'Metric': list(quality_metrics.keys()),
                'Value': list(quality_metrics.values())
            })
            
            # 保存報告
            with open(output_dir / 'evaluation_detailed.json', 'w', encoding='utf-8') as f:
                json.dump(detailed_report, f, indent=2, ensure_ascii=False)
                
            summary_report.to_csv(output_dir / 'evaluation_summary.csv', index=False)
            
            self.logger.info("評估報告生成完成")
            return detailed_report
            
        except Exception as e:
            self.logger.error(f"生成評估報告失敗: {str(e)}")
            raise
    
    def analyze_feature_distribution(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析特徵分佈
        
        參數:
            features_list: 特徵列表
            
        返回:
            分佈分析結果字典
        """
        if not features_list:
            return {}
            
        distribution_metrics = {}
        
        # 分析F0分佈
        f0_means = [f.get('f0_mean', 0) for f in features_list if f.get('f0_mean', 0) > 0]
        if f0_means:
            distribution_metrics['f0_distribution'] = {
                'mean': float(np.mean(f0_means)),
                'std': float(np.std(f0_means)),
                'percentiles': {
                    '25': float(np.percentile(f0_means, 25)),
                    '50': float(np.percentile(f0_means, 50)),
                    '75': float(np.percentile(f0_means, 75))
                }
            }
        
        # 分析MFCC分佈
        mfcc_means = []
        for f in features_list:
            if 'mfcc_mean' in f:
                mfcc_means.extend(f['mfcc_mean'])
        if mfcc_means:
            distribution_metrics['mfcc_distribution'] = {
                'mean': float(np.mean(mfcc_means)),
                'std': float(np.std(mfcc_means)),
                'percentiles': {
                    '25': float(np.percentile(mfcc_means, 25)),
                    '50': float(np.percentile(mfcc_means, 50)),
                    '75': float(np.percentile(mfcc_means, 75))
                }
            }
        
        # 分析能量分佈
        energy_means = [f.get('energy_mean', 0) for f in features_list]
        if energy_means:
            distribution_metrics['energy_distribution'] = {
                'mean': float(np.mean(energy_means)),
                'std': float(np.std(energy_means)),
                'percentiles': {
                    '25': float(np.percentile(energy_means, 25)),
                    '50': float(np.percentile(energy_means, 50)),
                    '75': float(np.percentile(energy_means, 75))
                }
            }
        
        return distribution_metrics 