"""
特徵提取實驗比較腳本
比較baseline和improved版本的特徵提取效果
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from feature_extraction import FeatureExtractor

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_audio_files(audio_dir, output_dir, version):
    """處理音頻文件並保存結果"""
    logger.info(f"開始處理 {version} 版本...")
    
    # 初始化特徵提取器
    extractor = FeatureExtractor()
    
    # 獲取所有音頻文件
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('**/*.wav'))
    
    if not audio_files:
        logger.error(f"在 {audio_dir} 中未找到音頻文件")
        return None
    
    # 用於存儲所有文件的統計數據
    all_statistics = []
    
    for audio_file in audio_files:
        logger.info(f"處理文件: {audio_file.name}")
        
        try:
            # 提取特徵
            features, quality_metrics = extractor.extract_features(str(audio_file))
            
            if features is None or quality_metrics is None:
                logger.warning(f"無法從 {audio_file.name} 提取特徵")
                continue
            
            # 創建統計數據DataFrame
            stats = pd.DataFrame()
            
            # MFCC統計
            if 'mfcc' in features:
                mfcc = features['mfcc']
                for i in range(mfcc.shape[0]):
                    stats.loc['mean', f'mfcc_{i+1}'] = np.mean(mfcc[i])
                    stats.loc['std', f'mfcc_{i+1}'] = np.std(mfcc[i])
                    stats.loc['min', f'mfcc_{i+1}'] = np.min(mfcc[i])
                    stats.loc['max', f'mfcc_{i+1}'] = np.max(mfcc[i])
            
            # F0統計
            if 'f0' in features:
                f0 = features['f0'][0]  # 只使用基本F0，不使用delta
                voiced = f0 > 0
                if np.sum(voiced) > 0:
                    stats.loc['mean', 'f0'] = np.mean(f0[voiced])
                    stats.loc['std', 'f0'] = np.std(f0[voiced])
                    stats.loc['min', 'f0'] = np.min(f0[voiced])
                    stats.loc['max', 'f0'] = np.max(f0[voiced])
                    stats.loc['missing_rate', 'f0'] = 1 - np.sum(voiced) / len(voiced)
                else:
                    stats.loc['mean', 'f0'] = 0
                    stats.loc['std', 'f0'] = 0
                    stats.loc['min', 'f0'] = 0
                    stats.loc['max', 'f0'] = 0
                    stats.loc['missing_rate', 'f0'] = 1
            
            # 能量統計
            if 'energy' in features:
                energy = features['energy'][0]  # 只使用基本能量，不使用delta
                stats.loc['mean', 'energy'] = np.mean(energy)
                stats.loc['std', 'energy'] = np.std(energy)
                stats.loc['min', 'energy'] = np.min(energy)
                stats.loc['max', 'energy'] = np.max(energy)
            
            # ZCR統計
            if 'zcr' in features:
                zcr = features['zcr'][0]  # 只使用基本ZCR，不使用delta
                stats.loc['mean', 'zcr'] = np.mean(zcr)
                stats.loc['std', 'zcr'] = np.std(zcr)
                stats.loc['min', 'zcr'] = np.min(zcr)
                stats.loc['max', 'zcr'] = np.max(zcr)
            
            # 添加質量指標
            for metric, value in quality_metrics.items():
                stats.loc['quality', metric] = value
            
            # 添加文件信息
            stats.loc[:, 'file_name'] = audio_file.name
            stats.loc[:, 'version'] = version
            
            # 添加到所有統計數據列表
            all_statistics.append(stats)
            
        except Exception as e:
            logger.error(f"處理文件 {audio_file.name} 時出錯: {e}")
            continue
    
    # 合併所有統計數據
    if all_statistics:
        try:
            combined_stats = pd.concat(all_statistics)
            
            # 保存詳細統計數據
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f'{version}_statistics.csv'
            combined_stats.to_csv(output_path)
            logger.info(f"已保存統計數據到: {output_path}")
            
            # 計算並保存平均值
            mean_stats = combined_stats.groupby(['version']).mean()
            mean_path = output_dir / f'{version}_mean_statistics.csv'
            mean_stats.to_csv(mean_path)
            logger.info(f"已保存平均統計數據到: {mean_path}")
            
            return combined_stats
        except Exception as e:
            logger.error(f"保存統計數據時出錯: {e}")
            return None
    else:
        logger.error("沒有找到任何統計數據")
        return None

def main():
    """主函數"""
    # 設置目錄
    audio_dir = "processed_audio"
    baseline_dir = "experiments/baseline"
    improved_dir = "experiments/improved"
    
    # 確保目錄存在
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(improved_dir, exist_ok=True)
    
    # 處理baseline版本
    baseline_stats = process_audio_files(audio_dir, baseline_dir, "baseline")
    
    # 處理improved版本
    improved_stats = process_audio_files(audio_dir, improved_dir, "improved")
    
    if baseline_stats is not None and improved_stats is not None:
        try:
            # 計算改進百分比
            comparison = pd.DataFrame()
            
            # 計算每個指標的平均值
            baseline_means = baseline_stats.groupby('version').mean()
            improved_means = improved_stats.groupby('version').mean()
            
            # 計算改進百分比
            for col in baseline_means.columns:
                if col not in ['file_name', 'version']:
                    baseline_mean = baseline_means.loc['baseline', col]
                    improved_mean = improved_means.loc['improved', col]
                    if baseline_mean != 0:
                        improvement = ((improved_mean - baseline_mean) / abs(baseline_mean)) * 100
                    else:
                        improvement = np.inf if improved_mean > 0 else -np.inf if improved_mean < 0 else 0
                    
                    comparison.loc[col, 'baseline'] = baseline_mean
                    comparison.loc[col, 'improved'] = improved_mean
                    comparison.loc[col, 'improvement_percentage'] = improvement
            
            # 保存比較結果
            comparison_path = Path("experiments") / "comparison_results.csv"
            comparison.to_csv(comparison_path)
            logger.info(f"已保存比較結果到: {comparison_path}")
            
            # 顯示主要改進
            logger.info("\n主要改進:")
            for idx, row in comparison.iterrows():
                if not pd.isna(row['improvement_percentage']):
                    logger.info(f"{idx}: {row['improvement_percentage']:.2f}%")
        
        except Exception as e:
            logger.error(f"計算比較結果時出錯: {e}")

if __name__ == "__main__":
    main() 