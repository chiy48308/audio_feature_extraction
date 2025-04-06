"""
特徵提取測試模組 - 最新版本 (v2.0.0)
"""

import os
import numpy as np
import pandas as pd
import logging
from feature_extraction import FeatureExtractor

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_extraction():
    """測試特徵提取"""
    logger.info("開始提取特徵...")
    
    # 初始化特徵提取器
    extractor = FeatureExtractor()
    
    # 測試音頻文件
    audio_path = "test_audio/test.wav"
    
    # 提取特徵
    features, quality_metrics = extractor.extract_features(audio_path)
    
    # 保存特徵
    extractor.save_features(features, audio_path)
    
    # 輸出評估結果
    logger.info("特徵質量評估結果:")
    for metric, value in quality_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
        
    # 檢查CSV文件是否生成
    timestamp = os.listdir('features/csv')[0].split('_')[0]
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    features_csv_path = f'features/csv/{timestamp}_{base_name}_features.csv'
    statistics_csv_path = f'features/csv/{timestamp}_{base_name}_statistics.csv'
    
    if os.path.exists(features_csv_path) and os.path.exists(statistics_csv_path):
        logger.info(f"CSV文件已成功生成: {features_csv_path}, {statistics_csv_path}")
        
        # 讀取並顯示CSV文件的前幾行
        features_df = pd.read_csv(features_csv_path)
        statistics_df = pd.read_csv(statistics_csv_path)
        
        logger.info("特徵CSV文件前5行:")
        logger.info(features_df.head())
        
        logger.info("統計數據CSV文件:")
        logger.info(statistics_df)
    else:
        logger.error("CSV文件生成失敗")
        
    return features, quality_metrics

if __name__ == "__main__":
    test_feature_extraction() 