"""
特徵提取測試模組 - 最新版本 (v2.0.0)
"""

import os
import numpy as np
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
        
    return features, quality_metrics

if __name__ == "__main__":
    test_feature_extraction() 