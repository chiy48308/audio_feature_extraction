import os
from pathlib import Path
from feature_extraction import FeatureExtractor
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 初始化特徵提取器
    extractor = FeatureExtractor(output_dir="features")
    
    # 測試音頻文件路徑
    audio_path = "test_audio.wav"  # 請確保此文件存在
    
    try:
        # 提取特徵
        logger.info("開始提取特徵...")
        features = extractor.extract_features(audio_path)
        
        # 評估特徵質量
        logger.info("評估特徵質量...")
        quality_metrics = extractor.evaluate_feature_quality(features)
        
        # 保存結果
        logger.info("保存結果...")
        extractor.save_results(features, quality_metrics, audio_path)
        
        # 顯示質量評估結果
        logger.info("特徵質量評估結果:")
        for metric, value in quality_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"測試失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 