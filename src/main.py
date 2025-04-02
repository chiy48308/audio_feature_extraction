import os
import pandas as pd
import numpy as np
from audio_processor import AudioProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer
import logging
from typing import List, Dict
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_audio_files(audio_dir: str, scores_file: str) -> Dict:
    """處理所有音訊檔案並返回特徵"""
    logger = setup_logging()
    processor = AudioProcessor()
    
    # 讀取評分數據
    scores_df = pd.read_excel(scores_file)
    
    baseline_features = []
    treatment_features = []
    cv_improvements = []
    
    for idx, row in scores_df.iterrows():
        audio_file = os.path.join(audio_dir, row['audio_filename'])
        if not os.path.exists(audio_file):
            logger.warning(f"找不到音訊檔案: {audio_file}")
            continue
        
        # 載入並處理音訊
        audio, sr = processor.load_audio(audio_file)
        normalized_audio, features = processor.process_audio(audio)
        
        # 儲存正規化後的音訊
        output_dir = 'normalized_audio'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"normalized_{Path(audio_file).name}")
        processor.save_audio(normalized_audio, output_file, sr)
        
        # 收集特徵
        baseline_features.append(features['original'])
        treatment_features.append(features['normalized'])
        
        # 計算改善率
        cv_improvement = (
            (features['original']['rms_cv'] - features['normalized']['rms_cv'])
            / features['original']['rms_cv'] * 100
        )
        cv_improvements.append(cv_improvement)
    
    return {
        'baseline': baseline_features,
        'treatment': treatment_features,
        'cv_improvements': cv_improvements,
        'scores': scores_df['score'].values
    }

def main():
    logger = setup_logging()
    
    # 設定參數
    audio_dir = 'audio_data'
    scores_file = 'scores.xlsx'
    
    # 處理音訊檔案
    logger.info("開始處理音訊檔案...")
    results = process_audio_files(audio_dir, scores_file)
    
    # 訓練和評估模型
    logger.info("開始訓練模型...")
    trainer = ModelTrainer()
    baseline_df = trainer.prepare_features(results['baseline'])
    treatment_df = trainer.prepare_features(results['treatment'])
    
    model_comparison = trainer.compare_models(
        baseline_df,
        treatment_df,
        results['scores']
    )
    
    # 視覺化結果
    logger.info("生成視覺化結果...")
    visualizer = Visualizer()
    
    # 繪製 CV 分布圖
    visualizer.plot_cv_distribution(
        [f['rms_cv'] for f in results['baseline']],
        [f['rms_cv'] for f in results['treatment']],
        'cv_distribution.png'
    )
    
    # 繪製模型誤差比較圖
    visualizer.plot_error_comparison(
        model_comparison,
        'model_errors.png'
    )
    
    # 繪製 CV 改善率圖
    visualizer.plot_cv_improvement(
        results['cv_improvements'],
        'cv_improvement.png'
    )
    
    logger.info("實驗完成！")

if __name__ == "__main__":
    main() 