import os
from pathlib import Path
from audio_feature_extraction_toolkit import AudioFeatureExtractor, FeatureEvaluator

def main():
    # 設置音頻文件目錄
    audio_dir = "path/to/your/audio/files"  # 請替換為實際的音頻文件目錄
    
    # 創建特徵提取器實例
    extractor = AudioFeatureExtractor(
        sr=22050,              # 採樣率
        frame_length=1024,     # 幀長度
        hop_length=256,        # 跳躍長度
        n_mfcc=13,            # MFCC特徵數量
        pre_emphasis=0.97      # 預加重係數
    )
    
    # 創建評估器實例
    evaluator = FeatureEvaluator()
    
    try:
        # 批量處理音頻文件
        print("開始處理音頻文件...")
        features_list = extractor.batch_process(audio_dir)
        print(f"成功處理 {len(features_list)} 個音頻文件")
        
        # 生成評估報告
        print("\n生成評估報告...")
        evaluation_report = evaluator.generate_evaluation_report(
            features_list,
            output_dir="feature_evaluation"
        )
        
        # 分析特徵分佈
        print("\n分析特徵分佈...")
        distribution = evaluator.analyze_feature_distribution(features_list)
        
        # 打印主要質量指標
        print("\n特徵質量指標:")
        quality_metrics = evaluation_report['quality_metrics']
        print(f"- 特徵完整性: {quality_metrics['feature_integrity_rate']:.2f}%")
        print(f"- F0質量率: {quality_metrics['f0_quality_rate']:.2f}%")
        print(f"- MFCC穩定性: {quality_metrics['mfcc_stability_rate']:.2f}%")
        print(f"- 能量穩定性: {quality_metrics['energy_stability_rate']:.2f}%")
        
        print("\n評估報告已保存到 'feature_evaluation' 目錄")
        
    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 