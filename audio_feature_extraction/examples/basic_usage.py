from audio_feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt
import numpy as np

def plot_features(features):
    """繪製特徵可視化圖"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MFCC
    axes[0, 0].imshow(features['mfcc'], aspect='auto', origin='lower')
    axes[0, 0].set_title('MFCC Features')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('MFCC Coefficient')
    
    # F0
    axes[0, 1].plot(features['f0'])
    axes[0, 1].set_title('F0 Contour')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    
    # Energy
    axes[1, 0].plot(features['energy'])
    axes[1, 0].set_title('Energy')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Energy')
    
    # ZCR
    axes[1, 1].plot(features['zcr'])
    axes[1, 1].set_title('Zero Crossing Rate')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('ZCR')
    
    plt.tight_layout()
    return fig

def main():
    # 初始化特徵提取器
    extractor = FeatureExtractor()
    
    # 處理音頻文件
    audio_path = "path/to/your/audio.wav"
    result = extractor.process_audio(audio_path)
    
    # 獲取特徵和評估結果
    features = result['features']
    evaluation = result['evaluation']
    
    # 打印評估結果
    print("\n特徵評估結果：")
    print("-" * 50)
    print(f"MFCC 穩定性：{evaluation['mfcc_stability']}")
    print(f"F0 缺失率：{evaluation['f0_missing_rate']:.2%}")
    print(f"F0 品質：{evaluation['f0_quality']}")
    print(f"能量穩定性：{evaluation['energy_stability']}")
    print(f"ZCR 合理性：{evaluation['zcr_rationality']}")
    print(f"特徵完整性：{evaluation['feature_integrity']}")
    
    # 繪製特徵可視化圖
    fig = plot_features(features)
    plt.show()

if __name__ == "__main__":
    main() 