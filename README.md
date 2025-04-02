# 音頻特徵提取工具包

這是一個用於音頻特徵提取和分析的Python工具包。該工具包提供了一套完整的功能，用於從音頻文件中提取各種聲學特徵，包括MFCC、基頻（F0）、能量和過零率等。

## 功能特點

- MFCC特徵提取和穩定性評估
- 基頻（F0）提取和質量分析
- 能量特徵提取和評估
- 過零率計算和合理性檢查
- 特徵評估和質量分析
- 可視化工具支持

## 安裝

### 方法1：從PyPI安裝（推薦）

```bash
pip install audio-feature-extraction
```

### 方法2：從GitHub安裝

```bash
pip install git+https://github.com/chiy48308/audio_feature_extraction.git
```

### 方法3：從源碼安裝

```bash
git clone https://github.com/chiy48308/audio_feature_extraction.git
cd audio_feature_extraction
pip install -e .
```

## 使用方法

### 1. 基本使用

```python
from audio_feature_extraction import FeatureExtractor

# 初始化特徵提取器
extractor = FeatureExtractor()

# 處理音頻文件
result = extractor.process_audio('path/to/your/audio.wav')

# 獲取特徵
features = result['features']
mfcc = features['mfcc']
f0 = features['f0']
energy = features['energy']
zcr = features['zcr']

# 獲取評估結果
evaluation = result['evaluation']
print(f"MFCC穩定性：{evaluation['mfcc_stability']}")
print(f"F0缺失率：{evaluation['f0_missing_rate']}")
```

### 2. 特徵可視化

```python
import matplotlib.pyplot as plt

def plot_features(features):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 繪製MFCC
    axes[0, 0].imshow(features['mfcc'], aspect='auto', origin='lower')
    axes[0, 0].set_title('MFCC特徵')
    axes[0, 0].set_xlabel('幀')
    axes[0, 0].set_ylabel('MFCC係數')
    
    # 繪製F0
    axes[0, 1].plot(features['f0'])
    axes[0, 1].set_title('基頻（F0）曲線')
    axes[0, 1].set_xlabel('幀')
    axes[0, 1].set_ylabel('頻率 (Hz)')
    
    # 繪製能量
    axes[1, 0].plot(features['energy'])
    axes[1, 0].set_title('能量曲線')
    axes[1, 0].set_xlabel('幀')
    axes[1, 0].set_ylabel('能量')
    
    # 繪製ZCR
    axes[1, 1].plot(features['zcr'])
    axes[1, 1].set_title('過零率')
    axes[1, 1].set_xlabel('幀')
    axes[1, 1].set_ylabel('ZCR')
    
    plt.tight_layout()
    plt.show()

# 使用示例
result = extractor.process_audio('audio.wav')
plot_features(result['features'])
```

### 3. 批量處理

```python
import os
from tqdm import tqdm
import pandas as pd

def batch_process(audio_dir):
    extractor = FeatureExtractor()
    results = []
    
    for audio_file in tqdm(os.listdir(audio_dir)):
        if audio_file.endswith(('.wav', '.mp3', '.flac')):
            audio_path = os.path.join(audio_dir, audio_file)
            result = extractor.process_audio(audio_path)
            
            # 整理評估結果
            evaluation = result['evaluation']
            evaluation['file_name'] = audio_file
            results.append(evaluation)
    
    # 轉換為DataFrame
    df = pd.DataFrame(results)
    return df

# 使用示例
results_df = batch_process('audio_folder')
results_df.to_csv('evaluation_results.csv', index=False)
```

### 4. 自定義參數

```python
extractor = FeatureExtractor(
    sr=16000,                # 採樣率
    pre_emphasis=0.95,       # 預加重係數
    frame_length=0.030,      # 幀長（秒）
    frame_shift=0.015,       # 幀移（秒）
    n_mels=26,              # Mel濾波器數量
    n_mfcc=13,              # MFCC係數數量
    window='hamming',        # 窗函數類型
    smooth_window=7,         # 平滑窗口大小
    freq_smooth_window=5     # 頻域平滑窗口大小
)
```

### 5. 特徵評估指標說明

```python
# 獲取完整評估結果
evaluation = result['evaluation']

# MFCC穩定性（0-1，越高越好）
mfcc_stability = evaluation['mfcc_stability']

# F0提取質量
f0_missing_rate = evaluation['f0_missing_rate']  # 缺失率，越低越好
f0_quality = evaluation['f0_quality']  # 質量分數，越高越好

# 能量特徵穩定性（0-1，越高越好）
energy_stability = evaluation['energy_stability']

# 過零率合理性（0或1）
zcr_rationality = evaluation['zcr_rationality']

# 整體特徵完整性（0-1，越高越好）
feature_integrity = evaluation['feature_integrity']
```

## 特徵說明

### MFCC特徵
- 使用預加重和窗函數處理
- 應用Mel濾波器組
- 進行DCT變換
- 包含時域和頻域平滑

### F0特徵
- 使用自相關法提取
- 包含質量評估指標
- 支持缺失值處理

### 能量特徵
- 計算短時能量
- 提供歸一化選項
- 包含穩定性評估

### 過零率特徵
- 計算短時過零率
- 提供合理性檢查
- 支持特徵歸一化

## 評估指標

- MFCC穩定性：評估MFCC特徵的時域穩定性
- F0質量：評估基頻提取的可靠性
- 能量穩定性：評估能量特徵的變化程度
- 特徵完整性：評估所有特徵的提取質量

## 許可證

本項目採用MIT許可證。詳見[LICENSE](LICENSE)文件。 