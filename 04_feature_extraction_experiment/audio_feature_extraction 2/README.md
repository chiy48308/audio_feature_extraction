# Audio Feature Extraction

音頻特徵提取工具包，用於提取和分析音頻特徵。

## 功能特點

- MFCC（梅爾頻率倒譜係數）特徵提取
- 基頻（F0）特徵提取
- 能量特徵提取
- 過零率（ZCR）特徵提取
- 特徵評估和質量分析

## 安裝方法

```bash
pip install git+https://github.com/chiy48308/audio_feature_extraction.git
```

## 使用方法

```python
from audio_feature_extraction import FeatureExtractor

# 初始化特徵提取器
extractor = FeatureExtractor()

# 處理單個音頻文件
result = extractor.process_audio("path/to/audio.wav")

# 獲取特徵
features = result['features']
evaluation = result['evaluation']

# 查看 MFCC 特徵
mfcc = features['mfcc']
mfcc_mean = features['mfcc_mean']
mfcc_std = features['mfcc_std']

# 查看其他特徵
f0 = features['f0']
energy = features['energy']
zcr = features['zcr']
```

## 特徵說明

### MFCC 特徵
- 使用改進的預處理和正規化方法
- 包含時域和頻域平滑處理
- 提供均值和標準差統計

### 基頻特徵
- 使用 PYIN 算法提取
- 包含基頻軌跡和變化率
- 提供插值處理未發聲區域

### 能量特徵
- 短時能量計算
- 能量包絡提取
- 提供穩定性評估

### 過零率特徵
- 計算信號過零率
- 提供合理性檢查

## 參數配置

可以在初始化時調整以下參數：

```python
extractor = FeatureExtractor(
    sr=16000,              # 採樣率
    pre_emphasis=0.95,     # 預加重係數
    frame_length=480,      # 幀長度
    frame_shift=160,       # 幀移
    n_mels=26,            # Mel 濾波器數量
    n_mfcc=13,            # MFCC 係數數量
    window='hamming',      # 窗函數類型
    smooth_window=7        # 平滑窗口大小
)
```

## 評估指標

特徵評估包含以下指標：

- MFCC 穩定性
- F0 品質評估
- 能量特徵穩定性
- ZCR 合理性
- 特徵完整性檢查

## 授權協議

MIT License 