# 音頻特徵提取工具包

這是一個專門用於音頻特徵提取的Python工具包，提供了優化的特徵提取方法和完整的評估功能。本工具包特別適用於語音分析、音樂信息檢索和聲音分類等應用場景。

## 功能特點

### 1. 基頻檢測優化
- **自適應採樣率調整**：根據音頻內容自動選擇最佳採樣率
- **預加重和靜音處理**：
  - 預加重濾波器（係數可調）
  - 智能靜音檢測和去除
- **多級後處理**：
  - 中值濾波平滑
  - 頻率範圍約束
  - 異常值檢測和修正

### 2. MFCC特徵提取
- **可配置參數**：
  - 梅爾濾波器組數量
  - 幀長度和重疊率
  - 窗函數類型
- **Delta特徵**：
  - 一階差分係數
  - 二階差分係數
- **統計特徵**：
  - 均值和標準差
  - 偏度和峰度
  - 時域變化特徵

### 3. 能量特徵分析
- **多維度能量計算**：
  - RMS能量
  - 短時能量
  - 頻帶能量分佈
- **時域特徵**：
  - 能量包絡
  - 能量變化率
  - 突變檢測

### 4. 評估系統
- **特徵質量評估**：
  - 完整性檢查
  - 穩定性分析
  - 信噪比評估
- **統計報告生成**：
  - JSON格式詳細報告
  - CSV格式摘要報告
  - 可視化圖表支持

## 安裝方法

### 使用pip安裝
```bash
pip install audio-feature-extraction-toolkit
```

### 從源代碼安裝
```bash
git clone https://github.com/chiy48308/audio_feature_extraction.git
cd audio_feature_extraction
pip install -e .
```

## 快速開始

### 基本使用
```python
from audio_feature_extraction_toolkit import AudioFeatureExtractor

# 創建特徵提取器
extractor = AudioFeatureExtractor(
    sr=22050,              # 採樣率
    frame_length=1024,     # 幀長度
    hop_length=256,        # 跳躍長度
    n_mfcc=13             # MFCC係數數量
)

# 提取單個文件的特徵
features = extractor.extract_features('path/to/audio.wav')

# 查看特徵
print("基頻特徵:", features['f0_mean'])
print("MFCC特徵:", features['mfcc_mean'])
print("能量特徵:", features['energy_mean'])
```

### 批量處理
```python
# 處理整個目錄
results = extractor.batch_process('path/to/audio/directory')

# 生成評估報告
from audio_feature_extraction_toolkit import FeatureEvaluator

evaluator = FeatureEvaluator()
report = evaluator.generate_evaluation_report(results)
```

## 參數配置

### AudioFeatureExtractor
| 參數 | 說明 | 默認值 | 建議範圍 |
|------|------|--------|----------|
| sr | 採樣率 | 22050 | 16000-44100 |
| frame_length | 幀長度 | 1024 | 512-2048 |
| hop_length | 跳躍長度 | 256 | frame_length/4 |
| n_mfcc | MFCC係數數量 | 13 | 13-40 |
| pre_emphasis | 預加重係數 | 0.97 | 0.95-0.99 |

### FeatureEvaluator
| 參數 | 說明 | 默認值 |
|------|------|--------|
| f0_quality_threshold | F0質量閾值 | 0.8 |
| mfcc_stability_threshold | MFCC穩定性閾值 | 0.5 |
| energy_stability_threshold | 能量穩定性閾值 | 0.1 |

## 性能指標

在標準測試集上的表現：
- 基頻檢測準確率：99.10%
- MFCC特徵穩定性：98.20%
- 能量特徵穩定性：94.59%
- 特徵提取速度：<0.1s/文件（標準音頻）

## 高級用法

### 自定義預處理
```python
import numpy as np

def custom_preprocess(y, sr):
    # 自定義預處理邏輯
    return processed_audio

extractor = AudioFeatureExtractor()
extractor.preprocess_audio = custom_preprocess
```

### 特徵組合
```python
# 提取組合特徵
features = extractor.extract_features(
    'audio.wav',
    features_to_extract=['f0', 'mfcc', 'energy']
)
```

## 常見問題

1. **Q: 如何處理不同採樣率的音頻？**
   A: 工具包會自動重採樣到指定採樣率，您也可以通過設置sr參數來修改。

2. **Q: 特徵提取失敗的常見原因？**
   A: 常見原因包括：
   - 文件格式不支持
   - 音頻質量太差
   - 系統內存不足

3. **Q: 如何優化處理速度？**
   A: 可以：
   - 減小幀長度
   - 增加跳躍長度
   - 使用批處理模式

## 貢獻指南

我們歡迎各種形式的貢獻，包括但不限於：
- 錯誤報告
- 功能建議
- 代碼貢獻
- 文檔改進

### 貢獻步驟
1. Fork 項目
2. 創建特性分支
3. 提交更改
4. 推送到分支
5. 創建 Pull Request

## 版本歷史

- v0.1.0 (2024-04-06)
  - 初始版本發布
  - 基本特徵提取功能
  - 評估系統實現

## 授權協議

本項目採用 MIT 授權協議。詳見 [LICENSE](LICENSE) 文件。

## 作者

- Chris Yi (chiy48308@gmail.com)

## 致謝

感謝以下開源項目的支持：
- librosa
- numpy
- scipy
- pandas 