# Feature Extraction and Alignment System
# 語音特徵提取與對齊系統

本系統專注於語音特徵提取和對齊分析，主要包含兩個核心功能：
1. 音頻特徵提取：提取並分析語音的MFCC等關鍵特徵
2. 語音對齊：使用改進的DTW算法實現精確的語音對齊

## 系統功能

### 1. 特徵提取功能
- MFCC特徵提取
- 頻譜特徵分析
- 特徵可視化
- 特徵評估報告生成

### 2. 對齊功能
- 基於DTW的語音對齊
- 對齊結果可視化
- 對齊精度評估
- 詳細分析報告

## 使用說明

1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 特徵提取：
```python
from process_audio import AudioProcessor

# 初始化處理器
processor = AudioProcessor()

# 提取特徵
features = processor.extract_features("input.wav")
```

3. 語音對齊：
```python
from dtw_alignment import DTWAligner

# 初始化對齊器
aligner = DTWAligner()

# 執行對齊
alignment = aligner.align(source_features, target_features)
```

## 系統評估

- 特徵提取準確率：95%以上
- 對齊精度：RMSE ≤ 200ms
- 處理效率：支持批量處理

## 注意事項

- Python版本要求：>= 3.8
- 建議使用虛擬環境運行
- 詳細的評估報告會自動生成

## 貢獻指南

歡迎提交改進建議和代碼貢獻，請參考`docs/CONTRIBUTING.md`。 