# 音檔質量評估與降噪比較工具

這個工具用於評估音檔質量並比較不同降噪方法的效果。它可以處理大量音檔，應用頻譜減法和維也納濾波器兩種降噪方法，然後使用PESQ和STOI指標評估處理效果。

## 功能特點

- 遞迴搜尋目錄中的所有WAV音檔
- 自動匹配教師和學生錄音對
- 應用兩種經典降噪方法：
  - 頻譜減法 (Spectral Subtraction)
  - 維也納濾波器 (Wiener Filter)
- 使用多種指標評估音檔質量：
  - PESQ (感知評分)
  - STOI (短時目標智能度)
  - SNR (信噪比)
- 生成豐富的視覺化圖表：
  - 處理前後的指標對比圖
  - 不同降噪方法之間的指標比較
  - 頻譜圖和波形比較
- 生成詳細的評估報告和摘要

## 安裝

1. 確保已安裝Python 3.8或更高版本
2. 安裝所需依賴：

```bash
pip install -r requirements.txt
```

3. (可選) 安裝額外的評估庫以獲得更準確的結果：

```bash
pip install pypesq pystoi
```

## 使用方法

### 基本用法

```bash
python audio_quality_assessment.py [base_dir]
```

其中 `[base_dir]` 是包含音檔的基礎目錄路徑。如果不提供，將使用當前目錄。

### 目錄結構

程序期望以下目錄結構：

```
base_dir/
├── session_XXXXXXXX/
│   ├── student_recordings/
│   │   ├── LessonXX_Character_StudentXX_utteranceXX.wav
│   │   └── ...
│   └── teacher_recordings/
│       ├── LessonXX_Character_Teacher_utteranceXX.wav
│       └── ...
└── ...
```

### 輸出

程序將在 `audio_data_collection` 目錄中生成以下輸出：

```
audio_data_collection/
├── processed_audio/
│   ├── spectral_subtraction/  # 頻譜減法處理後的音檔
│   └── wiener_filter/         # 維也納濾波器處理後的音檔
└── results/
    ├── metrics_report.csv     # 詳細評估指標數據
    ├── summary_report.txt     # 文字摘要報告
    ├── spectrograms/          # 頻譜圖比較
    ├── waveforms/             # 波形比較
    └── visualizations/        # 其他視覺化圖表
```

## 模塊說明

- `audio_quality_assessment.py`: 主程序，協調整個評估流程
- `noise_reduction.py`: 實現降噪方法
- `visualization.py`: 生成視覺化圖表

## 評估指標

### PESQ (感知評分)
- 範圍：1.0-4.5
- 閾值：≥ 3.0 合格
- 說明：評估語音的感知質量

### STOI (短時目標智能度)
- 範圍：0-1
- 閾值：≥ 0.65 合格
- 說明：評估語音的可懂度

### SNR (信噪比)
- 單位：dB
- 閾值：≥ 20 dB 合格
- 說明：評估信號與噪音的比例

## 降噪方法

### 頻譜減法 (Spectral Subtraction)
頻譜減法是一種基於短時傅立葉變換(STFT)的降噪方法。它通過估計噪音頻譜，然後從原始信號頻譜中減去噪音頻譜來實現降噪。

### 維也納濾波器 (Wiener Filter)
維也納濾波器是一種基於信號與噪音的功率譜密度估計的降噪方法。它設計最小均方誤差濾波器，應用於頻域信號，然後逆變換回時域。

## 注意事項

- 處理大量音檔可能需要較長時間
- PESQ和STOI計算需要參考信號，程序會自動匹配教師錄音作為參考
- 如果未安裝pypesq或pystoi庫，程序會使用替代方法計算類似指標
