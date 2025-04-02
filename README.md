# 語音評分實驗專案

本專案旨在研究音量正規化對語音評分模型的影響。

## 專案結構

```
.
├── README.md
├── requirements.txt
└── src/
    ├── audio_processor.py    # 音訊處理模組
    ├── model_trainer.py      # 模型訓練模組
    ├── visualizer.py         # 視覺化模組
    └── main.py              # 主程式
```

## 環境設置

1. 創建虛擬環境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 資料準備

1. 將音訊檔案放在 `audio_data/` 目錄下
2. 準備評分數據檔案 `scores.xlsx`，包含以下欄位：
   - audio_filename: 音訊檔案名稱
   - score: 教師評分 (0-4分)

## 運行實驗

```bash
python src/main.py
```

## 實驗輸出

1. 正規化後的音訊檔案將保存在 `normalized_audio/` 目錄
2. 視覺化結果將保存在 `visualization_results/` 目錄：
   - cv_distribution.png: 音量變異係數分布比較
   - model_errors.png: 模型評估指標比較
   - cv_improvement.png: 音量變異係數改善率分布

## 注意事項

1. 音訊檔案格式要求：
   - 格式：WAV
   - 採樣率：16kHz
   - 位元深度：16-bit PCM

2. 評分數據格式要求：
   - Excel 檔案 (.xlsx)
   - 必須包含 audio_filename 和 score 欄位
   - score 範圍：0-4 分 