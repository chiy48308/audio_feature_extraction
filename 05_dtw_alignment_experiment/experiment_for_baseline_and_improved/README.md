# 音頻特徵提取方法對比實驗

本實驗旨在比較基準版本和改進版本的音頻特徵提取方法的效果差異。

## 實驗目的

1. 評估改進版本在特徵提取穩定性上的提升
2. 比較兩個版本在處理速度上的差異
3. 分析特徵質量的改進效果

## 實驗內容

### 基準版本特徵
- 基本的 librosa MFCC 提取
- 簡單的基頻檢測
- 基礎能量特徵計算

### 改進版本特徵
- 優化的預處理流程
- 增強的基頻檢測
- 多維度能量特徵
- 完整的特徵評估系統

## 數據集

- 教師音頻：`preprocess_teacher_audio/`
- 學生音頻：`preprocess_student_audio/`

## 評估指標

1. 特徵穩定性
   - 基頻標準差
   - MFCC係數變化
   - 能量特徵一致性

2. 處理效率
   - 平均處理時間
   - 內存使用情況
   - CPU使用率

3. 特徵質量
   - 信噪比
   - 特徵完整性
   - 異常值比例

## 實驗結果

實驗結果將保存在以下文件中：

1. JSON格式的詳細數據
   - `teacher_baseline_results.json`
   - `teacher_improved_results.json`
   - `student_baseline_results.json`
   - `student_improved_results.json`

2. 可視化結果
   - `teacher_comparison/comparison_plots.png`
   - `student_comparison/comparison_plots.png`

3. 分析報告
   - `teacher_comparison/comparison_report.json`
   - `student_comparison/comparison_report.json`

## 使用方法

1. 運行實驗：
   ```bash
   python feature_extraction_comparison.py
   ```

2. 查看結果：
   - 檢查生成的JSON文件了解詳細數據
   - 查看PNG圖表直觀比較結果
   - 閱讀分析報告了解具體改進效果

## 注意事項

1. 確保安裝了所需的依賴包：
   - librosa
   - numpy
   - matplotlib
   - pandas
   - audio_feature_extraction_toolkit

2. 實驗過程中的錯誤處理：
   - 所有錯誤都會被記錄
   - 異常數據會被標記
   - 處理失敗的文件會被單獨列出

3. 結果解釋：
   - 較低的標準差表示更好的穩定性
   - 處理時間的減少表示效率提升
   - 特徵完整性的提高表示質量改進 