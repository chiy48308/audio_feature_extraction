# 音頻特徵提取工具 (v2.0.0)

這是一個用於音頻特徵提取的Python工具，支持多種音頻特徵的提取和評估。

## 功能特點

- 支持多種音頻特徵提取：
  - MFCC (梅爾頻率倒譜係數)
  - F0 (基頻)
  - 能量特徵
  - 過零率 (ZCR)
- 音頻預處理：
  - 降噪處理
  - 語音活動檢測 (VAD)
  - 音量正規化
- 特徵質量評估：
  - 信噪比 (SNR)
  - 特徵穩定性
  - F0連續性
  - F0範圍
- 特徵可視化：
  - 特徵時序圖
  - 特徵分布圖
  - 熱力圖

## 安裝要求

```bash
pip install -r requirements.txt
```

主要依賴：
- librosa==0.11.0
- numpy==1.24.4
- scipy==1.10.1
- soundfile==0.13.1
- noisereduce==3.0.3
- webrtcvad==2.0.10
- pyloudnorm==0.1.1

## 使用方法

1. 基本使用：
```python
from feature_extraction import FeatureExtractor

# 初始化特徵提取器
extractor = FeatureExtractor()

# 提取特徵
features, quality_metrics = extractor.extract_features("audio.wav")

# 保存特徵
extractor.save_features(features, "audio.wav")
```

2. 配置文件：
```yaml
# config/experiment_config.yaml
audio:
  sample_rate: 16000
  channels: 1
  duration: null

vad:
  aggressiveness: 3
  frame_duration: 30
  min_speech_duration: 0.3
  max_speech_duration: 10.0
  min_silence_duration: 0.3

noise_reduction:
  method: "wiener"
  stationary: true
  prop_decrease: 0.95
  n_fft: 2048
  win_length: 2048
  hop_length: 512

volume:
  reference_level: -23.0
  window_size: 0.4
  hop_size: 0.1
```

## 特徵說明

1. MFCC特徵：
   - 13維MFCC係數
   - 使用2048點FFT
   - 512點幀移

2. F0特徵：
   - 使用改進的PYIN算法
   - 頻率範圍：C2-C7
   - 中值濾波和Savitzky-Golay濾波

3. 能量特徵：
   - RMS能量
   - 2048點窗口
   - 512點幀移

4. 過零率特徵：
   - 2048點窗口
   - 512點幀移

## 質量評估指標

1. 信噪比 (SNR)：
   - 評估特徵的清晰度
   - 值越高表示信號質量越好

2. 穩定性：
   - 評估特徵的穩定程度
   - 範圍：0-1，越接近1越穩定

3. F0連續性：
   - 評估F0曲線的平滑程度
   - 範圍：0-1，越接近1越平滑

4. F0範圍：
   - 評估F0的變化範圍
   - 使用對數尺度

## 更新日誌

### v2.0.0 (2024-02-24)
- 改進F0提取算法
- 添加特徵質量評估功能
- 更新相關套件版本
- 添加配置文件支持
- 改進特徵可視化

### v1.0.0 (2024-02-23)
- 初始版本發布
- 基本特徵提取功能
- 簡單的特徵可視化

## 注意事項

1. 音頻文件要求：
   - 採樣率：16kHz
   - 格式：WAV
   - 聲道：單聲道

2. 記憶體使用：
   - 建議至少4GB RAM
   - 長時間音頻可能需要更多記憶體

3. 性能優化：
   - 使用多進程處理大量文件
   - 可以調整配置文件中的參數

## 貢獻指南

歡迎提交Issue和Pull Request來幫助改進這個項目。

## 授權

MIT License 