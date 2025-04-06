# API 文檔

## AudioFeatureExtractor

音頻特徵提取器的主類，提供了完整的音頻特徵提取功能。

### 初始化參數

```python
AudioFeatureExtractor(
    sr: int = 22050,
    frame_length: int = 1024,
    hop_length: int = 256,
    n_mfcc: int = 13,
    f0_min: float = librosa.note_to_hz('C2'),
    f0_max: float = librosa.note_to_hz('C7'),
    pre_emphasis: float = 0.97
)
```

#### 參數說明
- `sr`: 採樣率，默認為22050Hz
- `frame_length`: 幀長度，默認為1024樣本
- `hop_length`: 幀移動長度，默認為256樣本
- `n_mfcc`: MFCC係數數量，默認為13
- `f0_min`: 最小基頻，默認為C2音高（約65.4Hz）
- `f0_max`: 最大基頻，默認為C7音高（約2093Hz）
- `pre_emphasis`: 預加重係數，默認為0.97

### 主要方法

#### load_audio
```python
def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]
```
載入音頻文件。

**參數**
- `audio_path`: 音頻文件路徑

**返回**
- 音頻數據和採樣率的元組

#### preprocess_audio
```python
def preprocess_audio(self, y: np.ndarray) -> np.ndarray
```
對音頻數據進行預處理。

**參數**
- `y`: 音頻數據

**返回**
- 預處理後的音頻數據

#### extract_f0
```python
def extract_f0(self, y: np.ndarray) -> Dict[str, Any]
```
提取基頻特徵。

**參數**
- `y`: 音頻數據

**返回**
包含以下鍵的字典：
- `f0_mean`: 基頻均值
- `f0_std`: 基頻標準差
- `f0_missing_rate`: 基頻缺失率
- `f0_quality`: 基頻質量分數

#### extract_mfcc
```python
def extract_mfcc(self, y: np.ndarray) -> Dict[str, Any]
```
提取MFCC特徵。

**參數**
- `y`: 音頻數據

**返回**
包含以下鍵的字典：
- `mfcc_mean`: MFCC係數均值
- `mfcc_std`: MFCC係數標準差
- `mfcc_delta_mean`: Delta MFCC均值
- `mfcc_delta2_mean`: Delta2 MFCC均值

#### extract_energy
```python
def extract_energy(self, y: np.ndarray) -> Dict[str, Any]
```
提取能量特徵。

**參數**
- `y`: 音頻數據

**返回**
包含以下鍵的字典：
- `energy_mean`: 能量均值
- `energy_std`: 能量標準差
- `energy_range`: 能量範圍

#### extract_features
```python
def extract_features(self, audio_path: str) -> Dict[str, Any]
```
提取所有特徵。

**參數**
- `audio_path`: 音頻文件路徑

**返回**
包含所有特徵的字典

#### batch_process
```python
def batch_process(self, audio_dir: str) -> List[Dict[str, Any]]
```
批量處理音頻文件。

**參數**
- `audio_dir`: 音頻文件目錄

**返回**
特徵字典的列表

## FeatureEvaluator

特徵評估器類，用於評估提取的特徵質量。

### 初始化
```python
FeatureEvaluator()
```

### 主要方法

#### calculate_feature_statistics
```python
def calculate_feature_statistics(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]
```
計算特徵統計信息。

**參數**
- `features_list`: 特徵字典列表

**返回**
統計信息字典

#### evaluate_feature_quality
```python
def evaluate_feature_quality(self, features_list: List[Dict[str, Any]]) -> Dict[str, float]
```
評估特徵質量。

**參數**
- `features_list`: 特徵字典列表

**返回**
質量評估指標字典

#### generate_evaluation_report
```python
def generate_evaluation_report(
    self, 
    features_list: List[Dict[str, Any]], 
    output_dir: str = "feature_evaluation"
) -> Dict[str, Any]
```
生成評估報告。

**參數**
- `features_list`: 特徵字典列表
- `output_dir`: 輸出目錄

**返回**
評估報告字典

#### analyze_feature_distribution
```python
def analyze_feature_distribution(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]
```
分析特徵分佈。

**參數**
- `features_list`: 特徵字典列表

**返回**
分佈分析結果字典

## 使用示例

### 基本特徵提取
```python
from audio_feature_extraction_toolkit import AudioFeatureExtractor

# 創建提取器
extractor = AudioFeatureExtractor(sr=22050)

# 提取特徵
features = extractor.extract_features('audio.wav')

# 訪問特徵
print(f"基頻均值: {features['f0_mean']:.2f} Hz")
print(f"MFCC特徵: {features['mfcc_mean']}")
print(f"能量: {features['energy_mean']:.2f}")
```

### 特徵評估
```python
from audio_feature_extraction_toolkit import FeatureEvaluator

# 創建評估器
evaluator = FeatureEvaluator()

# 批量處理音頻
features_list = extractor.batch_process('audio_directory/')

# 生成評估報告
report = evaluator.generate_evaluation_report(features_list)

# 分析特徵分佈
distribution = evaluator.analyze_feature_distribution(features_list)
```

## 錯誤處理

所有方法都會在出錯時拋出異常，建議使用 try-except 進行錯誤處理：

```python
try:
    features = extractor.extract_features('audio.wav')
except Exception as e:
    print(f"特徵提取失敗: {str(e)}")
```

## 注意事項

1. 音頻格式支持
   - 支持的格式：WAV, MP3, FLAC, OGG
   - 建議使用WAV格式以獲得最佳性能

2. 內存使用
   - 批量處理時注意內存使用
   - 可以通過調整幀長度來控制內存使用

3. 性能優化
   - 使用較大的hop_length可以提高處理速度
   - 減少MFCC係數數量可以減少計算量

4. 特徵質量
   - 定期檢查特徵質量報告
   - 根據應用場景調整參數 