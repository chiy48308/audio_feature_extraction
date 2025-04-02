import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class AudioFeatureExtractor:
    def __init__(self):
        self.sr = 16000
        self.n_mfcc = 13
        self.n_mels = 40
        self.win_length = 400
        self.hop_length = 160
        self.pre_emphasis = 0.97
        self.smooth_window = 5  # 增加平滑窗口大小
        
    def extract_mfcc(self, audio):
        # 預處理
        emphasized_audio = librosa.effects.preemphasis(audio, coef=self.pre_emphasis)
        
        # 使用MinMaxScaler進行歸一化
        emphasized_audio = emphasized_audio / (np.max(np.abs(emphasized_audio)) + 1e-10)
        
        # 計算梅爾頻譜圖
        mel_spec = librosa.feature.melspectrogram(
            y=emphasized_audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=20,  # 降低最小頻率
            fmax=8000  # 提高最大頻率
        )
        
        # 對數壓縮
        log_mel_spec = np.log(mel_spec + 1e-9)
        
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            S=log_mel_spec,
            n_mfcc=self.n_mfcc,
            sr=self.sr
        )
        
        # 計算delta特徵
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 時間平滑
        mfcc_smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(self.smooth_window)/self.smooth_window, mode='same'),
            axis=1,
            arr=mfcc
        )
        
        # 特徵歸一化
        mfcc_mean = np.mean(mfcc_smoothed, axis=1)
        mfcc_std = np.std(mfcc_smoothed, axis=1)
        mfcc_normalized = (mfcc_smoothed - mfcc_mean[:, np.newaxis]) / (mfcc_std[:, np.newaxis] + 1e-10)
        
        # 限制極值
        mfcc_normalized = np.clip(mfcc_normalized, -3, 3)
        
        return {
            'mfcc': mfcc_normalized,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std
        }
    
    def evaluate_mfcc(self, mfcc_features):
        """評估MFCC特徵質量"""
        mfcc = mfcc_features['mfcc']
        
        # 計算統計指標
        mean_vals = np.mean(mfcc, axis=1)
        std_vals = np.std(mfcc, axis=1)
        
        # 評估穩定性（放寬標準）
        mean_stability = np.abs(mean_vals).mean() < 0.5  # 允許更大的偏差
        std_stability = 0.5 < np.mean(std_vals) < 1.5   # 放寬標準差範圍
        
        # 檢查是否有無效值
        has_nan = np.any(np.isnan(mfcc))
        
        # 計算額外的質量指標
        dynamic_range = np.max(mfcc) - np.min(mfcc)
        entropy = -np.sum(np.histogram(mfcc.flatten(), bins=50)[0] / len(mfcc.flatten()) * 
                         np.log2(np.histogram(mfcc.flatten(), bins=50)[0] / len(mfcc.flatten()) + 1e-6))
        
        return {
            'mean': mean_vals,
            'std': std_vals,
            'stability': mean_stability and std_stability,
            'has_nan': has_nan,
            'dynamic_range': dynamic_range,
            'entropy': entropy
        }
    
    def process_audio(self, audio_path):
        """處理音頻文件"""
        # 讀取音頻
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # 提取特徵
        mfcc_features = self.extract_mfcc(audio)
        
        # 評估特徵
        evaluation = self.evaluate_mfcc(mfcc_features)
        
        return {
            'features': mfcc_features,
            'evaluation': evaluation
        }

class FeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
        self.pre_emphasis = 0.95
        self.frame_length = int(0.030 * sr)
        self.frame_shift = int(0.015 * sr)
        self.n_mels = 26
        self.n_mfcc = 13
        self.window = 'hamming'
        self.lifter_param = 22
        self.smooth_window = 7  # 增加平滑窗口大小
        self.freq_smooth_window = 5  # 新增：頻率域平滑窗口大小
        
    def extract_all_features(self, audio):
        """提取所有特徵"""
        # 預加重
        emphasized = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])
        
        # 分幀和加窗
        frames = librosa.util.frame(emphasized, 
                                  frame_length=self.frame_length,
                                  hop_length=self.frame_shift)
        hamming = np.hamming(self.frame_length)
        windowed = frames.T * hamming
        
        # === MFCC 特徵 ===
        mfcc_features = self.extract_mfcc(windowed)
        
        # === 基頻特徵 ===
        f0, f0_delta = self.extract_pitch(audio)
        
        # === 能量特徵 ===
        energy_features = self.extract_energy(audio)
        
        # 計算 MFCC 統計量
        mfcc = mfcc_features['mfcc']
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return {
            'mfcc': mfcc,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'f0': f0,
            'f0_delta': f0_delta,
            'energy': energy_features['energy'],
            'zcr': energy_features['zcr'],
            'envelope': energy_features['envelope']
        }
    
    def extract_mfcc(self, windowed_frames):
        """提取 MFCC 特徵"""
        # 1. 增強的信號正規化
        frames_norm = np.zeros_like(windowed_frames)
        for i in range(windowed_frames.shape[0]):
            # 使用均值和標準差正規化
            frame = windowed_frames[i]
            frame_mean = np.mean(frame)
            frame_std = np.std(frame)
            frames_norm[i] = (frame - frame_mean) / (frame_std + 1e-6)
            # 使用 tanh 函數進行非線性壓縮
            frames_norm[i] = np.tanh(frames_norm[i])
        
        # 2. FFT 和功率譜計算
        spectrum = np.fft.rfft(frames_norm, n=self.frame_length)
        power_spectrum = np.abs(spectrum) ** 2
        
        # 3. 增強的 Mel 濾波器組
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.frame_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=8000,
            htk=True
        )
        
        # 4. 應用 Mel 濾波器並增強頻率域平滑
        mel_spectrum = np.dot(mel_basis, power_spectrum.T)
        
        # 4.1 中值濾波去除突變
        mel_spectrum = np.apply_along_axis(
            lambda x: np.median(np.lib.stride_tricks.sliding_window_view(
                np.pad(x, (2, 2), mode='edge'), 5
            ), axis=1),
            axis=0,
            arr=mel_spectrum
        )
        
        # 4.2 高斯平滑
        gaussian_window = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2)
        gaussian_window = gaussian_window / np.sum(gaussian_window)
        mel_spectrum = np.apply_along_axis(
            lambda x: np.convolve(x, gaussian_window, mode='same'),
            axis=0,
            arr=mel_spectrum
        )
        
        # 5. 改進的對數壓縮
        log_mel_spectrum = np.log10(mel_spectrum + 1e-5)
        
        # 6. DCT 轉換
        mfcc = librosa.feature.mfcc(
            S=log_mel_spectrum,
            n_mfcc=self.n_mfcc,
            lifter=self.lifter_param
        )
        
        # 7. 增強的時域平滑處理
        mfcc_smoothed = np.zeros_like(mfcc)
        for i in range(mfcc.shape[0]):
            # 7.1 中值濾波去除離群值
            mfcc_median = np.median(np.lib.stride_tricks.sliding_window_view(
                np.pad(mfcc[i], (3, 3), mode='edge'), 7
            ), axis=1)
            
            # 7.2 Savitzky-Golay 平滑
            window_length = min(7, len(mfcc_median))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                mfcc_smoothed[i] = savgol_filter(mfcc_median, window_length, 2)
            else:
                mfcc_smoothed[i] = mfcc_median
        
        # 8. 計算統計量並進行最終正規化
        mfcc_mean = np.mean(mfcc_smoothed, axis=1)
        mfcc_std = np.std(mfcc_smoothed, axis=1)
        
        # 8.1 使用 Robust Scaling
        q1 = np.percentile(mfcc_smoothed, 25, axis=1)
        q3 = np.percentile(mfcc_smoothed, 75, axis=1)
        iqr = q3 - q1
        mfcc_normalized = np.zeros_like(mfcc_smoothed)
        for i in range(mfcc_smoothed.shape[0]):
            mfcc_normalized[i] = (mfcc_smoothed[i] - q1[i]) / (iqr[i] + 1e-6)
        
        # 8.2 限制極值
        mfcc_normalized = np.clip(mfcc_normalized, -2, 2)
        
        return {
            'mfcc': mfcc_normalized,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std
        }
    
    def extract_pitch(self, audio):
        """提取基頻特徵"""
        # 使用 librosa 的 pyin 算法提取基頻
        f0, voiced_flag, voiced_probs = librosa.pyin(audio,
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'),
                                                    sr=self.sr)
        
        # 計算基頻變化率
        f0_delta = np.zeros_like(f0)
        f0_delta[1:-1] = (f0[2:] - f0[:-2]) / 2
        
        # 使用三次樣條插值處理未發聲區域
        t = np.arange(len(f0))
        voiced_indices = ~np.isnan(f0)
        if np.any(voiced_indices):
            cs = CubicSpline(t[voiced_indices], f0[voiced_indices])
            f0_interpolated = cs(t)
        else:
            f0_interpolated = f0
            
        return f0_interpolated, f0_delta
    
    def extract_energy(self, audio):
        """提取能量特徵"""
        # 短時能量
        frames = librosa.util.frame(audio, 
                                  frame_length=self.frame_length,
                                  hop_length=self.frame_shift)
        energy = np.sum(frames**2, axis=0)
        
        # 過零率
        zcr = librosa.feature.zero_crossing_rate(audio,
                                               frame_length=self.frame_length,
                                               hop_length=self.frame_shift)
        
        # 能量包絡
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        return {
            'energy': energy,
            'zcr': zcr[0],
            'envelope': envelope
        }
    
    def evaluate_features(self, features):
        """評估所有特徵"""
        results = {}
        
        # 評估MFCC特徵
        mfcc = features['mfcc']
        mfcc_mean = features['mfcc_mean']
        mfcc_std = features['mfcc_std']
        
        # 檢查MFCC穩定性
        mean_stable = np.all(np.abs(mfcc_mean) < 0.8)  # 放寬平均值限制
        std_stable = np.all((mfcc_std > 0.2) & (mfcc_std < 2.0))  # 放寬標準差範圍
        
        results['mfcc_mean'] = mfcc_mean
        results['mfcc_std'] = mfcc_std
        results['mfcc_stability'] = mean_stable and std_stable
        
        # 評估F0特徵
        f0 = features.get('f0', None)
        if f0 is not None:
            nan_rate = np.sum(np.isnan(f0)) / len(f0)
            results['f0_missing_rate'] = nan_rate
            results['f0_quality'] = nan_rate < 0.3  # 允許更高的缺失率
        else:
            results['f0_missing_rate'] = 1.0
            results['f0_quality'] = False
        
        # 評估能量特徵
        energy = features.get('energy', None)
        if energy is not None:
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            results['energy_mean'] = float(energy_mean)
            results['energy_std'] = float(energy_std)
            results['energy_stability'] = energy_std < (2.0 * energy_mean)  # 放寬能量穩定性標準
        else:
            results['energy_mean'] = 0.0
            results['energy_std'] = 0.0
            results['energy_stability'] = False
        
        # 評估ZCR特徵
        zcr = features.get('zcr', None)
        if zcr is not None:
            zcr_mean = np.mean(zcr)
            results['zcr_mean'] = float(zcr_mean)
            results['zcr_rationality'] = 0.0 <= zcr_mean <= 0.5  # 保持ZCR合理性標準
        else:
            results['zcr_mean'] = 0.0
            results['zcr_rationality'] = False
        
        # 檢查特徵完整性
        results['feature_integrity'] = True
        for feature_name, feature_data in features.items():
            if feature_data is None:
                results['feature_integrity'] = False
                break
            if isinstance(feature_data, np.ndarray) and (np.any(np.isinf(feature_data)) or np.any(np.isnan(feature_data))):
                results['feature_integrity'] = False
                break
        
        return results

    def process_audio(self, audio_path):
        """處理音頻文件"""
        # 讀取音頻
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # 提取特徵
        features = self.extract_all_features(audio)
        
        # 評估特徵
        evaluation = self.evaluate_features(features)
        
        return {
            'features': features,
            'evaluation': evaluation
        }

def main():
    """主函數"""
    # 設置音頻文件目錄
    audio_dir = Path('organized_audio')  # 使用已存在的音頻目錄
    
    # 初始化特徵提取器
    extractor = FeatureExtractor()
    
    # 遞歸搜索所有WAV文件
    audio_files = list(audio_dir.rglob('*.wav'))
    if not audio_files:
        print(f"錯誤：在目錄 {audio_dir} 及其子目錄中未找到任何WAV文件")
        return
    
    print(f"\n找到 {len(audio_files)} 個音頻文件待處理")
    
    # 初始化結果列表
    df_rows = []
    
    # 處理每個音頻文件
    for audio_file in tqdm(audio_files, desc='提取特徵'):
        print(f"\n處理檔案：{audio_file.name}")
        print(f"檔案路徑：{audio_file.relative_to(audio_dir)}")
        print(f"音檔長度：{librosa.get_duration(filename=str(audio_file)):.2f} 秒")
        
        try:
            # 提取特徵
            result = extractor.process_audio(audio_file)
            
            # 構建結果行
            row = {
                'file_name': str(audio_file.relative_to(audio_dir)),
                'category': audio_file.parent.name,  # 添加類別信息（student/teacher）
                # MFCC 相關指標
                'mfcc_mean': float(np.mean(result['features']['mfcc_mean'])),
                'mfcc_std': float(np.mean(result['features']['mfcc_std'])),
                'mfcc_stability': result['evaluation']['mfcc_stability'],
                # F0 相關指標
                'f0_missing_rate': result['evaluation']['f0_missing_rate'],
                'f0_quality': result['evaluation']['f0_quality'],
                # 能量相關指標
                'energy_mean': result['evaluation']['energy_mean'],
                'energy_std': result['evaluation']['energy_std'],
                'energy_stability': result['evaluation']['energy_stability'],
                # ZCR 相關指標
                'zcr_mean': result['evaluation']['zcr_mean'],
                'zcr_rationality': result['evaluation']['zcr_rationality'],
                # 特徵完整性指標
                'feature_integrity': result['evaluation']['feature_integrity']
            }
            df_rows.append(row)
        except Exception as e:
            print(f"處理檔案 {audio_file.name} 時發生錯誤：{str(e)}")
            continue
    
    if not df_rows:
        print("錯誤：沒有成功處理任何音頻文件")
        return
    
    df = pd.DataFrame(df_rows)
    
    # 儲存詳細結果
    output_dir = Path('feature_evaluation')
    output_dir.mkdir(exist_ok=True)
    
    # 保存詳細數據
    detail_output_path = output_dir / 'feature_evaluation_detailed.csv'
    df.to_csv(detail_output_path, index=False, encoding='utf-8')
    
    # 按類別計算統計摘要
    categories = df['category'].unique()
    for category in categories:
        category_df = df[df['category'] == category]
        print(f"\n{category} 類別統計摘要：")
        print("-" * 50)
        
        summary_data = {
            '指標': [
                '檔案數',
                'MFCC 平均值範圍',
                'MFCC 標準差範圍',
                'MFCC 特徵穩定率',
                'F0 缺失率平均值',
                'F0 品質合格率',
                '能量平均值範圍',
                '能量標準差範圍',
                '能量穩定率',
                'ZCR 平均值範圍',
                'ZCR 合理率',
                '特徵完整率'
            ],
            '數值': [
                len(category_df),
                f"{category_df['mfcc_mean'].min():.3f} ~ {category_df['mfcc_mean'].max():.3f}",
                f"{category_df['mfcc_std'].min():.3f} ~ {category_df['mfcc_std'].max():.3f}",
                f"{category_df['mfcc_stability'].mean():.2%}",
                f"{category_df['f0_missing_rate'].mean():.2%}",
                f"{category_df['f0_quality'].mean():.2%}",
                f"{category_df['energy_mean'].min():.2e} ~ {category_df['energy_mean'].max():.2e}",
                f"{category_df['energy_std'].min():.2e} ~ {category_df['energy_std'].max():.2e}",
                f"{category_df['energy_stability'].mean():.2%}",
                f"{category_df['zcr_mean'].min():.3f} ~ {category_df['zcr_mean'].max():.3f}",
                f"{category_df['zcr_rationality'].mean():.2%}",
                f"{category_df['feature_integrity'].mean():.2%}"
            ]
        }
        
        category_summary_df = pd.DataFrame(summary_data)
        category_summary_path = output_dir / f'feature_evaluation_summary_{category}.csv'
        category_summary_df.to_csv(category_summary_path, index=False, encoding='utf-8')
        
        # 輸出類別統計摘要
        for idx, row in category_summary_df.iterrows():
            print(f"{row['指標']}：{row['數值']}")
    
    print(f"\n詳細結果已保存至：{detail_output_path}")
    print(f"各類別統計摘要已保存至：{output_dir}")
    print("\n實驗完成！")

if __name__ == '__main__':
    main() 