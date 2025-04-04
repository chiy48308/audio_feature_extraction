import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import noisereduce as nr
from scipy import signal
from pydub import AudioSegment
import scipy.signal
import soundfile as sf

class AudioFeatureExtractor:
    def __init__(self):
        self.sr = 22050  # 採樣率
        self.n_mfcc = 20  # 增加MFCC特徵數量，從13增加到20
        self.frame_length = 2048  # 幀長度
        self.hop_length = 512  # 跳躍長度
        
    def load_audio(self, audio_path, sr=None):
        """使用 soundfile 讀取音頻文件
        
        Args:
            audio_path: 音頻文件路徑
            sr: 目標採樣率，如果為None則使用原始採樣率
            
        Returns:
            tuple: (音頻數據, 採樣率)
        """
        try:
            # 使用 soundfile 讀取音頻
            y, file_sr = sf.read(audio_path)
            
            # 如果是立體聲，轉換為單聲道
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            
            # 如果指定了採樣率且與文件不同，進行重採樣
            if sr is not None and sr != file_sr:
                y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
                return y, sr
            return y, file_sr
            
        except Exception as e:
            print(f"音頻讀取錯誤 {audio_path}: {str(e)}")
            return None, None
            
    def preprocess_audio(self, y):
        """音頻預處理：降噪和正規化"""
        try:
            # 正規化
            y = librosa.util.normalize(y)
            
            # 降噪 - 使用更強的預加重係數
            y = librosa.effects.preemphasis(y, coef=0.98)
            
            # 高通濾波（調整截止頻率）
            nyquist = self.sr / 2
            cutoff = 200 / nyquist  # 降低截止頻率，從300Hz降至200Hz
            b, a = scipy.signal.butter(5, cutoff, btype='high')  # 增加濾波器階數，從4增加到5
            y = scipy.signal.filtfilt(b, a, y)
            
            # 添加音量正規化
            y = librosa.util.normalize(y, norm=np.inf, axis=0)
            
            return y
        except Exception as e:
            print(f"音頻預處理錯誤: {str(e)}")
            return None
            
    def extract_mfcc(self, audio_path: str, n_mfcc: int = 13) -> dict:
        """提取 MFCC 特徵
        
        Args:
            audio_path: 音頻文件路徑
            n_mfcc: MFCC 係數數量
            
        Returns:
            MFCC 特徵字典
        """
        try:
            # 使用 soundfile 讀取音頻
            y, sr = self.load_audio(audio_path)
            if y is None or sr is None:
                return None
                
            # 確保音頻數據是有效的
            if len(y) == 0:
                raise ValueError("音頻文件為空")
            
            # 音頻預處理
            y = self.preprocess_audio(y)
            if y is None:
                return None
            
            # 提取 MFCC 特徵
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=512,
                window='hann',
                center=True
            )
            
            # 特徵後處理
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 合併特徵
            features = np.vstack([mfcc, delta, delta2])
            
            # 正規化
            features = (features - np.mean(features, axis=1, keepdims=True)) / \
                      (np.std(features, axis=1, keepdims=True) + 1e-8)
            
            # 計算統計特徵
            mfcc_mean = np.mean(features)
            mfcc_std = np.std(features)
            mfcc_cv = np.abs(mfcc_std / mfcc_mean) if mfcc_mean != 0 else float('inf')
            
            # 評估特徵穩定性
            mfcc_stability = mfcc_cv < 3.0
            mfcc_range_valid = -100 < mfcc_mean < 100
            mfcc_std_valid = 0 <= mfcc_std < 50
            
            # 計算特徵質量評分
            mfcc_quality_score = 1.0
            if not mfcc_stability:
                mfcc_quality_score -= 0.3
            if not mfcc_range_valid:
                mfcc_quality_score -= 0.3
            if not mfcc_std_valid:
                mfcc_quality_score -= 0.3
            mfcc_quality_score = max(0.0, mfcc_quality_score)
            
            return {
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'mfcc_cv': mfcc_cv,
                'mfcc_stability': mfcc_stability,
                'mfcc_range_valid': mfcc_range_valid,
                'mfcc_std_valid': mfcc_std_valid,
                'mfcc_quality_score': mfcc_quality_score
            }
            
        except Exception as e:
            print(f"提取 MFCC 特徵時發生錯誤: {str(e)}")
            return None
            
    def extract_f0(self, audio_path):
        """提取基頻F0特徵"""
        try:
            # 使用 soundfile 讀取音頻
            y, sr = self.load_audio(audio_path, sr=self.sr)
            if y is None or sr is None:
                return None
                
            # 預處理音頻
            y = self.preprocess_audio(y)
            if y is None:
                return None
                
            # 提取F0
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # 計算缺失率
            f0_missing_rate = np.sum(np.isnan(f0)) / len(f0)
            
            # 計算估計誤差（使用平滑後的F0作為參考）
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 5:  # 確保有足夠的數據進行平滑
                f0_smooth = savgol_filter(valid_f0, window_length=min(5, len(valid_f0)), polyorder=2)
                f0_rmse = np.sqrt(np.mean((valid_f0 - f0_smooth)**2))
            else:
                f0_rmse = 0
            
            # 評估F0特徵
            f0_accuracy = f0_missing_rate <= 0.05  # 缺失率 ≤ 5%
            f0_rmse_valid = f0_rmse <= 10  # RMSE ≤ 10 Hz
            
            # 添加F0質量評分
            f0_quality_score = 1.0
            if not f0_accuracy:
                f0_quality_score -= 0.5
            if not f0_rmse_valid:
                f0_quality_score -= 0.5
            f0_quality_score = max(0.0, f0_quality_score)
            
            return {
                'f0_missing_rate': f0_missing_rate,
                'f0_rmse': f0_rmse,
                'f0_accuracy': f0_accuracy,
                'f0_rmse_valid': f0_rmse_valid,
                'f0_quality': f0_accuracy and f0_rmse_valid,  # 總體質量評估
                'f0_quality_score': f0_quality_score
            }
        except Exception as e:
            print(f"F0提取錯誤 {audio_path}: {str(e)}")
            return None
            
    def extract_energy(self, audio_path):
        """提取能量特徵"""
        try:
            # 使用 soundfile 讀取音頻
            y, sr = self.load_audio(audio_path, sr=self.sr)
            if y is None or sr is None:
                return None
                
            # 預處理音頻
            y = self.preprocess_audio(y)
            if y is None:
                return None
                
            # 計算RMS能量
            energy = librosa.feature.rms(
                y=y,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # 計算統計值
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            energy_cv = energy_std / energy_mean if energy_mean != 0 else float('inf')
            
            # 計算信噪比
            noise_floor = np.percentile(energy, 10)  # 使用第10百分位作為噪聲基準
            snr = 20 * np.log10(energy_mean / noise_floor) if noise_floor > 0 else 0
            
            # 評估能量特徵
            energy_range_valid = (energy_mean >= 5.67e-03) and (energy_mean <= 2.62e+00)
            energy_stability = energy_cv <= 0.3  # 變異係數 ≤ 0.3
            energy_snr_valid = snr >= 20  # 信噪比 ≥ 20 dB
            
            # 添加能量質量評分
            energy_quality_score = 1.0
            if not energy_range_valid:
                energy_quality_score -= 0.3
            if not energy_stability:
                energy_quality_score -= 0.3
            if not energy_snr_valid:
                energy_quality_score -= 0.3
            energy_quality_score = max(0.0, energy_quality_score)
            
            return {
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'energy_cv': energy_cv,
                'energy_snr': snr,
                'energy_range_valid': energy_range_valid,
                'energy_stability': energy_stability,
                'energy_snr_valid': energy_snr_valid,
                'energy_quality_score': energy_quality_score
            }
        except Exception as e:
            print(f"能量特徵提取錯誤 {audio_path}: {str(e)}")
            return None
            
    def extract_zcr(self, audio_path):
        """提取過零率特徵"""
        try:
            # 使用 soundfile 讀取音頻
            y, sr = self.load_audio(audio_path, sr=self.sr)
            if y is None or sr is None:
                return None
                
            # 預處理音頻
            y = self.preprocess_audio(y)
            if y is None:
                return None
                
            # 計算過零率
            zcr = librosa.feature.zero_crossing_rate(
                y=y,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # 計算統計值
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            zcr_cv = zcr_std / zcr_mean if zcr_mean != 0 else float('inf')
            
            # 評估過零率特徵
            zcr_range_valid = (zcr_mean >= 0.034) and (zcr_mean <= 0.491)
            zcr_stability = zcr_cv <= 0.4  # 變異係數 ≤ 0.4
            
            # 添加過零率質量評分
            zcr_quality_score = 1.0
            if not zcr_range_valid:
                zcr_quality_score -= 0.5
            if not zcr_stability:
                zcr_quality_score -= 0.5
            zcr_quality_score = max(0.0, zcr_quality_score)
            
            return {
                'zcr_mean': zcr_mean,
                'zcr_std': zcr_std,
                'zcr_cv': zcr_cv,
                'zcr_range_valid': zcr_range_valid,
                'zcr_stability': zcr_stability,
                'zcr_quality_score': zcr_quality_score
            }
        except Exception as e:
            print(f"過零率特徵提取錯誤 {audio_path}: {str(e)}")
            return None
            
    def extract_spectral_features(self, y, sr):
        """提取頻譜特徵
        
        Args:
            y: 音頻信號
            sr: 採樣率
            
        Returns:
            頻譜特徵字典
        """
        try:
            # 計算頻譜質心
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # 計算頻譜帶寬
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # 計算頻譜滾降
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # 計算頻譜對比度
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 計算統計特徵
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_contrast_mean': np.mean(spectral_contrast),
                'spectral_contrast_std': np.std(spectral_contrast)
            }
            
            return features
        except Exception as e:
            print(f"頻譜特徵提取錯誤: {str(e)}")
            return None

    def extract_harmonic_features(self, y, sr):
        """提取諧波特徵
        
        Args:
            y: 音頻信號
            sr: 採樣率
            
        Returns:
            諧波特徵字典
        """
        try:
            # 計算諧波能量
            harmonic = librosa.effects.harmonic(y)
            harmonic_energy = np.sum(harmonic ** 2)
            
            # 計算諧波比例
            harmonic_ratio = harmonic_energy / (np.sum(y ** 2) + 1e-8)
            
            # 計算諧波頻率
            harmonic_freq = librosa.feature.spectral_centroid(y=harmonic, sr=sr)[0]
            
            features = {
                'harmonic_energy': harmonic_energy,
                'harmonic_ratio': harmonic_ratio,
                'harmonic_freq_mean': np.mean(harmonic_freq),
                'harmonic_freq_std': np.std(harmonic_freq)
            }
            
            return features
        except Exception as e:
            print(f"諧波特徵提取錯誤: {str(e)}")
            return None

    def extract_timbre_features(self, y, sr):
        """提取音色特徵
        
        Args:
            y: 音頻信號
            sr: 採樣率
            
        Returns:
            音色特徵字典
        """
        try:
            # 計算梅爾頻譜
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            
            # 計算色度特徵
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 計算音色特徵
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            features = {
                'mel_energy_mean': np.mean(mel_spec),
                'mel_energy_std': np.std(mel_spec),
                'chroma_mean': np.mean(chroma),
                'chroma_std': np.std(chroma),
                'mfcc_mean': np.mean(mfcc),
                'mfcc_std': np.std(mfcc)
            }
            
            return features
        except Exception as e:
            print(f"音色特徵提取錯誤: {str(e)}")
            return None

    def extract_rhythm_features(self, y, sr):
        """提取節奏特徵
        
        Args:
            y: 音頻信號
            sr: 採樣率
            
        Returns:
            節奏特徵字典
        """
        try:
            # 計算節奏強度
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # 計算節奏週期性
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # 計算節奏規律性
            rhythm_regularity = np.std(onset_env) / (np.mean(onset_env) + 1e-8)
            
            features = {
                'tempo': tempo,
                'rhythm_regularity': rhythm_regularity,
                'onset_strength_mean': np.mean(onset_env),
                'onset_strength_std': np.std(onset_env)
            }
            
            return features
        except Exception as e:
            print(f"節奏特徵提取錯誤: {str(e)}")
            return None

    def extract_all_features(self, audio_path):
        """提取所有特徵"""
        try:
            # 使用 soundfile 讀取音頻
            y, sr = self.load_audio(audio_path, sr=self.sr)
            if y is None or sr is None:
                return None
                
            # 預處理音頻
            y = self.preprocess_audio(y)
            if y is None:
                return None
            
            # 提取基本特徵
            features = {}
            
            # MFCC特徵
            mfcc_features = self.extract_mfcc(audio_path)
            if mfcc_features is not None:
                features.update(mfcc_features)
            
            # F0特徵
            f0_features = self.extract_f0(audio_path)
            if f0_features:
                features.update(f0_features)
            
            # 能量特徵
            energy_features = self.extract_energy(audio_path)
            if energy_features:
                features.update(energy_features)
            
            # ZCR特徵
            zcr_features = self.extract_zcr(audio_path)
            if zcr_features:
                features.update(zcr_features)
            
            # 頻譜特徵
            spectral_features = self.extract_spectral_features(y, sr)
            if spectral_features:
                features.update(spectral_features)
            
            # 諧波特徵
            harmonic_features = self.extract_harmonic_features(y, sr)
            if harmonic_features:
                features.update(harmonic_features)
            
            # 音色特徵
            timbre_features = self.extract_timbre_features(y, sr)
            if timbre_features:
                features.update(timbre_features)
            
            # 節奏特徵
            rhythm_features = self.extract_rhythm_features(y, sr)
            if rhythm_features:
                features.update(rhythm_features)
            
            # 添加檔案名稱
            features['filename'] = os.path.basename(audio_path)
            
            return features
            
        except Exception as e:
            print(f"特徵提取錯誤 {audio_path}: {str(e)}")
            return None

class FeatureExtractor:
    def __init__(self):
        self.extractor = AudioFeatureExtractor()
        
    def extract_all_features(self, audio_dir):
        """提取目錄中所有音頻文件的特徵"""
        results = []
        
        # 獲取所有WAV文件
        audio_files = list(Path(audio_dir).rglob('*.wav'))
        if not audio_files:
            print(f"\n在目錄 {audio_dir} 中未找到WAV文件")
            return None
            
        print(f"\n找到 {len(audio_files)} 個音頻文件")
        
        # 處理每個音頻文件
        for audio_file in tqdm(audio_files, desc="正在處理音頻文件"):
            features = self.extractor.extract_all_features(str(audio_file))
            if features:
                # 添加類別信息
                features['category'] = 'student' if 'student' in str(audio_file).lower() else 'teacher'
                results.append(features)
        
        return results
        
    def evaluate_features(self, features_list):
        """評估特徵提取結果"""
        if not features_list:
            return None
            
        df = pd.DataFrame(features_list)
        
        # 確保所有必要的列都存在
        required_columns = ['filename', 'category']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # 按類別分組計算統計信息
        stats = {}
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            
            # 計算每個特徵的統計信息
            feature_stats = {}
            for col in df.columns:
                if col not in ['filename', 'category']:
                    if col in category_df.columns:
                        feature_stats[f'{col}_mean'] = category_df[col].mean()
                        feature_stats[f'{col}_std'] = category_df[col].std()
                        feature_stats[f'{col}_min'] = category_df[col].min()
                        feature_stats[f'{col}_max'] = category_df[col].max()
            
            stats[category] = feature_stats
            
        return stats
        
    def save_results(self, features_list, stats, output_dir='feature_evaluation'):
        """保存特徵提取結果"""
        if not features_list or not stats:
            return
            
        # 創建輸出目錄
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存詳細結果
        df = pd.DataFrame(features_list)
        df.to_csv(output_dir / 'feature_evaluation_detailed.csv', index=False)
        
        # 保存統計摘要
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        stats_df.to_csv(output_dir / 'feature_evaluation_summary.csv')
        
        # 分別保存教師和學生的統計摘要
        if 'teacher' in stats:
            pd.DataFrame([stats['teacher']]).to_csv(output_dir / 'feature_evaluation_summary_teacher.csv')
        if 'student' in stats:
            pd.DataFrame([stats['student']]).to_csv(output_dir / 'feature_evaluation_summary_student.csv')
            
        # 保存baseline數據
        baseline_dir = Path('baseline')
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # 按特徵類型分組保存
        feature_groups = {
            'mfcc': ['mfcc_mean', 'mfcc_std', 'mfcc_cv', 'mfcc_stability', 'mfcc_range_valid', 'mfcc_std_valid'],
            'f0': ['f0_missing_rate', 'f0_rmse', 'f0_accuracy', 'f0_rmse_valid', 'f0_quality'],
            'energy': ['energy_mean', 'energy_std', 'energy_cv', 'energy_snr', 'energy_range_valid', 'energy_stability', 'energy_snr_valid'],
            'zcr': ['zcr_mean', 'zcr_std', 'zcr_cv', 'zcr_range_valid', 'zcr_stability']
        }
        
        for group_name, features in feature_groups.items():
            group_df = df[['filename', 'category'] + features]
            group_df.to_csv(baseline_dir / f'{group_name}_baseline.csv', index=False)
            
        print(f"\n結果已保存至：{output_dir}")
        print(f"Baseline數據已保存至：{baseline_dir}")

def main():
    # 創建特徵提取器
    extractor = FeatureExtractor()
    
    # 設置音頻目錄
    audio_dirs = ['organized_audio/teacher', 'organized_audio/student']
    
    all_features = []
    for audio_dir in audio_dirs:
        # 提取特徵
        features = extractor.extract_all_features(audio_dir)
        if features:
            all_features.extend(features)
    
    if all_features:
        # 評估特徵
        stats = extractor.evaluate_features(all_features)
        
        # 保存結果
        extractor.save_results(all_features, stats)
    else:
        print("未找到有效的特徵提取結果")

if __name__ == '__main__':
    main() 