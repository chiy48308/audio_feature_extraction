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

class AudioFeatureExtractor:
    def __init__(self):
        self.sr = 22050  # 採樣率
        self.n_mfcc = 20  # 增加MFCC特徵數量，從13增加到20
        self.frame_length = 2048  # 幀長度
        self.hop_length = 512  # 跳躍長度
        
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
            
    def extract_mfcc(self, audio_path: str, n_mfcc: int = 13) -> np.ndarray:
        """提取 MFCC 特徵
        
        Args:
            audio_path: 音頻文件路徑
            n_mfcc: MFCC 係數數量
            
        Returns:
            MFCC 特徵矩陣
        """
        try:
            # 讀取音頻文件
            y, sr = librosa.load(audio_path, sr=None)
            
            # 確保音頻數據是有效的
            if len(y) == 0:
                raise ValueError("音頻文件為空")
            
            # 音頻預處理
            # 1. 正規化音量
            y = librosa.util.normalize(y)
            
            # 2. 去除靜音段
            y, _ = librosa.effects.trim(y, top_db=30)
            
            # 3. 應用預加重濾波
            y = librosa.effects.preemphasis(y)
            
            # 4. 應用窗函數
            y = y * np.hanning(len(y))
            
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
            # 1. 計算 delta 特徵
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 2. 合併特徵
            features = np.vstack([mfcc, delta, delta2])
            
            # 3. 正規化
            features = (features - np.mean(features, axis=1, keepdims=True)) / \
                      (np.std(features, axis=1, keepdims=True) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"提取 MFCC 特徵時發生錯誤: {str(e)}")
            return np.array([])
            
    def extract_f0(self, audio_path):
        """提取基頻F0特徵"""
        try:
            # 讀取音頻
            y, sr = librosa.load(audio_path, sr=self.sr)
            
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
            # 讀取音頻
            y, sr = librosa.load(audio_path, sr=self.sr)
            
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
            # 讀取音頻
            y, sr = librosa.load(audio_path, sr=self.sr)
            
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
            
    def extract_all_features(self, audio_path):
        """提取所有特徵"""
        features = {}
        
        # 提取各項特徵
        mfcc = self.extract_mfcc(audio_path)
        if mfcc is not None:
            features.update(mfcc)  # 直接更新MFCC特徵，而不是嵌套存儲
        f0_features = self.extract_f0(audio_path)
        if f0_features:
            features.update(f0_features)
        energy_features = self.extract_energy(audio_path)
        if energy_features:
            features.update(energy_features)
        zcr_features = self.extract_zcr(audio_path)
        if zcr_features:
            features.update(zcr_features)
            
        # 添加檔案名稱
        features['filename'] = os.path.basename(audio_path)
        
        # 計算總體特徵質量評分
        quality_scores = []
        if 'mfcc_quality_score' in features:
            quality_scores.append(features['mfcc_quality_score'])
        if 'f0_quality_score' in features:
            quality_scores.append(features['f0_quality_score'])
        if 'energy_quality_score' in features:
            quality_scores.append(features['energy_quality_score'])
        if 'zcr_quality_score' in features:
            quality_scores.append(features['zcr_quality_score'])
            
        if quality_scores:
            features['overall_quality_score'] = np.mean(quality_scores)
        else:
            features['overall_quality_score'] = 0.0
        
        return features

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
        required_columns = ['filename', 'category', 'mfcc_mean', 'mfcc_std', 'mfcc_cv', 
                           'mfcc_stability', 'mfcc_range_valid', 'mfcc_std_valid']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # 按類別分組計算統計信息
        stats = {}
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            stats[category] = {
                'mfcc_mean_range': f"{category_df['mfcc_mean'].min():.3f} to {category_df['mfcc_mean'].max():.3f}",
                'mfcc_std_range': f"{category_df['mfcc_std'].min():.3f} to {category_df['mfcc_std'].max():.3f}",
                'mfcc_cv_range': f"{category_df['mfcc_cv'].min():.3f} to {category_df['mfcc_cv'].max():.3f}",
                'stability_rate': f"{(category_df['mfcc_stability'].sum() / len(category_df) * 100):.1f}%",
                'range_valid_rate': f"{(category_df['mfcc_range_valid'].sum() / len(category_df) * 100):.1f}%",
                'std_valid_rate': f"{(category_df['mfcc_std_valid'].sum() / len(category_df) * 100):.1f}%"
            }
            
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