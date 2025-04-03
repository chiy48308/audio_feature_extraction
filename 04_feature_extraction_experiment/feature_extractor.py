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

class AudioFeatureExtractor:
    def __init__(self):
        self.sr = 22050  # 採樣率
        self.n_mfcc = 13  # MFCC特徵數量
        self.frame_length = 2048  # 幀長度
        self.hop_length = 512  # 跳躍長度
        
    def extract_mfcc(self, audio_path):
        """提取MFCC特徵"""
        try:
            # 讀取音頻
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # 提取MFCC特徵
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            
            # 計算統計值
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            return {
                'mfcc_mean': np.mean(mfcc_mean),
                'mfcc_std': np.mean(mfcc_std),
                'mfcc_stability': True
            }
        except Exception as e:
            print(f"MFCC提取錯誤 {audio_path}: {str(e)}")
            return None
            
    def extract_f0(self, audio_path):
        """提取基頻F0特徵"""
        try:
            # 讀取音頻
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # 提取F0
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # 計算缺失率
            f0_missing_rate = np.sum(np.isnan(f0)) / len(f0)
            
            return {
                'f0_missing_rate': f0_missing_rate,
                'f0_quality': f0_missing_rate < 0.5
            }
        except Exception as e:
            print(f"F0提取錯誤 {audio_path}: {str(e)}")
            return None
            
    def extract_energy(self, audio_path):
        """提取能量特徵"""
        try:
            # 讀取音頻
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # 計算RMS能量
            energy = librosa.feature.rms(
                y=y,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # 計算統計值
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            
            # 判斷能量穩定性
            energy_stability = energy_std < (energy_mean * 2)
            
            return {
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'energy_stability': energy_stability
            }
        except Exception as e:
            print(f"能量特徵提取錯誤 {audio_path}: {str(e)}")
            return None
            
    def extract_zcr(self, audio_path):
        """提取過零率特徵"""
        try:
            # 讀取音頻
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # 計算ZCR
            zcr = librosa.feature.zero_crossing_rate(
                y,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # 計算統計值
            zcr_mean = np.mean(zcr)
            
            # 判斷ZCR合理性
            zcr_rationality = 0 <= zcr_mean <= 0.5
            
            return {
                'zcr_mean': zcr_mean,
                'zcr_rationality': zcr_rationality
            }
        except Exception as e:
            print(f"ZCR提取錯誤 {audio_path}: {str(e)}")
            return None
            
    def process_audio(self, audio_path):
        """處理單個音頻文件"""
        try:
            # 提取所有特徵
            mfcc_features = self.extract_mfcc(audio_path)
            f0_features = self.extract_f0(audio_path)
            energy_features = self.extract_energy(audio_path)
            zcr_features = self.extract_zcr(audio_path)
            
            # 檢查是否所有特徵都成功提取
            if all([mfcc_features, f0_features, energy_features, zcr_features]):
                features = {
                    'file_name': os.path.basename(audio_path),
                    **mfcc_features,
                    **f0_features,
                    **energy_features,
                    **zcr_features,
                    'feature_integrity': True
                }
                return features
            else:
                return None
        except Exception as e:
            print(f"音頻處理錯誤 {audio_path}: {str(e)}")
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
            features = self.extractor.process_audio(str(audio_file))
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
        
        # 按類別分組計算統計信息
        stats = {}
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            
            stats[category] = {
                'file_count': len(category_df),
                'mfcc_mean_range': f"{category_df['mfcc_mean'].min():.3f} to {category_df['mfcc_mean'].max():.3f}",
                'mfcc_std_range': f"{category_df['mfcc_std'].min():.3f} to {category_df['mfcc_std'].max():.3f}",
                'mfcc_stability_rate': f"{(category_df['mfcc_stability'].mean() * 100):.2f}%",
                'f0_missing_rate_avg': f"{(category_df['f0_missing_rate'].mean() * 100):.2f}%",
                'f0_quality_rate': f"{(category_df['f0_quality'].mean() * 100):.2f}%",
                'energy_mean_range': f"{category_df['energy_mean'].min():.2e} to {category_df['energy_mean'].max():.2e}",
                'energy_std_range': f"{category_df['energy_std'].min():.2e} to {category_df['energy_std'].max():.2e}",
                'energy_stability_rate': f"{(category_df['energy_stability'].mean() * 100):.2f}%",
                'zcr_mean_range': f"{category_df['zcr_mean'].min():.3f} to {category_df['zcr_mean'].max():.3f}",
                'zcr_rationality_rate': f"{(category_df['zcr_rationality'].mean() * 100):.2f}%",
                'feature_integrity_rate': f"{(category_df['feature_integrity'].mean() * 100):.2f}%"
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
            
        print(f"\n結果已保存至：{output_dir}")

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