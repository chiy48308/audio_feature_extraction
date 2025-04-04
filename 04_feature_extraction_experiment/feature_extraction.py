import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import logging
from pathlib import Path
import pandas as pd

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, output_dir: str = "features"):
        """初始化特徵提取器
        
        Args:
            output_dir: 特徵輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """載入音頻文件
        
        Args:
            audio_path: 音頻文件路徑
            
        Returns:
            (音頻數據, 採樣率)
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            return y, sr
        except Exception as e:
            logger.error(f"音頻載入失敗: {str(e)}")
            raise

    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """提取音頻特徵
        
        Args:
            audio_path: 音頻文件路徑
            
        Returns:
            特徵字典，包含以下特徵：
            - mfcc: MFCC特徵 (n_frames, 13)
            - f0: 基頻特徵 (n_frames, 1)
            - energy: 能量特徵 (n_frames, 1)
            - zcr: 過零率特徵 (n_frames, 1)
        """
        try:
            # 載入音頻
            y, sr = self.load_audio(audio_path)
            
            # 計算幀長和幀移（以秒為單位）
            frame_length = 0.025  # 25ms
            frame_shift = 0.010   # 10ms
            
            # 將秒轉換為採樣點數
            frame_length_samples = int(frame_length * sr)
            frame_shift_samples = int(frame_shift * sr)
            
            # 計算總幀數
            n_frames = 1 + (len(y) - frame_length_samples) // frame_shift_samples
            
            # 初始化特徵數組
            mfcc_features = np.zeros((n_frames, 13))
            f0_features = np.zeros((n_frames, 1))
            energy_features = np.zeros((n_frames, 1))
            zcr_features = np.zeros((n_frames, 1))
            
            # 逐幀提取特徵
            for i in range(n_frames):
                # 提取當前幀
                start = i * frame_shift_samples
                end = start + frame_length_samples
                frame = y[start:end]
                
                # 提取MFCC
                mfcc = librosa.feature.mfcc(
                    y=frame,
                    sr=sr,
                    n_mfcc=13,
                    n_fft=frame_length_samples,
                    hop_length=frame_length_samples
                )
                mfcc_features[i] = mfcc[:, 0]
                
                # 提取F0
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    frame,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr,
                    frame_length=frame_length_samples,
                    hop_length=frame_length_samples
                )
                f0_features[i] = f0[0] if not np.isnan(f0[0]) else 0
                
                # 提取能量
                energy = librosa.feature.rms(
                    y=frame,
                    frame_length=frame_length_samples,
                    hop_length=frame_length_samples
                )
                energy_features[i] = energy[0]
                
                # 提取過零率
                zcr = librosa.feature.zero_crossing_rate(
                    frame,
                    frame_length=frame_length_samples,
                    hop_length=frame_length_samples
                )
                zcr_features[i] = zcr[0]
            
            # 正規化特徵
            mfcc_features = (mfcc_features - np.mean(mfcc_features, axis=0)) / (np.std(mfcc_features, axis=0) + 1e-8)
            f0_features = (f0_features - np.mean(f0_features)) / (np.std(f0_features) + 1e-8)
            energy_features = (energy_features - np.mean(energy_features)) / (np.std(energy_features) + 1e-8)
            zcr_features = (zcr_features - np.mean(zcr_features)) / (np.std(zcr_features) + 1e-8)
            
            return {
                'mfcc': mfcc_features,
                'f0': f0_features,
                'energy': energy_features,
                'zcr': zcr_features
            }
            
        except Exception as e:
            logger.error(f"特徵提取失敗: {str(e)}")
            raise

    def visualize_features(self, features: Dict[str, np.ndarray], save_path: str = None):
        """可視化特徵
        
        Args:
            features: 特徵字典
            save_path: 可視化結果保存路徑
        """
        try:
            # 創建圖形
            fig, axes = plt.subplots(4, 1, figsize=(12, 8))
            fig.suptitle('音頻特徵可視化')
            
            # 繪製MFCC
            sns.heatmap(features['mfcc'].T, ax=axes[0], cmap='viridis')
            axes[0].set_title('MFCC')
            axes[0].set_xlabel('幀')
            axes[0].set_ylabel('係數')
            
            # 繪製F0
            axes[1].plot(features['f0'])
            axes[1].set_title('基頻 (F0)')
            axes[1].set_xlabel('幀')
            axes[1].set_ylabel('頻率 (Hz)')
            
            # 繪製能量
            axes[2].plot(features['energy'])
            axes[2].set_title('能量')
            axes[2].set_xlabel('幀')
            axes[2].set_ylabel('能量')
            
            # 繪製過零率
            axes[3].plot(features['zcr'])
            axes[3].set_title('過零率')
            axes[3].set_xlabel('幀')
            axes[3].set_ylabel('過零率')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"特徵可視化結果已保存至: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"特徵可視化失敗: {str(e)}")
            raise

    def evaluate_feature_quality(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """評估特徵質量
        
        Args:
            features: 特徵字典
            
        Returns:
            質量評估指標字典
        """
        try:
            quality_metrics = {}
            
            # 評估MFCC質量
            mfcc = features['mfcc']
            quality_metrics['mfcc_snr'] = self._calculate_snr(mfcc)
            quality_metrics['mfcc_stability'] = self._calculate_stability(mfcc)
            
            # 評估F0質量
            f0 = features['f0']
            quality_metrics['f0_continuity'] = self._calculate_continuity(f0)
            quality_metrics['f0_range'] = np.ptp(f0)
            
            # 評估能量質量
            energy = features['energy']
            quality_metrics['energy_snr'] = self._calculate_snr(energy)
            quality_metrics['energy_stability'] = self._calculate_stability(energy)
            
            # 評估過零率質量
            zcr = features['zcr']
            quality_metrics['zcr_snr'] = this._calculate_snr(zcr)
            quality_metrics['zcr_stability'] = this._calculate_stability(zcr)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"特徵質量評估失敗: {str(e)}")
            raise

    def _calculate_snr(self, feature: np.ndarray) -> float:
        """計算信噪比
        
        Args:
            feature: 特徵數組
            
        Returns:
            信噪比
        """
        signal = np.mean(feature, axis=0)
        noise = feature - signal
        snr = 10 * np.log10(np.sum(signal**2) / (np.sum(noise**2) + 1e-8))
        return snr

    def _calculate_stability(self, feature: np.ndarray) -> float:
        """計算特徵穩定性
        
        Args:
            feature: 特徵數組
            
        Returns:
            穩定性指標
        """
        diff = np.diff(feature, axis=0)
        stability = 1 / (1 + np.mean(np.abs(diff)))
        return stability

    def _calculate_continuity(self, feature: np.ndarray) -> float:
        """計算特徵連續性
        
        Args:
            feature: 特徵數組
            
        Returns:
            連續性指標
        """
        diff = np.diff(feature, axis=0)
        continuity = 1 / (1 + np.mean(np.abs(diff)))
        return continuity

    def save_results(self, features: Dict[str, np.ndarray], quality_metrics: Dict[str, float], 
                    audio_path: str, save_dir: str = None):
        """保存特徵提取結果
        
        Args:
            features: 特徵字典
            quality_metrics: 質量評估指標字典
            audio_path: 音頻文件路徑
            save_dir: 保存目錄
        """
        try:
            if save_dir is None:
                save_dir = self.output_dir
            else:
                save_dir = Path(save_dir)
                save_dir.mkdir(exist_ok=True)
            
            # 保存特徵
            audio_name = Path(audio_path).stem
            feature_path = save_dir / f"{audio_name}_features.npz"
            np.savez(feature_path, **features)
            
            # 保存質量評估結果
            quality_path = save_dir / f"{audio_name}_quality.csv"
            pd.DataFrame([quality_metrics]).to_csv(quality_path, index=False)
            
            # 保存可視化結果
            viz_path = save_dir / f"{audio_name}_visualization.png"
            self.visualize_features(features, str(viz_path))
            
            logger.info(f"特徵提取結果已保存至: {save_dir}")
            
        except Exception as e:
            logger.error(f"結果保存失敗: {str(e)}")
            raise 