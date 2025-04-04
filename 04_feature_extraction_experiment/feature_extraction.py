"""
特徵提取模組 - 最新版本 (v2.0.0)
包含改進的F0提取和特徵評估功能
"""

import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import webrtcvad
import pyloudnorm as pyln
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
import yaml
import warnings
warnings.filterwarnings('ignore')

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """特徵提取器 - 最新版本"""
    
    def __init__(self, config_path='config/experiment_config.yaml'):
        """初始化特徵提取器"""
        self.config = self._load_config(config_path)
        self.sample_rate = self.config['audio']['sample_rate']
        self.vad = webrtcvad.Vad(self.config['vad']['aggressiveness'])
        self.frame_duration = self.config['vad']['frame_duration']
        self.min_speech_duration = self.config['vad']['min_speech_duration']
        self.max_speech_duration = self.config['vad']['max_speech_duration']
        self.min_silence_duration = self.config['vad']['min_silence_duration']
        self.reference_level = self.config['volume']['reference_level']
        self.version = "2.0.0"  # 添加版本標記
        
    def _load_config(self, config_path):
        """載入配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def extract_features(self, audio_path):
        """提取所有特徵"""
        # 載入音頻
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # 預處理
        audio = self._preprocess_audio(audio)
        
        # 提取特徵
        features = {
            'mfcc': self.extract_mfcc(audio),
            'f0': self.extract_f0(audio),
            'energy': self.extract_energy(audio),
            'zcr': self.extract_zcr(audio)
        }
        
        # 評估特徵質量
        quality_metrics = self.evaluate_features(features)
        
        return features, quality_metrics
        
    def _preprocess_audio(self, audio):
        """音頻預處理"""
        # 降噪
        audio = nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=0.95
        )
        
        # VAD
        audio = self._apply_vad(audio)
        
        # 音量正規化
        audio = self._normalize_volume(audio)
        
        return audio
        
    def _apply_vad(self, audio):
        """應用VAD"""
        frame_size = int(self.frame_duration * self.sample_rate / 1000)
        frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=frame_size)
        
        speech_frames = []
        for frame in frames.T:
            is_speech = self.vad.is_speech(frame.astype(np.int16).tobytes(), self.sample_rate)
            speech_frames.extend([1 if is_speech else 0] * frame_size)
            
        speech_frames = np.array(speech_frames[:len(audio)])
        return audio * speech_frames
        
    def _normalize_volume(self, audio):
        """音量正規化"""
        meter = pyln.Meter(self.sample_rate)
        loudness = meter.integrated_loudness(audio)
        return pyln.normalize.loudness(audio, loudness, self.reference_level)
        
    def extract_mfcc(self, audio):
        """提取MFCC特徵"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        return mfcc
        
    def extract_f0(self, audio):
        """提取F0特徵（改進版本）"""
        # 預處理
        audio = librosa.effects.preemphasis(audio)
        
        # 使用改進的PYIN算法
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            frame_length=2048,
            hop_length=512,
            center=True,
            pad_mode='reflect'
        )
        
        # 後處理
        f0 = self._post_process_f0(f0, voiced_flag)
        
        return f0
        
    def _post_process_f0(self, f0, voiced_flag):
        """F0後處理"""
        # 中值濾波
        f0 = signal.medfilt(f0, kernel_size=5)
        
        # Savitzky-Golay濾波
        f0 = signal.savgol_filter(f0, window_length=5, polyorder=2)
        
        # 只保留有聲音的部分
        f0[~voiced_flag] = 0
        
        return f0
        
    def extract_energy(self, audio):
        """提取能量特徵"""
        energy = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=512
        )
        return energy
        
    def extract_zcr(self, audio):
        """提取過零率特徵"""
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=2048,
            hop_length=512
        )
        return zcr
        
    def evaluate_features(self, features):
        """評估特徵質量"""
        metrics = {}
        
        # MFCC評估
        metrics['mfcc_snr'] = self._calculate_snr(features['mfcc'])
        metrics['mfcc_stability'] = self._calculate_stability(features['mfcc'])
        
        # F0評估
        metrics['f0_continuity'] = self._calculate_f0_continuity(features['f0'])
        metrics['f0_range'] = self._calculate_f0_range(features['f0'])
        
        # 能量評估
        metrics['energy_snr'] = self._calculate_snr(features['energy'])
        metrics['energy_stability'] = self._calculate_stability(features['energy'])
        
        # ZCR評估
        metrics['zcr_snr'] = self._calculate_snr(features['zcr'])
        metrics['zcr_stability'] = self._calculate_stability(features['zcr'])
        
        return metrics
        
    def _calculate_snr(self, feature):
        """計算信噪比"""
        signal = np.mean(feature, axis=1)
        noise = feature - signal[:, np.newaxis]
        return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
        
    def _calculate_stability(self, feature):
        """計算穩定性"""
        return 1 - np.std(feature) / np.mean(np.abs(feature))
        
    def _calculate_f0_continuity(self, f0):
        """計算F0連續性"""
        voiced = f0 > 0
        if np.sum(voiced) < 2:
            return 0
        return 1 - np.mean(np.abs(np.diff(f0[voiced])))
        
    def _calculate_f0_range(self, f0):
        """計算F0範圍"""
        voiced = f0 > 0
        if np.sum(voiced) < 2:
            return 0
        return np.log2(np.max(f0[voiced]) / np.min(f0[voiced]))
        
    def save_features(self, features, audio_path):
        """保存特徵"""
        # 創建保存目錄
        os.makedirs('features', exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = f'features/{timestamp}_{base_name}_processed_features.npz'
        
        # 保存特徵
        np.savez(save_path, **features)
        logger.info(f'特徵提取結果已保存至: {save_path}')
        
        # 保存可視化
        self._save_visualization(features, timestamp, base_name)
        
    def _save_visualization(self, features, timestamp, base_name):
        """保存特徵可視化"""
        plt.figure(figsize=(15, 10))
        
        # MFCC
        plt.subplot(4, 1, 1)
        sns.heatmap(features['mfcc'], cmap='viridis')
        plt.title('MFCC特徵')
        
        # F0
        plt.subplot(4, 1, 2)
        plt.plot(features['f0'])
        plt.title('F0特徵')
        
        # 能量
        plt.subplot(4, 1, 3)
        plt.plot(features['energy'].T)
        plt.title('能量特徵')
        
        # ZCR
        plt.subplot(4, 1, 4)
        plt.plot(features['zcr'].T)
        plt.title('過零率特徵')
        
        plt.tight_layout()
        save_path = f'features/{timestamp}_{base_name}_processed_visualization.png'
        plt.savefig(save_path)
        logger.info(f'特徵可視化結果已保存至: {save_path}')
        plt.close()

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
            quality_metrics['zcr_snr'] = self._calculate_snr(zcr)
            quality_metrics['zcr_stability'] = self._calculate_stability(zcr)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"特徵質量評估失敗: {str(e)}")
            raise

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