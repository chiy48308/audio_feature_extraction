import numpy as np
import librosa
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

class AudioFeatureExtractor:
    """音頻特徵提取器類"""
    
    def __init__(self, 
                 sr: int = 22050,
                 frame_length: int = 1024,
                 hop_length: int = 256,
                 n_mfcc: int = 13,
                 f0_min: float = librosa.note_to_hz('C2'),
                 f0_max: float = librosa.note_to_hz('C7'),
                 pre_emphasis: float = 0.97):
        """
        初始化特徵提取器
        
        參數:
            sr: 採樣率
            frame_length: 幀長度
            hop_length: 跳躍長度
            n_mfcc: MFCC特徵數量
            f0_min: 最小基頻
            f0_max: 最大基頻
            pre_emphasis: 預加重係數
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.pre_emphasis = pre_emphasis
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        載入音頻文件
        
        參數:
            audio_path: 音頻文件路徑
            
        返回:
            (音頻數據, 採樣率)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            return y, sr
        except Exception as e:
            self.logger.error(f"載入音頻文件失敗: {str(e)}")
            raise
    
    def preprocess_audio(self, y: np.ndarray) -> np.ndarray:
        """
        音頻預處理
        
        參數:
            y: 音頻數據
            
        返回:
            預處理後的音頻數據
        """
        # 預加重
        y_pre = librosa.effects.preemphasis(y, coef=self.pre_emphasis)
        
        # 靜音去除
        y_trim, _ = librosa.effects.trim(y_pre, top_db=30)
        
        return y_trim
    
    def extract_f0(self, y: np.ndarray) -> Dict[str, Any]:
        """
        提取基頻特徵
        
        參數:
            y: 音頻數據
            
        返回:
            基頻特徵字典
        """
        # 提取基頻
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=self.f0_min,
            fmax=self.f0_max,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            sr=self.sr
        )
        
        # 計算基頻統計特徵
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            f0_mean = np.mean(f0_valid)
            f0_std = np.std(f0_valid)
            f0_missing_rate = np.sum(np.isnan(f0)) / len(f0)
            f0_quality = 1 - f0_missing_rate
        else:
            f0_mean = 0
            f0_std = 0
            f0_missing_rate = 1
            f0_quality = 0
            
        return {
            'f0_mean': float(f0_mean),
            'f0_std': float(f0_std),
            'f0_missing_rate': float(f0_missing_rate),
            'f0_quality': float(f0_quality)
        }
    
    def extract_mfcc(self, y: np.ndarray) -> Dict[str, Any]:
        """
        提取MFCC特徵
        
        參數:
            y: 音頻數據
            
        返回:
            MFCC特徵字典
        """
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            window='hamming'
        )
        
        # 計算delta特徵
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 計算統計特徵
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        
        return {
            'mfcc_mean': mfcc_mean.tolist(),
            'mfcc_std': mfcc_std.tolist(),
            'mfcc_delta_mean': mfcc_delta_mean.tolist(),
            'mfcc_delta2_mean': mfcc_delta2_mean.tolist()
        }
    
    def extract_energy(self, y: np.ndarray) -> Dict[str, Any]:
        """
        提取能量特徵
        
        參數:
            y: 音頻數據
            
        返回:
            能量特徵字典
        """
        # 計算RMS能量
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # 計算統計特徵
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_range = np.ptp(rms)
        
        return {
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'energy_range': float(energy_range)
        }
    
    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """
        提取所有特徵
        
        參數:
            audio_path: 音頻文件路徑
            
        返回:
            所有特徵的字典
        """
        try:
            # 載入並預處理音頻
            y, _ = self.load_audio(audio_path)
            y_processed = self.preprocess_audio(y)
            
            # 提取各類特徵
            f0_features = self.extract_f0(y_processed)
            mfcc_features = self.extract_mfcc(y_processed)
            energy_features = self.extract_energy(y_processed)
            
            # 合併所有特徵
            features = {
                'file_path': audio_path,
                **f0_features,
                **mfcc_features,
                **energy_features
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"特徵提取失敗: {str(e)}")
            raise
    
    def batch_process(self, audio_dir: str) -> List[Dict[str, Any]]:
        """
        批量處理音頻文件
        
        參數:
            audio_dir: 音頻文件目錄
            
        返回:
            特徵列表
        """
        results = []
        audio_dir = Path(audio_dir)
        
        for audio_file in audio_dir.glob('*.wav'):
            try:
                features = self.extract_features(str(audio_file))
                results.append(features)
                self.logger.info(f"成功處理文件: {audio_file.name}")
            except Exception as e:
                self.logger.error(f"處理文件 {audio_file.name} 失敗: {str(e)}")
                continue
        
        return results 