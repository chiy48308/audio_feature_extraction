import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Dict
import logging
import os
from scipy import stats

class AudioProcessor:
    def __init__(self, target_dBFS: float = -20):
        self.target_dBFS = target_dBFS
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """載入音訊檔案"""
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            return audio, sr
        except Exception as e:
            self.logger.error(f"載入音訊檔案失敗: {file_path}, 錯誤: {str(e)}")
            raise
    
    def calculate_rms_features(self, audio: np.ndarray) -> Dict[str, float]:
        """計算 RMS 特徵"""
        rms = librosa.feature.rms(y=audio)[0]
        cv = np.std(rms) / np.mean(rms) if np.mean(rms) != 0 else 0
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_cv': float(cv)
        }
    
    def rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """RMS 正規化"""
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio
        scalar = 10**(self.target_dBFS / 20) / rms
        return audio * scalar
    
    def process_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """處理音訊並返回正規化後的音訊和特徵"""
        # 計算原始特徵
        original_features = self.calculate_rms_features(audio)
        
        # 進行正規化
        normalized_audio = self.rms_normalize(audio)
        
        # 計算正規化後特徵
        normalized_features = self.calculate_rms_features(normalized_audio)
        
        return normalized_audio, {
            'original': original_features,
            'normalized': normalized_features
        }
    
    def save_audio(self, audio: np.ndarray, file_path: str, sr: int = 16000):
        """儲存音訊檔案"""
        try:
            sf.write(file_path, audio, sr)
            self.logger.info(f"已儲存音訊檔案: {file_path}")
        except Exception as e:
            self.logger.error(f"儲存音訊檔案失敗: {file_path}, 錯誤: {str(e)}")
            raise 