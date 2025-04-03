import unittest
import numpy as np
from audio_feature_extraction import FeatureExtractor
import os
import librosa

class TestFeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """設置測試環境"""
        cls.extractor = FeatureExtractor()
        
        # 創建測試音頻數據
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        cls.test_audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        cls.test_audio_path = "test_audio.wav"
        librosa.output.write_wav(cls.test_audio_path, cls.test_audio, sr)
    
    @classmethod
    def tearDownClass(cls):
        """清理測試環境"""
        if os.path.exists(cls.test_audio_path):
            os.remove(cls.test_audio_path)
    
    def test_mfcc_extraction(self):
        """測試MFCC特徵提取"""
        result = self.extractor.process_audio(self.test_audio_path)
        mfcc = result['features']['mfcc']
        
        self.assertIsNotNone(mfcc)
        self.assertEqual(mfcc.shape[1], 13)  # 檢查MFCC係數數量
        self.assertTrue(np.all(np.isfinite(mfcc)))  # 檢查是否有無效值
    
    def test_f0_extraction(self):
        """測試F0特徵提取"""
        result = self.extractor.process_audio(self.test_audio_path)
        f0 = result['features']['f0']
        
        self.assertIsNotNone(f0)
        self.assertTrue(np.all(np.isfinite(f0)))  # 檢查是否有無效值
        self.assertTrue(np.mean(f0) > 0)  # 檢查F0值是否合理
    
    def test_energy_extraction(self):
        """測試能量特徵提取"""
        result = self.extractor.process_audio(self.test_audio_path)
        energy = result['features']['energy']
        
        self.assertIsNotNone(energy)
        self.assertTrue(np.all(energy >= 0))  # 檢查能量是否非負
        self.assertTrue(np.all(np.isfinite(energy)))  # 檢查是否有無效值
    
    def test_zcr_extraction(self):
        """測試過零率特徵提取"""
        result = self.extractor.process_audio(self.test_audio_path)
        zcr = result['features']['zcr']
        
        self.assertIsNotNone(zcr)
        self.assertTrue(np.all((zcr >= 0) & (zcr <= 1)))  # 檢查ZCR是否在[0,1]範圍內
        self.assertTrue(np.all(np.isfinite(zcr)))  # 檢查是否有無效值
    
    def test_feature_evaluation(self):
        """測試特徵評估"""
        result = self.extractor.process_audio(self.test_audio_path)
        evaluation = result['evaluation']
        
        # 檢查評估指標是否存在且有效
        self.assertIn('mfcc_stability', evaluation)
        self.assertIn('f0_missing_rate', evaluation)
        self.assertIn('f0_quality', evaluation)
        self.assertIn('energy_stability', evaluation)
        self.assertIn('zcr_rationality', evaluation)
        self.assertIn('feature_integrity', evaluation)
        
        # 檢查評估值是否在合理範圍內
        self.assertTrue(0 <= evaluation['mfcc_stability'] <= 1)
        self.assertTrue(0 <= evaluation['f0_missing_rate'] <= 1)
        self.assertTrue(0 <= evaluation['energy_stability'] <= 1)
        self.assertTrue(0 <= evaluation['zcr_rationality'] <= 1)
        self.assertTrue(0 <= evaluation['feature_integrity'] <= 1)

if __name__ == '__main__':
    unittest.main() 