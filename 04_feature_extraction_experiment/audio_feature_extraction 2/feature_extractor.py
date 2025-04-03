import numpy as np
import librosa
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt, savgol_filter

class AudioFeatureExtractor:
    def __init__(self, sr=16000, pre_emphasis=0.95, frame_length=0.030,
                 frame_shift=0.015, n_mels=26, n_mfcc=13, window='hamming',
                 smooth_window=7, freq_smooth_window=5):
        self.sr = sr
        self.pre_emphasis = pre_emphasis
        self.frame_length = int(sr * frame_length)
        self.frame_shift = int(sr * frame_shift)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.window = window
        self.smooth_window = smooth_window
        self.freq_smooth_window = freq_smooth_window
        
    def extract_mfcc(self, audio):
        # 預加重
        emphasized = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])
        
        # 分幀和加窗
        frames = librosa.util.frame(emphasized, frame_length=self.frame_length,
                                  hop_length=self.frame_shift)
        window = librosa.filters.get_window(self.window, self.frame_length, fftbins=True)
        frames = frames.T * window
        
        # 歸一化
        frames = frames / np.max(np.abs(frames))
        frames = np.clip(frames, -1.5, 1.5)
        
        # 計算功率譜
        mag_spec = np.abs(np.fft.rfft(frames, n=self.frame_length))**2
        
        # Mel濾波器組
        mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.frame_length,
                                      n_mels=self.n_mels, fmin=80, fmax=8000)
        mel_spec = np.dot(mel_basis, mag_spec.T)
        
        # 頻域平滑
        mel_spec = medfilt(mel_spec, kernel_size=(self.freq_smooth_window, 1))
        
        # 對數壓縮
        log_mel_spec = np.log10(mel_spec + 1e-6)
        
        # DCT變換
        mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=self.n_mfcc)
        
        # 時域平滑
        mfcc = savgol_filter(mfcc, self.smooth_window, 3, axis=1)
        
        # 計算統計量
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return mfcc, mfcc_mean, mfcc_std
    
    def evaluate_mfcc(self, mfcc):
        # 計算時域穩定性
        temporal_stability = np.mean(np.std(mfcc, axis=1))
        return temporal_stability
    
    def process_audio(self, audio_path):
        # 讀取音頻
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # 提取MFCC
        mfcc, mfcc_mean, mfcc_std = self.extract_mfcc(audio)
        
        # 評估MFCC
        stability = self.evaluate_mfcc(mfcc)
        
        return {
            'features': {
                'mfcc': mfcc,
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std
            },
            'evaluation': {
                'stability': stability
            }
        }

class FeatureExtractor(AudioFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def extract_pitch(self, audio):
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        return f0, voiced_flag, voiced_probs
    
    def extract_energy(self, audio):
        energy = librosa.feature.rms(y=audio, frame_length=self.frame_length,
                                   hop_length=self.frame_shift)
        return energy.squeeze()
    
    def extract_zcr(self, audio):
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.frame_shift
        )
        return zcr.squeeze()
    
    def evaluate_features(self, features):
        evaluation = {}
        
        # MFCC穩定性
        evaluation['mfcc_stability'] = self.evaluate_mfcc(features['mfcc'])
        
        # F0評估
        f0_missing = np.isnan(features['f0']).mean()
        evaluation['f0_missing_rate'] = f0_missing
        evaluation['f0_quality'] = 1 - f0_missing
        
        # 能量穩定性
        energy_std = np.std(features['energy'])
        evaluation['energy_stability'] = 1 / (1 + energy_std)
        
        # ZCR合理性
        zcr_mean = np.mean(features['zcr'])
        evaluation['zcr_rationality'] = 1 if 0 <= zcr_mean <= 1 else 0
        
        # 特徵完整性
        evaluation['feature_integrity'] = np.mean([
            evaluation['mfcc_stability'],
            evaluation['f0_quality'],
            evaluation['energy_stability'],
            evaluation['zcr_rationality']
        ])
        
        return evaluation
    
    def process_audio(self, audio_path):
        # 讀取音頻
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # 提取所有特徵
        mfcc, mfcc_mean, mfcc_std = self.extract_mfcc(audio)
        f0, voiced_flag, voiced_probs = self.extract_pitch(audio)
        energy = self.extract_energy(audio)
        zcr = self.extract_zcr(audio)
        
        # 整理特徵
        features = {
            'mfcc': mfcc,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'f0': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'energy': energy,
            'zcr': zcr
        }
        
        # 評估特徵
        evaluation = self.evaluate_features(features)
        
        return {
            'features': features,
            'evaluation': evaluation
        } 