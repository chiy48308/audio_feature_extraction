import numpy as np
import librosa
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, savgol_filter

class AudioFeatureExtractor:
    def __init__(self):
        self.sr = 16000
        self.n_mfcc = 13
        self.n_mels = 40
        self.win_length = 400
        self.hop_length = 160
        self.pre_emphasis = 0.97
        self.smooth_window = 5
        
    def extract_mfcc(self, audio):
        emphasized_audio = librosa.effects.preemphasis(audio, coef=self.pre_emphasis)
        emphasized_audio = emphasized_audio / (np.max(np.abs(emphasized_audio)) + 1e-10)
        
        mel_spec = librosa.feature.melspectrogram(
            y=emphasized_audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=20,
            fmax=8000
        )
        
        log_mel_spec = np.log(mel_spec + 1e-9)
        
        mfcc = librosa.feature.mfcc(
            S=log_mel_spec,
            n_mfcc=self.n_mfcc,
            sr=self.sr
        )
        
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc_smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(self.smooth_window)/self.smooth_window, mode='same'),
            axis=1,
            arr=mfcc
        )
        
        mfcc_mean = np.mean(mfcc_smoothed, axis=1)
        mfcc_std = np.std(mfcc_smoothed, axis=1)
        mfcc_normalized = (mfcc_smoothed - mfcc_mean[:, np.newaxis]) / (mfcc_std[:, np.newaxis] + 1e-10)
        
        mfcc_normalized = np.clip(mfcc_normalized, -3, 3)
        
        return {
            'mfcc': mfcc_normalized,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std
        }
    
    def evaluate_mfcc(self, mfcc_features):
        mfcc = mfcc_features['mfcc']
        mean_vals = np.mean(mfcc, axis=1)
        std_vals = np.std(mfcc, axis=1)
        
        mean_stability = np.abs(mean_vals).mean() < 0.5
        std_stability = 0.5 < np.mean(std_vals) < 1.5
        
        has_nan = np.any(np.isnan(mfcc))
        
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
        audio, _ = librosa.load(audio_path, sr=self.sr)
        mfcc_features = self.extract_mfcc(audio)
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
        self.smooth_window = 7
        self.freq_smooth_window = 5
        
    def extract_all_features(self, audio):
        emphasized = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])
        
        frames = librosa.util.frame(emphasized, 
                                  frame_length=self.frame_length,
                                  hop_length=self.frame_shift)
        hamming = np.hamming(self.frame_length)
        windowed = frames.T * hamming
        
        mfcc_features = self.extract_mfcc(windowed)
        f0, f0_delta = self.extract_pitch(audio)
        energy_features = self.extract_energy(audio)
        
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
        frames_norm = np.zeros_like(windowed_frames)
        for i in range(windowed_frames.shape[0]):
            frame = windowed_frames[i]
            frame_mean = np.mean(frame)
            frame_std = np.std(frame)
            frames_norm[i] = (frame - frame_mean) / (frame_std + 1e-6)
            frames_norm[i] = np.tanh(frames_norm[i])
        
        spectrum = np.fft.rfft(frames_norm, n=self.frame_length)
        power_spectrum = np.abs(spectrum) ** 2
        
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.frame_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=8000,
            htk=True
        )
        
        mel_spectrum = np.dot(mel_basis, power_spectrum.T)
        
        mel_spectrum = np.apply_along_axis(
            lambda x: np.median(np.lib.stride_tricks.sliding_window_view(
                np.pad(x, (2, 2), mode='edge'), 5
            ), axis=1),
            axis=0,
            arr=mel_spectrum
        )
        
        gaussian_window = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2)
        gaussian_window = gaussian_window / np.sum(gaussian_window)
        mel_spectrum = np.apply_along_axis(
            lambda x: np.convolve(x, gaussian_window, mode='same'),
            axis=0,
            arr=mel_spectrum
        )
        
        log_mel_spectrum = np.log10(mel_spectrum + 1e-5)
        
        mfcc = librosa.feature.mfcc(
            S=log_mel_spectrum,
            n_mfcc=self.n_mfcc,
            lifter=self.lifter_param
        )
        
        mfcc_smoothed = np.zeros_like(mfcc)
        for i in range(mfcc.shape[0]):
            mfcc_median = np.median(np.lib.stride_tricks.sliding_window_view(
                np.pad(mfcc[i], (3, 3), mode='edge'), 7
            ), axis=1)
            
            window_length = min(7, len(mfcc_median))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                mfcc_smoothed[i] = savgol_filter(mfcc_median, window_length, 2)
            else:
                mfcc_smoothed[i] = mfcc_median
        
        mfcc_mean = np.mean(mfcc_smoothed, axis=1)
        mfcc_std = np.std(mfcc_smoothed, axis=1)
        
        q1 = np.percentile(mfcc_smoothed, 25, axis=1)
        q3 = np.percentile(mfcc_smoothed, 75, axis=1)
        iqr = q3 - q1
        mfcc_normalized = np.zeros_like(mfcc_smoothed)
        for i in range(mfcc_smoothed.shape[0]):
            mfcc_normalized[i] = (mfcc_smoothed[i] - q1[i]) / (iqr[i] + 1e-6)
        
        mfcc_normalized = np.clip(mfcc_normalized, -2, 2)
        
        return {
            'mfcc': mfcc_normalized,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std
        }
    
    def extract_pitch(self, audio):
        f0, voiced_flag, voiced_probs = librosa.pyin(audio,
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'),
                                                    sr=self.sr)
        
        f0_delta = np.zeros_like(f0)
        f0_delta[1:-1] = (f0[2:] - f0[:-2]) / 2
        
        t = np.arange(len(f0))
        voiced_indices = ~np.isnan(f0)
        if np.any(voiced_indices):
            cs = CubicSpline(t[voiced_indices], f0[voiced_indices])
            f0_interpolated = cs(t)
        else:
            f0_interpolated = f0
            
        return f0_interpolated, f0_delta
    
    def extract_energy(self, audio):
        frames = librosa.util.frame(audio, 
                                  frame_length=self.frame_length,
                                  hop_length=self.frame_shift)
        energy = np.sum(frames**2, axis=0)
        
        zcr = librosa.feature.zero_crossing_rate(audio,
                                               frame_length=self.frame_length,
                                               hop_length=self.frame_shift)
        
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        return {
            'energy': energy,
            'zcr': zcr[0],
            'envelope': envelope
        }
    
    def evaluate_features(self, features):
        results = {}
        
        mfcc = features['mfcc']
        mfcc_mean = features['mfcc_mean']
        mfcc_std = features['mfcc_std']
        
        mean_stable = np.all(np.abs(mfcc_mean) < 0.8)
        std_stable = np.all((mfcc_std > 0.2) & (mfcc_std < 2.0))
        
        results['mfcc_mean'] = mfcc_mean
        results['mfcc_std'] = mfcc_std
        results['mfcc_stability'] = mean_stable and std_stable
        
        f0 = features.get('f0', None)
        if f0 is not None:
            nan_rate = np.sum(np.isnan(f0)) / len(f0)
            results['f0_missing_rate'] = nan_rate
            results['f0_quality'] = nan_rate < 0.3
        else:
            results['f0_missing_rate'] = 1.0
            results['f0_quality'] = False
        
        energy = features.get('energy', None)
        if energy is not None:
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            results['energy_mean'] = float(energy_mean)
            results['energy_std'] = float(energy_std)
            results['energy_stability'] = energy_std < (2.0 * energy_mean)
        else:
            results['energy_mean'] = 0.0
            results['energy_std'] = 0.0
            results['energy_stability'] = False
        
        zcr = features.get('zcr', None)
        if zcr is not None:
            zcr_mean = np.mean(zcr)
            results['zcr_mean'] = float(zcr_mean)
            results['zcr_rationality'] = 0.0 <= zcr_mean <= 0.5
        else:
            results['zcr_mean'] = 0.0
            results['zcr_rationality'] = False
        
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
        audio, _ = librosa.load(audio_path, sr=self.sr)
        features = self.extract_all_features(audio)
        evaluation = self.evaluate_features(features)
        
        return {
            'features': features,
            'evaluation': evaluation
        } 