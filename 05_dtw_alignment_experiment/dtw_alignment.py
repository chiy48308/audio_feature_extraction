import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from tqdm import tqdm
import sys
import time
import psutil
import tracemalloc
import functools
import traceback
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import glob
import re
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter as medfilt
import pywt
import scipy.signal

# 設置更詳細的日誌格式
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('dtw_alignment_detailed.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 配置設置
CONFIG = {
    'min_confidence_threshold': 0.7,  # 最小信心閾值
    'max_time_difference': 1000,  # 最大時間差異（毫秒）
    'batch_size': 20,  # 增加批次大小
    'pause_between_batches': 1,  # 減少批次間暫停時間
    'num_workers': max(1, multiprocessing.cpu_count() - 1),  # CPU核心數
    'memory_limit': 0.8  # 最大記憶體使用率
}

def log_performance(func_name, start_time, start_memory):
    """記錄性能指標"""
    end_time = time.time()
    process = psutil.Process()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    memory_diff = end_memory - start_memory
    
    logging.debug(f"{func_name} 執行時間: {duration:.2f} 秒")
    logging.debug(f"{func_name} 內存使用: {memory_diff:.2f} MB")

class PerformanceMonitor:
    @staticmethod
    def log_time_and_memory(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            logging.debug(f"{func.__name__} 執行時間: {end_time - start_time:.2f} 秒")
            logging.debug(f"{func.__name__} 內存使用: {end_memory - start_memory:.2f} MB")
            
            return result
        return wrapper

class FastDTWAligner:
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.num_levels = 3  # 多尺度層級數
        self.reduction_ratio = 2  # 降採樣比例
        self.min_window_size = 25  # 最小窗口大小
        self.teacher_pattern = "*Teacher*processed_features.npz"
        self.student_pattern = "*Student*processed_features.npz"
        self.timeout = 5
        self.last_progress_time = time.time()
        
        # 基本參數設置
        self.slope_constraint = 1.2
        self.step_size_constraint = 1.1
        
        # 預處理參數
        self.smoothing_window = 35
        self.gaussian_sigma = 5.0
        self.noise_threshold = 0.02
        self.endpoint_threshold = 0.12
        
        # DTW約束參數
        self.max_deviation_ratio = 0.08
        self.local_constraint_weight = 3.5
        
        # 特徵權重
        self.feature_weights = {
            'mfcc': 1.2,
            'energy': 1.5,
            'pitch': 1.3,
            'delta': 0.8,
            'delta2': 0.6
        }
        
        # 多尺度參數
        self.num_levels = 5
        self.reduction_ratio = 1.6

    def _enhanced_denoising(self, features):
        """增強的降噪處理"""
        # 改進的信噪比估計
        def estimate_snr(signal):
            noise = signal - scipy.signal.medfilt(signal, kernel_size=5)
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100

        # 自適應小波閾值計算
        def adaptive_wavelet_threshold(coeffs, snr):
            threshold_scale = np.exp(-snr / 20)  # SNR越高，閾值越小
            mad = np.median(np.abs(coeffs - np.median(coeffs)))
            threshold = threshold_scale * mad * np.sqrt(2 * np.log(len(coeffs)))
            return threshold

        # 頻域濾波
        def spectral_filter(signal, fs=100):
            """增強的頻域濾波，使用自適應頻帶劃分"""
            n = len(signal)
            # 使用漢寧窗減少頻譜洩漏
            window = np.hanning(n)
            windowed_signal = signal * window
            
            # 計算頻譜
            freqs = np.fft.fftfreq(n, 1/fs)
            fft = np.fft.fft(windowed_signal)
            power_spectrum = np.abs(fft) ** 2
            
            # 計算信號能量分佈
            total_power = np.sum(power_spectrum)
            cumulative_power = np.cumsum(power_spectrum) / total_power
            
            # 自適應確定頻帶數量（基於信號特性）
            energy_threshold = 0.95  # 能量覆蓋閾值
            significant_freqs = np.where(cumulative_power > energy_threshold)[0][0]
            
            # 改進的頻帶數量自適應算法
            base_num_bands = 4
            max_num_bands = 12
            
            # 使用多維度特徵計算信號複雜度
            # 1. 頻譜熵
            spectral_entropy = -np.sum((power_spectrum / total_power) * np.log2(power_spectrum / total_power + 1e-10))
            normalized_entropy = spectral_entropy / np.log2(len(power_spectrum))
            
            # 2. 頻譜峰值數量
            peak_threshold = np.mean(power_spectrum) + np.std(power_spectrum)
            peaks = scipy.signal.find_peaks(power_spectrum, height=peak_threshold)[0]
            normalized_peaks = len(peaks) / (len(power_spectrum) / 4)  # 歸一化峰值數量
            
            # 3. 頻譜傾斜度
            freqs_normalized = freqs / (fs/2)
            spectral_slope = np.polyfit(freqs_normalized[1:], np.log(power_spectrum[1:] + 1e-10), 1)[0]
            normalized_slope = np.clip((spectral_slope + 20) / 40, 0, 1)  # 歸一化傾斜度
            
            # 綜合計算複雜度因子
            complexity_weights = [0.4, 0.3, 0.3]  # 各特徵權重
            complexity_features = [normalized_entropy, normalized_peaks, normalized_slope]
            complexity_factor = np.dot(complexity_weights, complexity_features)
            
            # 動態調整頻帶數量
            num_bands = int(base_num_bands + (max_num_bands - base_num_bands) * complexity_factor)
            num_bands = max(base_num_bands, min(max_num_bands, num_bands))
            
            # 改進的Mel尺度頻帶邊界計算
            max_freq = fs/2
            mel_max = hz2mel(max_freq)
            
            # 使用非線性分佈改善頻帶劃分
            alpha = 0.6  # 基礎非線性係數
            beta = 1.2   # 高頻密度調整係數
            gamma = 0.8  # 能量分佈權重
            
            # 計算能量密度分佈
            energy_density = power_spectrum / np.sum(power_spectrum)
            cumulative_energy = np.cumsum(energy_density)
            
            # 生成改進的頻帶邊界
            mel_points = np.zeros(num_bands + 1)
            for i in range(num_bands + 1):
                # 基礎非線性分佈
                base_pos = (i / num_bands) ** alpha
                
                # 根據能量分佈調整位置
                energy_weight = gamma * cumulative_energy[int(base_pos * len(cumulative_energy))]
                
                # 高頻段使用更密集的分佈
                if i > num_bands * 0.6:
                    base_pos = base_pos ** (1/beta)
                
                # 組合最終位置
                normalized_pos = base_pos * (1 - gamma) + energy_weight
                mel_points[i] = normalized_pos * mel_max
            
            freq_bands = mel2hz(mel_points)
            
            # 優化頻帶邊界
            min_bandwidth = 50  # Hz
            max_bandwidth = 2000  # Hz
            
            # 使用動態規劃優化頻帶寬度
            for i in range(1, len(freq_bands)):
                bandwidth = freq_bands[i] - freq_bands[i-1]
                
                if bandwidth < min_bandwidth:
                    # 根據能量分佈調整窄頻帶
                    center_freq = (freq_bands[i] + freq_bands[i-1]) / 2
                    energy_ratio = np.sum(power_spectrum[int(freq_bands[i-1]):int(freq_bands[i])]) / total_power
                    
                    if energy_ratio > 0.1:  # 重要頻帶
                        # 擴展到最小帶寬
                        freq_bands[i-1] = center_freq - min_bandwidth/2
                        freq_bands[i] = center_freq + min_bandwidth/2
                    else:  # 非重要頻帶
                        # 合併到相鄰頻帶
                        if i > 1:
                            freq_bands[i-1] = freq_bands[i-2]
                        freq_bands[i] = freq_bands[i-1] + min_bandwidth
                
                elif bandwidth > max_bandwidth:
                    # 分析大頻帶中的能量分佈
                    band_spectrum = power_spectrum[int(freq_bands[i-1]):int(freq_bands[i])]
                    band_energy = np.sum(band_spectrum)
                    
                    if band_energy / total_power > 0.2:  # 高能量頻帶
                        # 尋找最佳分割點
                        split_point = np.argmax(band_spectrum) + int(freq_bands[i-1])
                        split_freq = freqs[split_point]
                        
                        # 確保分割後的頻帶寬度合理
                        if (split_freq - freq_bands[i-1] >= min_bandwidth and
                            freq_bands[i] - split_freq >= min_bandwidth):
                            freq_bands[i-1] = split_freq
                    else:  # 低能量頻帶
                        # 使用最大帶寬限制
                        center_freq = (freq_bands[i] + freq_bands[i-1]) / 2
                        freq_bands[i-1] = center_freq - max_bandwidth/2
                        freq_bands[i] = center_freq + max_bandwidth/2
            
            # 確保頻帶邊界不超出有效範圍
            freq_bands = np.clip(freq_bands, 0, fs/2)
            
            # 計算每個頻帶的能量分佈
            band_energies = []
            band_snrs = []
            for i in range(len(freq_bands)-1):
                band_mask = (np.abs(freqs) >= freq_bands[i]) & (np.abs(freqs) < freq_bands[i+1])
                if np.any(band_mask):
                    band_power = power_spectrum[band_mask]
                    band_energy = np.sum(band_power)
                    band_energies.append(band_energy)
                    
                    # 計算頻帶SNR
                    band_noise = scipy.signal.medfilt(band_power, kernel_size=5)
                    band_noise_floor = np.median(band_noise)
                    band_snr = 10 * np.log10(np.mean(band_power) / (band_noise_floor + 1e-10))
                    band_snrs.append(band_snr)
            
            # 自適應閾值計算
            band_thresholds = []
            for i in range(len(freq_bands)-1):
                band_mask = (np.abs(freqs) >= freq_bands[i]) & (np.abs(freqs) < freq_bands[i+1])
                if np.any(band_mask):
                    # 計算頻帶統計特性
                    band_power = power_spectrum[band_mask]
                    band_mean = np.mean(band_power)
                    band_std = np.std(band_power)
                    
                    # 使用改進的SNR和能量權重
                    snr_weight = 1 / (1 + np.exp(-band_snrs[i]/10))  # Sigmoid映射
                    energy_weight = band_energies[i] / max(band_energies)
                    
                    # 綜合閾值計算
                    base_threshold = band_mean * 0.1 + band_std
                    adaptive_factor = 0.7 * snr_weight + 0.3 * energy_weight
                    threshold = base_threshold * (1 - 0.8 * adaptive_factor)
                    
                    band_thresholds.append((freq_bands[i], freq_bands[i+1], threshold))
            
            # 應用頻帶濾波和平滑過渡
            filtered_fft = np.zeros_like(fft, dtype=complex)
            transition_width = 2  # Hz
            
            for i, (start_freq, end_freq, threshold) in enumerate(band_thresholds):
                # 主頻帶遮罩
                main_mask = (np.abs(freqs) >= start_freq) & (np.abs(freqs) < end_freq)
                
                # 創建過渡區域
                if i > 0:  # 低頻過渡
                    trans_low = (np.abs(freqs) >= (start_freq - transition_width)) & (np.abs(freqs) < start_freq)
                    trans_weight_low = (np.abs(freqs[trans_low]) - (start_freq - transition_width)) / transition_width
                else:
                    trans_low = np.zeros_like(freqs, dtype=bool)
                    trans_weight_low = np.array([])
                
                if i < len(band_thresholds)-1:  # 高頻過渡
                    trans_high = (np.abs(freqs) >= end_freq) & (np.abs(freqs) < (end_freq + transition_width))
                    trans_weight_high = 1 - (np.abs(freqs[trans_high]) - end_freq) / transition_width
                else:
                    trans_high = np.zeros_like(freqs, dtype=bool)
                    trans_weight_high = np.array([])
                
                # 應用主頻帶濾波
                mask = power_spectrum > threshold
                filtered_fft[main_mask] = fft[main_mask] * mask[main_mask]
                
                # 應用過渡區域
                if len(trans_weight_low) > 0:
                    filtered_fft[trans_low] = fft[trans_low] * trans_weight_low[:, np.newaxis]
                if len(trans_weight_high) > 0:
                    filtered_fft[trans_high] = fft[trans_high] * trans_weight_high[:, np.newaxis]
            
            # 相位保持重建
            filtered_signal = np.fft.ifft(filtered_fft)
            
            # 去除窗口效應
            filtered_signal = np.real(filtered_signal) / window
            
            # 處理窗口邊緣
            edge_size = int(n * 0.1)
            filtered_signal[:edge_size] = signal[:edge_size]
            filtered_signal[-edge_size:] = signal[-edge_size:]
            
            return filtered_signal

        denoised_features = np.zeros_like(features)
        for i in range(features.shape[1]):
            signal = features[:, i]
            
            # 估計信噪比
            snr = estimate_snr(signal)
            
            # 自適應中值濾波窗口
            window_size = max(3, min(25, int(len(signal) * 0.05)))
            if window_size % 2 == 0:
                window_size += 1
            
            # 應用中值濾波
            median_filtered = scipy.signal.medfilt(signal, kernel_size=window_size)
            
            # 小波降噪
            wavelet = 'db4'
            level = min(5, pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len))
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            
            # 對每個小波係數應用自適應閾值
            threshold = adaptive_wavelet_threshold(coeffs[0], snr)
            coeffs_thresholded = [pywt.threshold(c, threshold * (0.8 ** i), mode='soft') 
                                for i, c in enumerate(coeffs)]
            
            # 重建信號
            wavelet_denoised = pywt.waverec(coeffs_thresholded, wavelet)
            if len(wavelet_denoised) > len(signal):
                wavelet_denoised = wavelet_denoised[:len(signal)]
            
            # 頻域濾波
            spectral_denoised = spectral_filter(wavelet_denoised)
            
            # 組合不同的降噪結果
            snr_weight = 1 / (1 + np.exp(-snr/10))  # Sigmoid函數映射SNR到權重
            denoised_features[:, i] = (
                snr_weight * spectral_denoised + 
                (1 - snr_weight) * median_filtered
            )
        
        return denoised_features

    def _robust_normalization(self, features):
        """改進的特徵標準化方法"""
        normalized = np.zeros_like(features)
        
        for i in range(features.shape[1]):
            # 計算穩健的統計量
            q1, median, q3 = np.percentile(features[:, i], [25, 50, 75])
            iqr = q3 - q1
            
            # 識別和處理異常值
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data = np.clip(features[:, i], lower_bound, upper_bound)
            
            # 應用改進的標準化
            if iqr > 1e-10:  # 避免除以零
                normalized[:, i] = (data - median) / iqr
            else:
                normalized[:, i] = data - median
                
            # 限制標準化後的範圍
            normalized[:, i] = np.clip(normalized[:, i], -3, 3)
            
        return normalized

    def _apply_smoothing(self, features, window_size=5):
        """應用平滑處理
        
        Args:
            features: 輸入特徵
            window_size: 平滑窗口大小
        Returns:
            平滑後的特徵
        """
        # 計算能量分佈
        energy = np.sum(features ** 2, axis=1)
        
        # 自適應窗口大小
        local_variance = np.var(energy, keepdims=True)
        adaptive_window = np.maximum(3, np.minimum(9, int(window_size * (1 + local_variance))))
        
        # 高斯平滑核
        gaussian_kernel = np.exp(-np.linspace(-2, 2, adaptive_window)**2)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        
        # 應用平滑
        smoothed_features = np.zeros_like(features)
        pad_width = adaptive_window // 2
        
        for i in range(features.shape[1]):
            padded = np.pad(features[:, i], (pad_width, pad_width), mode='edge')
            smoothed_features[:, i] = np.convolve(padded, gaussian_kernel, mode='valid')
        
        return smoothed_features

    def _optimize_band_transitions(self, features, num_bands=8):
        """優化頻帶過渡區域
        
        Args:
            features: 輸入特徵
            num_bands: 頻帶數量
        Returns:
            優化後的特徵
        """
        # 計算頻帶邊界
        band_boundaries = np.linspace(0, features.shape[1], num_bands + 1).astype(int)
        optimized_features = features.copy()
        
        # 過渡區域寬度
        transition_width = 3
        
        for i in range(1, len(band_boundaries) - 1):
            start_idx = max(0, band_boundaries[i] - transition_width)
            end_idx = min(features.shape[1], band_boundaries[i] + transition_width)
            
            # 計算過渡權重
            weights = np.cos(np.linspace(-np.pi/2, np.pi/2, end_idx - start_idx)) * 0.5 + 0.5
            
            # 應用平滑過渡
            left_band = features[:, start_idx:end_idx]
            right_band = features[:, start_idx:end_idx]
            optimized_features[:, start_idx:end_idx] = (
                left_band * weights + right_band * (1 - weights)
            )
        
        return optimized_features

    def _adjust_band_boundaries(self, features, energy_threshold=0.1):
        """調整頻帶邊界
        
        Args:
            features: 輸入特徵
            energy_threshold: 能量閾值
        Returns:
            調整後的特徵
        """
        # 計算累積能量
        energy = np.sum(features ** 2, axis=1)
        cumulative_energy = np.cumsum(energy)
        normalized_energy = cumulative_energy / cumulative_energy[-1]
        
        # 找到能量突變點
        energy_diff = np.diff(normalized_energy)
        boundary_indices = np.where(energy_diff > energy_threshold)[0]
        
        # 應用邊界調整
        adjusted_features = features.copy()
        for idx in boundary_indices:
            start_idx = max(0, idx - 2)
            end_idx = min(features.shape[0], idx + 3)
            
            # 計算局部平均
            local_mean = np.mean(features[start_idx:end_idx], axis=0)
            
            # 平滑邊界過渡
            transition_weights = np.cos(np.linspace(-np.pi/2, np.pi/2, end_idx - start_idx)) * 0.5 + 0.5
            transition_weights = transition_weights.reshape(-1, 1)
            
            adjusted_features[start_idx:end_idx] = (
                features[start_idx:end_idx] * transition_weights +
                local_mean * (1 - transition_weights)
            )
        
        return adjusted_features

    def process_features(self, features):
        """特徵處理主函數
        
        Args:
            features: 輸入特徵
        Returns:
            處理後的特徵
        """
        # 應用平滑處理
        smoothed = self._apply_smoothing(features)
        
        # 優化頻帶過渡
        band_optimized = self._optimize_band_transitions(smoothed)
        
        # 調整頻帶邊界
        boundary_adjusted = self._adjust_band_boundaries(band_optimized)
        
        return boundary_adjusted

    def compute_distance_matrix(self, x, y):
        """增強的距離矩陣計算"""
        # 初始化距離矩陣
        n, m = len(x), len(y)
        D = np.zeros((n, m))
        
        # 計算基礎距離
        for i in range(n):
            for j in range(m):
                # 計算多種距離度量
                euclidean_dist = np.sqrt(np.sum((x[i] - y[j])**2))
                cosine_sim = np.dot(x[i], y[j]) / (np.linalg.norm(x[i]) * np.linalg.norm(y[j]) + 1e-8)
                correlation = np.corrcoef(x[i], y[j])[0,1]
                
                # 動態權重計算
                w1 = 0.6  # 歐氏距離權重
                w2 = 0.25 * (1 + cosine_sim)  # 餘弦相似度權重（動態調整）
                w3 = 0.15 * (1 + abs(correlation))  # 相關係數權重（動態調整）
                
                # 組合距離
                D[i,j] = w1 * euclidean_dist - w2 * cosine_sim - w3 * correlation
        
        # 應用局部約束
        D = self._apply_local_constraints(D)
        
        return D

    def _apply_local_constraints(self, D):
        """增強的局部約束應用，特別處理大偏差"""
        n, m = D.shape
        constrained_D = D.copy()
        
        # 計算理想路徑和全局趨勢
        ideal_path = np.linspace(0, m-1, n)
        x = np.arange(n)
        z = np.polyfit(x, ideal_path, 2)
        global_trend = np.poly1d(z)
        
        # 計算全局統計量
        path_diffs = np.abs(np.arange(m) - global_trend(np.arange(n)))
        mean_diff = np.mean(path_diffs)
        std_diff = np.std(path_diffs)
        
        # 計算全局時間比例和偏差閾值
        time_ratio = m / n
        expected_ratio = 1.0
        ratio_diff = abs(time_ratio - expected_ratio)
        
        # 動態調整約束參數
        base_constraint_weight = 2.0
        max_constraint_weight = 5.0
        constraint_weight = base_constraint_weight + (max_constraint_weight - base_constraint_weight) * min(ratio_diff / 0.3, 1.0)
        
        # 計算偏差閾值
        deviation_threshold = mean_diff + 2 * std_diff
        severe_deviation_threshold = mean_diff + 3 * std_diff
        
        # 應用約束
        for i in range(n):
            for j in range(m):
                # 計算當前位置的偏差
                expected_j = global_trend(i)
                pos_diff = abs(j - expected_j)
                
                # 基礎位置懲罰
                pos_penalty = constraint_weight * (pos_diff / m)
                
                # 特殊處理大偏差
                if pos_diff > severe_deviation_threshold:
                    # 指數增長的懲罰
                    severity_factor = ((pos_diff - severe_deviation_threshold) / severe_deviation_threshold) ** 2
                    pos_penalty *= (1.0 + 2.0 * severity_factor)
                
                # 計算局部趨勢
                if i > 0 and j > 0:
                    local_slope = (j - global_trend(i-1)) / 1.0
                    ideal_slope = (expected_j - global_trend(i-1)) / 1.0
                    slope_diff = abs(local_slope - ideal_slope)
                    
                    # 動態調整斜率懲罰
                    slope_weight = 1.5 + 0.5 * (slope_diff / ideal_slope)
                    slope_penalty = constraint_weight * slope_diff * slope_weight
                    
                    # 特殊處理大斜率偏差
                    if slope_diff > 2.0:
                        slope_penalty *= (1.0 + (slope_diff - 2.0))
                else:
                    slope_penalty = 0
                
                # 計算速度約束
                if i > 1 and j > 1:
                    actual_speed = (j - global_trend(i-2)) / 2.0
                    ideal_speed = (expected_j - global_trend(i-2)) / 2.0
                    speed_diff = abs(actual_speed - ideal_speed)
                    
                    # 動態調整速度懲罰
                    speed_weight = 1.2 + 0.3 * (speed_diff / ideal_speed)
                    speed_penalty = constraint_weight * speed_diff * speed_weight
                    
                    # 特殊處理大速度偏差
                    if speed_diff > 1.5:
                        speed_penalty *= (1.0 + 0.8 * (speed_diff - 1.5))
                else:
                    speed_penalty = 0
                
                # 增強的邊界約束
                boundary_penalty = 0
                if i < n * 0.15 or i > n * 0.85:
                    boundary_weight = 2.0 * constraint_weight
                    boundary_penalty = boundary_weight * abs(j/m - i/n)
                    
                    # 特殊處理邊界區域的大偏差
                    if pos_diff > deviation_threshold:
                        boundary_penalty *= (1.0 + (pos_diff - deviation_threshold) / deviation_threshold)
                
                # 平滑性約束
                smoothness_penalty = 0
                if i > 0 and j > 0 and i < n-1 and j < m-1:
                    local_var = np.var(constrained_D[max(0, i-2):min(n, i+3), max(0, j-2):min(m, j+3)])
                    smoothness_weight = 0.3 * (1.0 + local_var / np.mean(constrained_D))
                    smoothness = abs(constrained_D[i+1,j] + constrained_D[i-1,j] - 2*constrained_D[i,j])
                    smoothness_penalty = smoothness_weight * smoothness
                
                # 組合所有懲罰項
                total_penalty = (
                    pos_penalty * 2.0 +  # 增加位置懲罰權重
                    slope_penalty * 1.5 +  # 增加斜率懲罰權重
                    speed_penalty * 1.2 +  # 增加速度懲罰權重
                    boundary_penalty * 1.8 +  # 增加邊界懲罰權重
                    smoothness_penalty  # 保持平滑性懲罰權重不變
                )
                
                # 應用懲罰
                constrained_D[i,j] += total_penalty
        
        return constrained_D

    def apply_multi_scale_dtw(self, x, y):
        """多尺度DTW對齊"""
        levels = self.num_levels
        ratio = self.reduction_ratio
        
        # 初始化最粗糙尺度
        current_x = x
        current_y = y
        path = None
        
        for level in range(levels - 1, -1, -1):
            # 計算當前尺度的DTW
            if path is None:
                # 第一次對齊使用完整DTW
                distance_matrix = self.compute_distance_matrix(current_x, current_y)
                distance, current_path = self.dtw_with_constraints(distance_matrix)
            else:
                # 使用前一層的路徑作為約束
                refined_radius = max(self.min_window_size, int(len(current_x) * 0.1))
                distance_matrix = self.compute_distance_matrix(current_x, current_y)
                distance, current_path = self.constrained_dtw(distance_matrix, path, refined_radius)
            
            # 為下一個尺度準備
            if level > 0:
                # 上採樣路徑
                path = self.upsample_path(current_path, ratio)
                # 準備下一層的特徵
                current_x = self.refine_features(x, level)
                current_y = self.refine_features(y, level)
            else:
                path = current_path
        
        return distance, path

    def refine_features(self, features, level):
        """特徵細化"""
        if level == 0:
            return features
        
        # 計算當前層級的降採樣率
        ratio = self.reduction_ratio ** level
        target_length = len(features) // ratio
        
        # 使用插值進行重採樣
        indices = np.linspace(0, len(features) - 1, target_length)
        refined_features = np.zeros((target_length, features.shape[1]))
        
        for i in range(features.shape[1]):
            refined_features[:, i] = np.interp(indices, np.arange(len(features)), features[:, i])
        
        return refined_features

    def upsample_path(self, path, ratio):
        """路徑上採樣"""
        upsampled_path = []
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            # 在兩點之間插入點
            for j in range(ratio):
                interp_x = current[0] + (next_point[0] - current[0]) * j / ratio
                interp_y = current[1] + (next_point[1] - current[1]) * j / ratio
                upsampled_path.append((int(interp_x), int(interp_y)))
        
        upsampled_path.append(path[-1])
        return np.array(upsampled_path)

    def constrained_dtw(self, D, guide_path, radius):
        """基於引導路徑的約束DTW"""
        n, m = D.shape
        guide_path = np.array(guide_path)
        
        # 創建mask矩陣
        mask = np.full_like(D, np.inf)
        
        # 根據引導路徑設置有效區域
        for i, j in guide_path:
            i_start = max(0, i - radius)
            i_end = min(n, i + radius + 1)
            j_start = max(0, j - radius)
            j_end = min(m, j + radius + 1)
            mask[i_start:i_end, j_start:j_end] = 1
        
        # 應用mask
        D = D * mask
        
        # 計算累積距離矩陣
        C = np.full_like(D, np.inf)
        C[0, 0] = D[0, 0]
        
        for i in range(1, n):
            for j in range(1, m):
                if mask[i, j] != np.inf:
                    candidates = [
                        C[i-1, j-1],
                        C[i-1, j],
                        C[i, j-1]
                    ]
                    C[i, j] = D[i, j] + min(candidates)
        
        # 回溯找到最佳路徑
        path = [(n-1, m-1)]
        i, j = n-1, m-1
        
        while i > 0 or j > 0:
            candidates = []
            if i > 0 and j > 0:
                candidates.append((C[i-1, j-1], i-1, j-1))
            if i > 0:
                candidates.append((C[i-1, j], i-1, j))
            if j > 0:
                candidates.append((C[i, j-1], i, j-1))
            
            next_cost, i, j = min(candidates)
            path.append((i, j))
        
        path.reverse()
        return C[n-1, m-1], path

    def align_features(self, teacher_features, student_features):
        """對齊特徵主函數，增強時間對齊
        
        Args:
            teacher_features: 教師特徵
            student_features: 學生特徵
        Returns:
            對齊後的特徵和相關指標
        """
        # 特徵預處理
        teacher_processed = self.process_features(teacher_features)
        student_processed = self.process_features(student_features)
        
        # 特徵標準化
        teacher_features = self.normalize_features(teacher_processed, method='robust')
        student_features = self.normalize_features(student_processed, method='robust')
        
        # 計算全局時間比例和統計量
        time_ratio = len(student_features) / len(teacher_features)
        expected_ratio = 1.0
        ratio_diff = abs(time_ratio - expected_ratio)
        
        # 自適應窗口大小
        base_window = int(min(len(teacher_features), len(student_features)) * 0.1)
        window_size = max(25, min(base_window, 100))
        
        # 計算距離矩陣
        distance_matrix = self.compute_distance_matrix(teacher_features, student_features)
        
        # 增強的時間差異補償
        time_penalty = self._compute_enhanced_time_penalty(
            distance_matrix.shape,
            time_ratio,
            ratio_diff
        )
        
        # 組合距離矩陣
        adjusted_distance = distance_matrix + time_penalty
        
        # 應用增強的路徑約束
        constrained_distance = self._apply_enhanced_path_constraints(
            adjusted_distance,
            window_size,
            time_ratio
        )
        
        # 計算累積成本矩陣
        accumulated_cost = self._compute_accumulated_cost(constrained_distance)
        
        # 使用增強的路徑查找
        path = self._find_enhanced_path(
            accumulated_cost,
            time_ratio,
            teacher_features,
            student_features
        )
        
        # 計算最終距離
        distance = accumulated_cost[-1,-1]
        
        return distance, path

    def _compute_enhanced_time_penalty(self, shape, time_ratio, ratio_diff):
        """增強的時間差異補償計算"""
        n, m = shape
        time_penalty = np.zeros((n, m))
        
        # 基礎時間權重
        base_weight = 1.5
        
        # 根據全局比例差異調整權重
        ratio_weight = base_weight * (1 + ratio_diff)
        
        # 計算理想路徑
        ideal_path = np.linspace(0, m-1, n)
        
        # 為每個點計算時間懲罰
        for i in range(n):
            for j in range(m):
                # 計算當前點與理想路徑的偏差
                expected_j = ideal_path[i]
                deviation = abs(j - expected_j)
                
                # 計算局部時間比例
                local_ratio = (j + 1) / (i + 1)
                local_diff = abs(local_ratio - time_ratio)
                
                # 組合懲罰項
                time_penalty[i,j] = (
                    ratio_weight * local_diff +  # 時間比例懲罰
                    0.8 * deviation / m +        # 路徑偏差懲罰
                    0.5 * (local_diff ** 2)      # 非線性懲罰項
                )
        
        return time_penalty

    def _apply_enhanced_path_constraints(self, distance_matrix, window_size, time_ratio):
        """增強的路徑約束應用"""
        n, m = distance_matrix.shape
        constrained = np.full_like(distance_matrix, np.inf)
        
        # 計算理想路徑
        ideal_path = np.linspace(0, m-1, n)
        
        # 動態窗口大小
        adaptive_window = np.zeros(n)
        for i in range(n):
            # 根據位置調整窗口
            pos_factor = 1.0 - 0.3 * abs(2 * i/n - 1)  # 中間位置窗口較大
            adaptive_window[i] = max(window_size * pos_factor, self.min_window_size)
        
        # 應用自適應窗口約束
        for i in range(n):
            center = int(ideal_path[i])
            current_window = int(adaptive_window[i])
            start = max(0, center - current_window)
            end = min(m, center + current_window + 1)
            
            # 主要路徑區域
            constrained[i, start:end] = distance_matrix[i, start:end]
            
            # 平滑過渡區
            if start > 0:
                # 指數衰減過渡
                trans_width = min(10, start)
                transition = np.exp(-np.arange(trans_width)**2 / (2 * (trans_width/3)**2))
                trans_start = max(0, start - trans_width)
                constrained[i, trans_start:start] = (
                    distance_matrix[i, trans_start:start] * transition[:start-trans_start]
                )
            
            if end < m:
                # 指數衰減過渡
                trans_width = min(10, m - end)
                transition = np.exp(-np.arange(trans_width)**2 / (2 * (trans_width/3)**2))
                constrained[i, end:end+trans_width] = (
                    distance_matrix[i, end:end+trans_width] * transition
                )
        
        return constrained

    def _compute_accumulated_cost(self, distance_matrix):
        """計算增強的累積成本矩陣"""
        n, m = distance_matrix.shape
        accumulated_cost = np.zeros_like(distance_matrix)
        accumulated_cost[0,0] = distance_matrix[0,0]
        
        # 填充第一行和第一列，使用改進的邊界處理
        for i in range(1, n):
            accumulated_cost[i,0] = (
                accumulated_cost[i-1,0] + 
                distance_matrix[i,0] * (1 + 0.1 * i)  # 增加邊界懲罰
            )
        for j in range(1, m):
            accumulated_cost[0,j] = (
                accumulated_cost[0,j-1] + 
                distance_matrix[0,j] * (1 + 0.1 * j)  # 增加邊界懲罰
            )
        
        # 填充其餘部分，使用改進的路徑選擇
        for i in range(1, n):
            for j in range(1, m):
                # 計算三個方向的成本
                vertical = accumulated_cost[i-1,j]
                horizontal = accumulated_cost[i,j-1]
                diagonal = accumulated_cost[i-1,j-1]
                
                # 計算路徑偏好權重
                diag_weight = 0.8  # 偏好對角線移動
                vert_horz_weight = 1.2  # 垂直和水平移動的懲罰
                
                # 選擇最小成本路徑
                min_cost = min(
                    vertical * vert_horz_weight,
                    diagonal * diag_weight,
                    horizontal * vert_horz_weight
                )
                
                accumulated_cost[i,j] = distance_matrix[i,j] + min_cost
        
        return accumulated_cost

    def _find_enhanced_path(self, acc_cost_mat, time_ratio, teacher_features, student_features):
        """增強的路徑查找"""
        n, m = acc_cost_mat.shape
        path = [(n-1, m-1)]
        
        # 初始化路徑統計
        path_slopes = []
        path_speeds = []
        time_diffs = []
        
        # 動態調整參數
        slope_weight = 1.2
        speed_weight = 0.8
        time_diff_weight = 1.5  # 增加時間差異權重
        stability_weight = 0.9
        
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((0, j-1))
            elif j == 0:
                path.append((i-1, 0))
            else:
                # 計算候選點
                candidates = [(i-1, j), (i-1, j-1), (i, j-1)]
                costs = [acc_cost_mat[i-1, j], 
                        acc_cost_mat[i-1, j-1], 
                        acc_cost_mat[i, j-1]]
                
                # 計算時間差異
                current_ratio = j / i
                ratio_diff = abs(current_ratio - time_ratio)
                time_diff_penalty = ratio_diff * time_diff_weight
                
                # 計算路徑特性
                if len(path) >= 5:  # 使用更多的歷史點
                    # 計算最近5個點的統計量
                    recent_path = np.array(path[-5:])
                    slopes = np.diff(recent_path[:, 1]) / np.maximum(1e-6, np.diff(recent_path[:, 0]))
                    speeds = np.sqrt(np.diff(recent_path[:, 1])**2 + np.diff(recent_path[:, 0])**2)
                    
                    # 計算統計量
                    mean_slope = np.mean(slopes)
                    std_slope = np.std(slopes)
                    mean_speed = np.mean(speeds)
                    std_speed = np.std(speeds)
                    
                    # 穩定性指標
                    stability_score = 1.0 / (1.0 + std_slope + std_speed)
                    
                    # 為每個候選路徑添加約束
                    for idx, (ni, nj) in enumerate(candidates):
                        if ni != i:  # 非水平移動
                            # 斜率約束
                            candidate_slope = (nj - j) / (ni - i)
                            slope_diff = abs(candidate_slope - mean_slope)
                            slope_penalty = slope_weight * slope_diff * (1 + std_slope)
                            
                            # 速度約束
                            candidate_speed = np.sqrt((nj - j)**2 + (ni - i)**2)
                            speed_diff = abs(candidate_speed - mean_speed)
                            speed_penalty = speed_weight * speed_diff * (1 + std_speed)
                            
                            # 穩定性控制
                            stability_penalty = stability_weight * (1 - stability_score)
                            
                            # 組合懲罰項
                            total_penalty = (
                                slope_penalty * 1.2 +
                                speed_penalty * 0.8 +
                                time_diff_penalty * 1.5 +  # 增加時間差異權重
                                stability_penalty * 1.0
                            )
                            
                            costs[idx] += total_penalty
                
                # 選擇最佳路徑
                best_idx = np.argmin(costs)
                path.append(candidates[best_idx])
        
        # 路徑後處理
        path = path[::-1]
        path = np.array(path)
        
        # 自適應平滑
        if len(path) > 10:
            # 計算局部變化率
            diffs = np.diff(path, axis=0)
            local_vars = np.sqrt(np.sum(diffs**2, axis=1))
            
            # 自適應窗口大小
            window_sizes = np.clip(5 + 10 * (1 - local_vars/np.max(local_vars)), 3, 15).astype(int)
            
            # 平滑處理
            smoothed_path = np.copy(path)
            for i in range(5, len(path)-5):
                window_size = window_sizes[i-1]
                if window_size % 2 == 0:
                    window_size += 1
                
                # 使用加權中值濾波
                window = path[max(0, i-window_size//2):min(len(path), i+window_size//2+1)]
                weights = np.exp(-0.5 * np.arange(-window_size//2, window_size//2+1)**2 / (window_size/4)**2)
                weights = weights[:len(window)]
                weights /= weights.sum()
                
                # 應用平滑
                smoothed_path[i] = np.average(window, axis=0, weights=weights)
            
            path = smoothed_path
        
        # 確保端點約束
        path[0] = [0, 0]
        path[-1] = [n-1, m-1]
        
        # 確保單調性
        path[:,0] = np.maximum.accumulate(path[:,0])
        path[:,1] = np.maximum.accumulate(path[:,1])
        
        return path.tolist()

    def evaluate_alignment(self, path, student_features, teacher_features):
        """評估對齊結果
        
        評估面向：
        1. 對齊誤差範圍：RMSE ≤ 200 毫秒
        2. 對齊一致性：每句偏差 < 250ms 且無過早/過晚切點
        3. 語音-文本對應率：對應準確率 ≥ 95%
        """
        path_np = np.array(path)
        
        # 1. 計算RMSE（毫秒）
        time_diffs = (path_np[:, 0] - path_np[:, 1]) * 10  # 轉換為毫秒
        rmse = np.sqrt(np.mean(time_diffs ** 2))
        
        # 2. 對齊一致性評估
        max_deviation = np.max(np.abs(time_diffs))
        early_cuts = np.sum(time_diffs < -250)  # 過早切點數
        late_cuts = np.sum(time_diffs > 250)    # 過晚切點數
        alignment_consistency = (early_cuts == 0) and (late_cuts == 0)
        
        # 3. 語音-文本對應率
        correct_alignments = np.sum(np.abs(time_diffs) < 250)  # 250ms內視為正確對應
        correspondence_rate = (correct_alignments / len(path)) * 100
        
        # 計算分段統計
        segments = np.array_split(time_diffs, 10)  # 將路徑分成10段
        segment_stats = []
        for i, segment in enumerate(segments):
            segment_stats.append({
                'segment_id': i + 1,
                'mean_diff': float(np.mean(segment)),
                'std_diff': float(np.std(segment)),
                'max_diff': float(np.max(np.abs(segment)))
            })
        
        # 評估結果
        evaluation = {
            'rmse': float(rmse),
            'max_deviation': float(max_deviation),
            'correspondence_rate': float(correspondence_rate),
            'early_cuts': int(early_cuts),
            'late_cuts': int(late_cuts),
            'alignment_consistency': bool(alignment_consistency),
            'path_length': len(path),
            'segment_analysis': segment_stats,
            'meets_standards': {
                'rmse_standard': rmse <= 200,
                'consistency_standard': alignment_consistency,
                'correspondence_standard': correspondence_rate >= 95
            },
            'improvement_suggestions': []
        }
        
        # 生成改進建議
        if rmse > 200:
            evaluation['improvement_suggestions'].append({
                'aspect': 'RMSE',
                'current': float(rmse),
                'target': 200,
                'suggestion': '考慮增加動態窗口大小和使用局部約束來改善對齊精度'
            })
        
        if not alignment_consistency:
            evaluation['improvement_suggestions'].append({
                'aspect': '一致性',
                'current': {
                    'early_cuts': int(early_cuts),
                    'late_cuts': int(late_cuts)
                },
                'target': 'no_cuts',
                'suggestion': '建議添加平滑處理和異常點檢測來提高一致性'
            })
        
        if correspondence_rate < 95:
            evaluation['improvement_suggestions'].append({
                'aspect': '對應率',
                'current': float(correspondence_rate),
                'target': 95,
                'suggestion': '考慮使用特徵權重和多尺度DTW來提高對應準確率'
            })
        
        # 記錄評估結果
        segment_analysis = '\n'.join(f'   段落 {stats["segment_id"]}: 平均偏差={stats["mean_diff"]:.1f}ms, 標準差={stats["std_diff"]:.1f}ms' for stats in segment_stats)
        improvement_suggestions = '\n'.join(f'- {sugg["aspect"]}: 當前={sugg["current"]}, 目標={sugg["target"]}\n  {sugg["suggestion"]}' for sugg in evaluation['improvement_suggestions'])
        
        logger.info(f"""
對齊評估結果：
1. RMSE: {rmse:.2f} ms {'✓' if rmse <= 200 else '✗'}
2. 最大偏差: {max_deviation:.2f} ms
   過早切點: {early_cuts} 個
   過晚切點: {late_cuts} 個
   一致性: {'✓' if alignment_consistency else '✗'}
3. 對應率: {correspondence_rate:.2f}% {'✓' if correspondence_rate >= 95 else '✗'}

分段分析：
{segment_analysis}

改進建議：
{improvement_suggestions}
""")
        
        return evaluation
    
    def align_files(self, student_features, teacher_features):
        """對齊兩個特徵序列"""
        # 執行FastDTW對齊
        distance, path = self.align_features(student_features, teacher_features)
        
        # 評估對齊結果
        evaluation = self.evaluate_alignment(path, student_features, teacher_features)
        evaluation['dtw_distance'] = float(distance)
        
        return path.tolist(), evaluation

    def process_file_pair(self, teacher_file, student_file):
        """處理一對教師和學生的特徵文件"""
        try:
            logger.info(f"開始處理文件對:\n教師: {os.path.basename(teacher_file)}\n學生: {os.path.basename(student_file)}")
            
            # 加載特徵
            teacher_data = np.load(teacher_file)
            student_data = np.load(student_file)
            
            # 獲取特徵數據 - 只使用MFCC特徵
            teacher_features = teacher_data['mfcc']  # 先不轉置
            student_features = student_data['mfcc']  # 先不轉置
            
            # 檢查並修正特徵維度
            def normalize_features(features, name):
                """標準化特徵維度為(frames, 39)"""
                logger.debug(f"原始{name}特徵形狀: {features.shape}")
                
                # 如果已經是正確的形狀(frames, 39)
                if features.shape[1] == 39:
                    return features
                
                # 如果是(39, frames)需要轉置
                if features.shape[0] == 39:
                    return features.T
                
                # 處理13維MFCC特徵
                if 13 in features.shape:
                    # 確保13維在第二維
                    if features.shape[0] == 13:
                        features = features.T  # 轉置為(frames, 13)
                    
                    # 現在features應該是(frames, 13)
                    if features.shape[1] != 13:
                        raise ValueError(f"{name}特徵維度不正確: {features.shape}, 無法轉換為(frames, 13)")
                    
                    # 複製三次得到39維
                    features_39 = np.concatenate([features] * 3, axis=1)
                    logger.debug(f"轉換後{name}特徵形狀: {features_39.shape}")
                    return features_39
                
                raise ValueError(f"{name}特徵維度不正確: {features.shape}, 需要是(frames, 39)或可轉換的形式")
            
            teacher_features = normalize_features(teacher_features, "教師")
            student_features = normalize_features(student_features, "學生")
            
            logger.debug(f"教師特徵形狀: {teacher_features.shape}")
            logger.debug(f"學生特徵形狀: {student_features.shape}")
            
            # 計算DTW距離和路徑
            start_time = time.time()
            distance, path = self.align_features(teacher_features, student_features)
            processing_time = time.time() - start_time
            
            # 計算對齊質量指標
            path_np = np.array(path)
            teacher_indices = path_np[:, 0]
            student_indices = path_np[:, 1]
            
            # 計算時間戳對齊
            teacher_timestamps = np.arange(len(teacher_features)) * 0.01  # 假設幀移為10ms
            student_timestamps = np.arange(len(student_features)) * 0.01
            
            aligned_teacher_times = teacher_timestamps[teacher_indices]
            aligned_student_times = student_timestamps[student_indices]
            
            # 計算時間差異統計
            time_differences = aligned_teacher_times - aligned_student_times
            mean_difference = np.mean(time_differences)
            std_difference = np.std(time_differences)
            
            # 生成結果報告
            result = {
                "teacher_file": os.path.basename(teacher_file),
                "student_file": os.path.basename(student_file),
                "dtw_distance": float(distance),
                "processing_time": processing_time,
                "teacher_length": len(teacher_features),
                "student_length": len(student_features),
                "mean_time_difference": float(mean_difference),
                "std_time_difference": float(std_difference),
                "alignment_path": [[int(i), int(j)] for i, j in path]  # 使用alignment_path作為鍵名
            }
            
            logger.info(f"處理完成:\nDTW距離: {distance:.2f}\n處理時間: {processing_time:.2f}秒\n平均時間差異: {mean_difference:.3f}秒\n時間差異標準差: {std_difference:.3f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"處理文件對時出錯: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return None

def process_all_files(feature_dir):
    """處理所有配對的文件"""
    try:
        # 初始化DTW對齊器
        aligner = FastDTWAligner(feature_dir)
        
        # 查找並配對特徵文件
        valid_pairs = aligner.find_feature_files()
        
        if not valid_pairs:
            logger.warning("未找到有效的文件對")
            return
        
        logger.info(f"開始處理 {len(valid_pairs)} 對文件")
        
        # 逐個處理文件對
        results = []
        for i, (teacher_file, student_file) in enumerate(valid_pairs, 1):
            logger.info(f"處理第 {i}/{len(valid_pairs)} 對文件")
            
            result = aligner.process_file_pair(teacher_file, student_file)
            if result:
                results.append(result)
                
                # 保存中間結果
                if i % 10 == 0 or i == len(valid_pairs):
                    save_results(results)
        
        # 保存最終結果
        save_results(results)
        
    except Exception as e:
        logger.error(f"處理過程中出錯: {str(e)}")
        logger.error(traceback.format_exc())

def save_results(results):
    """保存處理結果"""
    try:
        os.makedirs("baseline", exist_ok=True)
        output_file = os.path.join("baseline", "alignment_results.json")
        
        # 轉換numpy數組為列表
        serializable_results = []
        for result in results:
            if result:
                result_copy = result.copy()
                result_copy["alignment_path"] = [
                    [int(i), int(j)] for i, j in result["alignment_path"]
                ]
                serializable_results.append(result_copy)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"結果已保存到 {output_file}")
        
    except Exception as e:
        logger.error(f"保存結果時出錯: {str(e)}")
        logger.error(traceback.format_exc())

def validate_directories():
    """驗證必要的目錄結構"""
    required_dirs = [
        'preprocess_teacher_audio',
        'preprocess_student_audio',
        'baseline'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.error(f"找不到必要的目錄：{dir_name}")
            return False
        if not any(dir_path.glob('*_features.npy')):
            logger.warning(f"目錄 {dir_name} 中沒有找到特徵文件")
            
    return True

def create_pairing_map():
    """創建並驗證配對映射，支持一個老師對應多個學生"""
    teacher_dir = Path('preprocess_teacher_audio')
    student_dir = Path('preprocess_student_audio')
    pairing_map = {}
    
    # 收集教師音頻
    logger.info("正在收集教師音頻文件...")
    teacher_files = list(teacher_dir.glob('*_features.npy'))
    for teacher_file in teacher_files:
        filename = teacher_file.stem
        parts = filename.split('_')
        lesson = '_'.join(parts[:2])
        utterance = parts[-2]
        key = (lesson, utterance)
        
        if key not in pairing_map:
            pairing_map[key] = {
                'teacher': teacher_file,
                'students': {},  # 改為字典以存儲每個學生的信息
                'status': 'pending'
            }
    
    # 收集學生音頻
    logger.info("正在收集學生音頻文件...")
    student_files = list(student_dir.glob('*_features.npy'))
    for student_file in student_files:
        filename = student_file.stem
        parts = filename.split('_')
        lesson = '_'.join(parts[:2])
        utterance = parts[-2]
        student_id = next(part for part in parts if part.startswith('Student'))
        key = (lesson, utterance)
        
        if key in pairing_map:
            if student_id not in pairing_map[key]['students']:
                pairing_map[key]['students'][student_id] = []
            pairing_map[key]['students'][student_id].append(student_file)
    
    return pairing_map

def validate_pairing(pairing_map):
    """驗證配對的有效性，考慮多個學生的情況"""
    validation_results = {
        'total_utterances': len(pairing_map),
        'total_students': 0,
        'valid_pairs': 0,
        'invalid_pairs': 0,
        'missing_student_audio': 0,
        'student_statistics': {},
        'details': []
    }
    
    # 收集所有學生ID
    all_students = set()
    for pair_info in pairing_map.values():
        all_students.update(pair_info['students'].keys())
    
    # 初始化每個學生的統計信息
    for student_id in all_students:
        validation_results['student_statistics'][student_id] = {
            'total_utterances': 0,
            'completed_utterances': 0,
            'missing_utterances': 0
        }
    
    for (lesson, utterance), pair_info in pairing_map.items():
        status = {
            'lesson': lesson,
            'utterance': utterance,
            'teacher_file': str(pair_info['teacher'].name),
            'student_count': len(pair_info['students']),
            'status': 'valid' if pair_info['students'] else 'missing_student_audio',
            'students': {}
        }
        
        # 統計每個學生的情況
        for student_id, student_files in pair_info['students'].items():
            status['students'][student_id] = {
                'files': [str(f.name) for f in student_files],
                'count': len(student_files)
            }
            
            validation_results['student_statistics'][student_id]['total_utterances'] += 1
            if student_files:
                validation_results['student_statistics'][student_id]['completed_utterances'] += 1
            else:
                validation_results['student_statistics'][student_id]['missing_utterances'] += 1
        
        if not pair_info['students']:
            validation_results['missing_student_audio'] += 1
            pair_info['status'] = 'invalid'
        else:
            validation_results['valid_pairs'] += sum(len(files) for files in pair_info['students'].values())
            pair_info['status'] = 'valid'
        
        validation_results['details'].append(status)
    
    validation_results['total_students'] = len(all_students)
    validation_results['invalid_pairs'] = validation_results['missing_student_audio']
    
    return validation_results

def save_validation_report(validation_results, baseline_dir):
    """保存驗證報告，包含學生間的比較"""
    report_path = baseline_dir / 'pairing_validation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n配對驗證報告：")
    logger.info(f"總話語數：{validation_results['total_utterances']}")
    logger.info(f"總學生數：{validation_results['total_students']}")
    logger.info(f"有效配對數：{validation_results['valid_pairs']}")
    logger.info(f"無效配對數：{validation_results['invalid_pairs']}")
    logger.info(f"缺少學生音頻：{validation_results['missing_student_audio']}")
    
    logger.info("\n學生完成情況：")
    for student_id, stats in validation_results['student_statistics'].items():
        completion_rate = (stats['completed_utterances'] / stats['total_utterances']) * 100
        logger.info(f"\n{student_id}:")
        logger.info(f"  總話語數：{stats['total_utterances']}")
        logger.info(f"  已完成：{stats['completed_utterances']}")
        logger.info(f"  缺失：{stats['missing_utterances']}")
        logger.info(f"  完成率：{completion_rate:.1f}%")
    
    logger.info(f"\n詳細報告已保存至：{report_path}")

def generate_final_report(results, baseline_dir, total_pairs, valid_pairs):
    """生成最終報告，包含詳細的評估指標分析"""
    # 基本統計
    avg_rmse = np.mean([r['evaluation']['rmse'] for r in results])
    avg_max_deviation = np.mean([r['evaluation']['max_deviation'] for r in results])
    avg_correspondence_rate = np.mean([r['evaluation']['correspondence_rate'] for r in results])
    
    # 標準達成統計
    standards_met = {
        'rmse': sum(1 for r in results if r['evaluation']['meets_standards']['rmse_standard']) / len(results) * 100,
        'consistency': sum(1 for r in results if r['evaluation']['meets_standards']['consistency_standard']) / len(results) * 100,
        'correspondence': sum(1 for r in results if r['evaluation']['meets_standards']['correspondence_standard']) / len(results) * 100
    }
    
    # 按課程和學生分組
    lessons = sorted(set(r['lesson'] for r in results))
    students = sorted(set(r['student_id'] for r in results))
    
    # 課程統計
    lesson_stats = {}
    for lesson in lessons:
        lesson_results = [r for r in results if r['lesson'] == lesson]
        lesson_stats[lesson] = {
            'total_utterances': len(lesson_results),
            'average_rmse': float(np.mean([r['evaluation']['rmse'] for r in lesson_results])),
            'average_max_deviation': float(np.mean([r['evaluation']['max_deviation'] for r in lesson_results])),
            'average_correspondence_rate': float(np.mean([r['evaluation']['correspondence_rate'] for r in lesson_results])),
            'standards_met': {
                'rmse': sum(1 for r in lesson_results if r['evaluation']['meets_standards']['rmse_standard']) / len(lesson_results) * 100,
                'consistency': sum(1 for r in lesson_results if r['evaluation']['meets_standards']['consistency_standard']) / len(lesson_results) * 100,
                'correspondence': sum(1 for r in lesson_results if r['evaluation']['meets_standards']['correspondence_standard']) / len(lesson_results) * 100
            }
        }
    
    # 學生統計
    student_stats = {}
    for student in students:
        student_results = [r for r in results if r['student_id'] == student]
        student_stats[student] = {
            'total_utterances': len(student_results),
            'average_rmse': float(np.mean([r['evaluation']['rmse'] for r in student_results])),
            'average_max_deviation': float(np.mean([r['evaluation']['max_deviation'] for r in student_results])),
            'average_correspondence_rate': float(np.mean([r['evaluation']['correspondence_rate'] for r in student_results])),
            'standards_met': {
                'rmse': sum(1 for r in student_results if r['evaluation']['meets_standards']['rmse_standard']) / len(student_results) * 100,
                'consistency': sum(1 for r in student_results if r['evaluation']['meets_standards']['consistency_standard']) / len(student_results) * 100,
                'correspondence': sum(1 for r in student_results if r['evaluation']['meets_standards']['correspondence_standard']) / len(student_results) * 100
            }
        }
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(results),
        'total_pairs': total_pairs,
        'valid_pairs': valid_pairs,
        'overall_metrics': {
            'average_rmse': float(avg_rmse),
            'average_max_deviation': float(avg_max_deviation),
            'average_correspondence_rate': float(avg_correspondence_rate)
        },
        'standards_compliance': standards_met,
        'lesson_statistics': lesson_stats,
        'student_statistics': student_stats
    }
    
    # 保存報告
    with open(baseline_dir / 'detailed_evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 輸出報告摘要
    logger.info(f"""
詳細評估報告：

1. 整體評估指標
   - RMSE: {avg_rmse:.2f} ms (標準: ≤ 200ms)
   - 最大偏差: {avg_max_deviation:.2f} ms
   - 對應率: {avg_correspondence_rate:.2f}% (標準: ≥ 95%)

2. 標準達成率
   - RMSE標準達成率: {standards_met['rmse']:.1f}%
   - 一致性標準達成率: {standards_met['consistency']:.1f}%
   - 對應率標準達成率: {standards_met['correspondence']:.1f}%

3. 課程分析
{chr(10).join(f'   {lesson}：RMSE={stats["average_rmse"]:.1f}ms, 對應率={stats["average_correspondence_rate"]:.1f}%' for lesson, stats in lesson_stats.items())}

4. 學生分析
{chr(10).join(f'   {student}：RMSE={stats["average_rmse"]:.1f}ms, 對應率={stats["average_correspondence_rate"]:.1f}%' for student, stats in student_stats.items())}

詳細報告已保存至：{baseline_dir}/detailed_evaluation_report.json
""")

if __name__ == "__main__":
    # 設置日誌記錄
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dtw_alignment_detailed.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 設置特徵文件目錄
    feature_dir = os.path.join("audio_feature_extraction", "04_feature_extraction_experiment")
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        
    logger.info(f"使用特徵文件目錄: {feature_dir}")
    
    # 處理所有文件
    process_all_files(feature_dir)
    
    try:
        # 設置日誌格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # 獲取當前目錄作為特徵目錄
        feature_dir = os.getcwd()
        
        # 處理所有文件
        process_all_files(feature_dir)
        
    except Exception as e:
        logging.error(f"程序執行出錯：{str(e)}\n{traceback.format_exc()}")
        sys.exit(1) 