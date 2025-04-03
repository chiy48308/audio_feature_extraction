#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
降噪方法實現模塊
包含頻譜減法和維也納濾波器兩種降噪方法
"""

import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm

def spectral_subtraction(audio_data, sr, frame_length=2048, hop_length=512, beta=0.01, noise_frames=10):
    """
    使用頻譜減法進行降噪
    
    參數:
    - audio_data: 輸入音頻數據
    - sr: 採樣率
    - frame_length: STFT窗口大小
    - hop_length: STFT跳躍大小
    - beta: 頻譜減法參數，控制減法強度
    - noise_frames: 用於估計噪音的幀數
    
    返回:
    - denoised_audio: 降噪後的音頻
    """
    # 計算STFT
    stft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    
    # 估計噪音頻譜 (使用前幾幀或靜音段)
    noise_magnitude = np.mean(np.abs(stft[:, :noise_frames]), axis=1, keepdims=True)
    
    # 頻譜減法
    denoised_magnitude = np.maximum(np.abs(stft) - beta * noise_magnitude, 0)
    
    # 重建信號
    denoised_stft = denoised_magnitude * np.exp(1j * np.angle(stft))
    denoised_audio = librosa.istft(denoised_stft, hop_length=hop_length)
    
    # 確保長度一致
    if len(denoised_audio) < len(audio_data):
        denoised_audio = np.pad(denoised_audio, (0, len(audio_data) - len(denoised_audio)))
    else:
        denoised_audio = denoised_audio[:len(audio_data)]
    
    return denoised_audio

def wiener_filter(audio_data, sr, frame_length=2048, hop_length=512, noise_frames=10):
    """
    使用維也納濾波器進行降噪
    
    參數:
    - audio_data: 輸入音頻數據
    - sr: 採樣率
    - frame_length: STFT窗口大小
    - hop_length: STFT跳躍大小
    - noise_frames: 用於估計噪音的幀數
    
    返回:
    - denoised_audio: 降噪後的音頻
    """
    # 計算STFT
    stft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
    
    # 估計噪音功率譜
    noise_power = np.mean(np.abs(stft[:, :noise_frames])**2, axis=1, keepdims=True)
    
    # 估計信號功率譜
    sig_power = np.abs(stft)**2
    
    # 計算維也納濾波器增益
    # 避免除以零
    eps = np.finfo(float).eps
    gain = sig_power / (sig_power + noise_power + eps)
    
    # 應用濾波器
    denoised_stft = stft * gain
    
    # 重建信號
    denoised_audio = librosa.istft(denoised_stft, hop_length=hop_length)
    
    # 確保長度一致
    if len(denoised_audio) < len(audio_data):
        denoised_audio = np.pad(denoised_audio, (0, len(audio_data) - len(denoised_audio)))
    else:
        denoised_audio = denoised_audio[:len(audio_data)]
    
    return denoised_audio

def process_audio_file(input_file, output_dir_spectral, output_dir_wiener):
    """
    處理單個音頻文件，應用兩種降噪方法並保存結果
    
    參數:
    - input_file: 輸入音頻文件路徑
    - output_dir_spectral: 頻譜減法輸出目錄
    - output_dir_wiener: 維也納濾波器輸出目錄
    
    返回:
    - spectral_output_path: 頻譜減法處理後的文件路徑
    - wiener_output_path: 維也納濾波器處理後的文件路徑
    """
    # 讀取音頻
    audio_data, sr = librosa.load(input_file, sr=None, mono=True)
    
    # 應用頻譜減法
    spectral_denoised = spectral_subtraction(audio_data, sr)
    
    # 應用維也納濾波器
    wiener_denoised = wiener_filter(audio_data, sr)
    
    # 構建輸出文件路徑
    filename = os.path.basename(input_file)
    spectral_output_path = os.path.join(output_dir_spectral, filename)
    wiener_output_path = os.path.join(output_dir_wiener, filename)
    
    # 保存處理後的音頻
    sf.write(spectral_output_path, spectral_denoised, sr)
    sf.write(wiener_output_path, wiener_denoised, sr)
    
    return spectral_output_path, wiener_output_path

def batch_process_audio(input_files, output_dir_spectral, output_dir_wiener):
    """
    批量處理音頻文件
    
    參數:
    - input_files: 輸入音頻文件路徑列表
    - output_dir_spectral: 頻譜減法輸出目錄
    - output_dir_wiener: 維也納濾波器輸出目錄
    
    返回:
    - processed_files: 字典，包含原始文件和處理後文件的對應關係
    """
    processed_files = {}
    
    for input_file in tqdm(input_files, desc="處理音檔"):
        try:
            spectral_output, wiener_output = process_audio_file(
                input_file, output_dir_spectral, output_dir_wiener
            )
            processed_files[input_file] = {
                'spectral': spectral_output,
                'wiener': wiener_output
            }
        except Exception as e:
            print(f"處理文件 {input_file} 時出錯: {str(e)}")
    
    return processed_files

if __name__ == "__main__":
    # 測試代碼
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_dir_spectral = "processed_audio/spectral_subtraction"
        output_dir_wiener = "processed_audio/wiener_filter"
        
        os.makedirs(output_dir_spectral, exist_ok=True)
        os.makedirs(output_dir_wiener, exist_ok=True)
        
        spectral_output, wiener_output = process_audio_file(
            input_file, output_dir_spectral, output_dir_wiener
        )
        
        print(f"頻譜減法處理結果: {spectral_output}")
        print(f"維也納濾波器處理結果: {wiener_output}")
    else:
        print("請提供輸入音頻文件路徑")
