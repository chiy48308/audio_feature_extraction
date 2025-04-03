#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示範腳本
用於展示音檔質量評估與降噪比較工具的功能
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

# 導入自定義模塊
from noise_reduction import spectral_subtraction, wiener_filter
from visualization import plot_waveform_comparison, plot_spectrogram_comparison

def add_noise(audio, noise_level=0.1):
    """
    向音頻添加高斯白噪聲
    
    參數:
    - audio: 輸入音頻數據
    - noise_level: 噪聲水平 (0-1)
    
    返回:
    - noisy_audio: 添加噪聲後的音頻
    """
    noise = np.random.normal(0, noise_level, len(audio))
    noisy_audio = audio + noise
    return noisy_audio

def demo_noise_reduction(input_file, output_dir):
    """
    演示降噪效果
    
    參數:
    - input_file: 輸入音頻文件路徑
    - output_dir: 輸出目錄
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取音頻
    print(f"讀取音頻文件: {input_file}")
    audio, sr = librosa.load(input_file, sr=None, mono=True)
    
    # 添加噪聲
    print("添加噪聲...")
    noisy_audio = add_noise(audio, noise_level=0.05)
    
    # 應用降噪方法
    print("應用頻譜減法...")
    spectral_denoised = spectral_subtraction(noisy_audio, sr)
    
    print("應用維也納濾波器...")
    wiener_denoised = wiener_filter(noisy_audio, sr)
    
    # 保存處理後的音頻
    noisy_output = os.path.join(output_dir, "noisy.wav")
    spectral_output = os.path.join(output_dir, "spectral_denoised.wav")
    wiener_output = os.path.join(output_dir, "wiener_denoised.wav")
    
    print("保存處理後的音頻...")
    sf.write(noisy_output, noisy_audio, sr)
    sf.write(spectral_output, spectral_denoised, sr)
    sf.write(wiener_output, wiener_denoised, sr)
    
    # 生成視覺化
    print("生成視覺化...")
    
    # 波形比較
    waveform_path = os.path.join(output_dir, "waveform_comparison.png")
    plot_waveform_comparison(noisy_audio, spectral_denoised, wiener_denoised, sr, waveform_path)
    
    # 頻譜圖比較
    spectrogram_path = os.path.join(output_dir, "spectrogram_comparison.png")
    plot_spectrogram_comparison(noisy_audio, spectral_denoised, wiener_denoised, sr, spectrogram_path)
    
    print("演示完成!")
    print(f"所有結果已保存至: {output_dir}")
    
    return {
        "noisy": noisy_output,
        "spectral": spectral_output,
        "wiener": wiener_output,
        "waveform": waveform_path,
        "spectrogram": spectrogram_path
    }

def main():
    """
    主函數
    """
    # 檢查命令行參數
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 搜尋第一個可用的音頻文件
        session_dirs = [d for d in os.listdir(".") if d.startswith("session_")]
        
        if not session_dirs:
            print("錯誤: 未找到session目錄")
            return
        
        # 搜尋第一個可用的音頻文件
        for session_dir in session_dirs:
            student_dir = os.path.join(session_dir, "student_recordings")
            if os.path.exists(student_dir):
                wav_files = [f for f in os.listdir(student_dir) if f.endswith(".wav")]
                if wav_files:
                    input_file = os.path.join(student_dir, wav_files[0])
                    break
        else:
            print("錯誤: 未找到音頻文件")
            return
    
    # 設置輸出目錄
    output_dir = "audio_data_collection/demo_output"
    
    # 運行演示
    results = demo_noise_reduction(input_file, output_dir)
    
    # 打印結果
    print("\n演示結果:")
    print(f"原始音頻: {input_file}")
    print(f"添加噪聲後的音頻: {results['noisy']}")
    print(f"頻譜減法處理後的音頻: {results['spectral']}")
    print(f"維也納濾波器處理後的音頻: {results['wiener']}")
    print(f"波形比較圖: {results['waveform']}")
    print(f"頻譜圖比較: {results['spectrogram']}")

if __name__ == "__main__":
    main()
