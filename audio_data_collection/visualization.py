#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
視覺化工具模塊
用於生成各種比較圖表
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator
import librosa
import librosa.display

def plot_metrics_comparison(results_df, output_path):
    """
    繪製處理前後的指標對比圖
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - output_path: 輸出圖片路徑
    """
    plt.figure(figsize=(14, 10))
    
    # PESQ比較
    plt.subplot(2, 2, 1)
    sns.boxplot(x='method', y='pesq', data=results_df)
    plt.title('PESQ分數比較', fontsize=14)
    plt.ylabel('PESQ分數', fontsize=12)
    plt.xlabel('處理方法', fontsize=12)
    plt.axhline(y=3.0, color='r', linestyle='--', alpha=0.5)
    plt.text(0, 3.05, 'PESQ閾值(3.0)', color='r', alpha=0.7)
    
    # STOI比較
    plt.subplot(2, 2, 2)
    sns.boxplot(x='method', y='stoi', data=results_df)
    plt.title('STOI分數比較', fontsize=14)
    plt.ylabel('STOI分數', fontsize=12)
    plt.xlabel('處理方法', fontsize=12)
    plt.axhline(y=0.65, color='r', linestyle='--', alpha=0.5)
    plt.text(0, 0.66, 'STOI閾值(0.65)', color='r', alpha=0.7)
    
    # PESQ折線圖
    plt.subplot(2, 2, 3)
    
    # 獲取唯一的文件ID
    file_ids = sorted(results_df['file_id'].unique())
    
    # 為每個方法繪製折線
    for method in results_df['method'].unique():
        method_data = []
        for file_id in file_ids:
            subset = results_df[(results_df['method'] == method) & (results_df['file_id'] == file_id)]
            if not subset.empty:
                method_data.append(subset['pesq'].values[0])
            else:
                method_data.append(np.nan)
        
        plt.plot(range(len(file_ids)), method_data, marker='o', label=method)
    
    plt.title('各音檔PESQ分數', fontsize=14)
    plt.ylabel('PESQ分數', fontsize=12)
    plt.xlabel('音檔ID', fontsize=12)
    plt.xticks(range(len(file_ids)), [f"{i+1}" for i in range(len(file_ids))], rotation=45)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # STOI折線圖
    plt.subplot(2, 2, 4)
    
    # 為每個方法繪製折線
    for method in results_df['method'].unique():
        method_data = []
        for file_id in file_ids:
            subset = results_df[(results_df['method'] == method) & (results_df['file_id'] == file_id)]
            if not subset.empty:
                method_data.append(subset['stoi'].values[0])
            else:
                method_data.append(np.nan)
        
        plt.plot(range(len(file_ids)), method_data, marker='o', label=method)
    
    plt.title('各音檔STOI分數', fontsize=14)
    plt.ylabel('STOI分數', fontsize=12)
    plt.xlabel('音檔ID', fontsize=12)
    plt.xticks(range(len(file_ids)), [f"{i+1}" for i in range(len(file_ids))], rotation=45)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"指標對比圖已保存至: {output_path}")

def plot_method_comparison(results_df, output_path):
    """
    繪製不同降噪方法之間的指標比較
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - output_path: 輸出圖片路徑
    """
    # 提取不同方法的結果
    original_results = results_df[results_df['method'] == 'original']
    spectral_results = results_df[results_df['method'] == 'spectral_subtraction']
    wiener_results = results_df[results_df['method'] == 'wiener_filter']
    
    plt.figure(figsize=(12, 10))
    
    # 散點圖比較
    plt.subplot(2, 1, 1)
    plt.scatter(original_results['pesq'], original_results['stoi'], 
                label='原始音檔', alpha=0.7, s=80, marker='o', color='blue')
    plt.scatter(spectral_results['pesq'], spectral_results['stoi'], 
                label='頻譜減法', alpha=0.7, s=80, marker='^', color='green')
    plt.scatter(wiener_results['pesq'], wiener_results['stoi'], 
                label='維也納濾波器', alpha=0.7, s=80, marker='s', color='red')
    
    plt.xlabel('PESQ分數', fontsize=12)
    plt.ylabel('STOI分數', fontsize=12)
    plt.title('降噪方法比較: PESQ vs STOI', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加參考線
    plt.axhline(y=0.65, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=3.0, color='g', linestyle='--', alpha=0.5)
    plt.text(3.05, 0.5, 'PESQ閾值(3.0)', color='g', alpha=0.7, rotation=90)
    plt.text(2.0, 0.66, 'STOI閾值(0.65)', color='r', alpha=0.7)
    
    # 計算平均值並繪製條形圖
    plt.subplot(2, 1, 2)
    
    # 計算每種方法的平均PESQ和STOI
    methods = ['original', 'spectral_subtraction', 'wiener_filter']
    method_names = ['原始音檔', '頻譜減法', '維也納濾波器']
    
    pesq_means = [results_df[results_df['method'] == method]['pesq'].mean() for method in methods]
    stoi_means = [results_df[results_df['method'] == method]['stoi'].mean() for method in methods]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, pesq_means, width, label='平均PESQ')
    rects2 = ax.bar(x + width/2, stoi_means, width, label='平均STOI')
    
    ax.set_ylabel('分數')
    ax.set_title('各方法平均PESQ和STOI分數')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    
    # 在柱狀圖上添加數值標籤
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_bar.png'), dpi=300)
    plt.close()
    
    print(f"方法比較圖已保存至: {output_path}")

def plot_spectrogram_comparison(original_audio, spectral_audio, wiener_audio, sr, output_path):
    """
    繪製頻譜圖比較
    
    參數:
    - original_audio: 原始音頻數據
    - spectral_audio: 頻譜減法處理後的音頻數據
    - wiener_audio: 維也納濾波器處理後的音頻數據
    - sr: 採樣率
    - output_path: 輸出圖片路徑
    """
    plt.figure(figsize=(15, 12))
    
    # 原始音頻頻譜圖
    plt.subplot(3, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始音頻頻譜圖')
    
    # 頻譜減法處理後的頻譜圖
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(spectral_audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('頻譜減法處理後的頻譜圖')
    
    # 維也納濾波器處理後的頻譜圖
    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wiener_audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('維也納濾波器處理後的頻譜圖')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"頻譜圖比較已保存至: {output_path}")

def plot_waveform_comparison(original_audio, spectral_audio, wiener_audio, sr, output_path):
    """
    繪製波形比較
    
    參數:
    - original_audio: 原始音頻數據
    - spectral_audio: 頻譜減法處理後的音頻數據
    - wiener_audio: 維也納濾波器處理後的音頻數據
    - sr: 採樣率
    - output_path: 輸出圖片路徑
    """
    plt.figure(figsize=(15, 10))
    
    # 原始音頻波形
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(original_audio, sr=sr)
    plt.title('原始音頻波形')
    plt.ylabel('振幅')
    
    # 頻譜減法處理後的波形
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(spectral_audio, sr=sr)
    plt.title('頻譜減法處理後的波形')
    plt.ylabel('振幅')
    
    # 維也納濾波器處理後的波形
    plt.subplot(3, 1, 3)
    librosa.display.waveshow(wiener_audio, sr=sr)
    plt.title('維也納濾波器處理後的波形')
    plt.ylabel('振幅')
    plt.xlabel('時間 (秒)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"波形比較已保存至: {output_path}")

def plot_snr_comparison(results_df, output_path):
    """
    繪製SNR比較圖
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - output_path: 輸出圖片路徑
    """
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(x='method', y='snr', data=results_df)
    plt.title('信噪比(SNR)比較', fontsize=14)
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.xlabel('處理方法', fontsize=12)
    plt.axhline(y=20, color='r', linestyle='--', alpha=0.5)
    plt.text(0, 21, 'SNR閾值(20 dB)', color='r', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"SNR比較圖已保存至: {output_path}")

if __name__ == "__main__":
    # 測試代碼
    # 創建模擬數據
    np.random.seed(42)
    methods = ['original', 'spectral_subtraction', 'wiener_filter']
    file_ids = [f'file_{i}' for i in range(10)]
    
    data = []
    for method in methods:
        for file_id in file_ids:
            if method == 'original':
                pesq = np.random.uniform(2.0, 3.0)
                stoi = np.random.uniform(0.5, 0.7)
                snr = np.random.uniform(15, 25)
            elif method == 'spectral_subtraction':
                pesq = np.random.uniform(2.5, 3.5)
                stoi = np.random.uniform(0.6, 0.8)
                snr = np.random.uniform(18, 28)
            else:  # wiener_filter
                pesq = np.random.uniform(3.0, 4.0)
                stoi = np.random.uniform(0.7, 0.9)
                snr = np.random.uniform(20, 30)
            
            data.append({
                'method': method,
                'file_id': file_id,
                'pesq': pesq,
                'stoi': stoi,
                'snr': snr
            })
    
    results_df = pd.DataFrame(data)
    
    # 測試繪圖函數
    os.makedirs('test_plots', exist_ok=True)
    plot_metrics_comparison(results_df, 'test_plots/metrics_comparison.png')
    plot_method_comparison(results_df, 'test_plots/method_comparison.png')
    plot_snr_comparison(results_df, 'test_plots/snr_comparison.png')
    
    # 測試頻譜圖和波形比較
    # 創建模擬音頻數據
    duration = 3  # 秒
    sr = 22050  # 採樣率
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 原始音頻 (帶噪聲的正弦波)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # 模擬處理後的音頻
    spectral_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
    wiener_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(len(t))
    
    plot_spectrogram_comparison(original_audio, spectral_audio, wiener_audio, sr, 'test_plots/spectrogram_comparison.png')
    plot_waveform_comparison(original_audio, spectral_audio, wiener_audio, sr, 'test_plots/waveform_comparison.png')
