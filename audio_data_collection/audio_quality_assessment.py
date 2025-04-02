#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音檔質量評估主程式
用於收集音檔、應用降噪方法、評估質量並生成報告
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

# 導入自定義模塊
from noise_reduction import spectral_subtraction, wiener_filter, batch_process_audio
from visualization import (
    plot_metrics_comparison, 
    plot_method_comparison, 
    plot_spectrogram_comparison, 
    plot_waveform_comparison,
    plot_snr_comparison
)

def find_wav_files(base_dir):
    """
    遞迴搜尋所有.wav音檔
    
    參數:
    - base_dir: 基礎目錄路徑
    
    返回:
    - wav_files: 所有.wav文件的路徑列表
    """
    wav_files = []
    
    # 遞迴搜尋所有session目錄
    session_dirs = glob.glob(os.path.join(base_dir, "session_*"))
    
    for session_dir in session_dirs:
        # 搜尋學生錄音
        student_wavs = glob.glob(os.path.join(session_dir, "student_recordings", "*.wav"))
        wav_files.extend(student_wavs)
        
        # 搜尋教師錄音
        teacher_wavs = glob.glob(os.path.join(session_dir, "teacher_recordings", "*.wav"))
        wav_files.extend(teacher_wavs)
    
    return wav_files

def match_teacher_student_recordings(wav_files):
    """
    匹配對應的教師和學生錄音
    
    參數:
    - wav_files: 所有音頻文件的路徑列表
    
    返回:
    - matched_pairs: 字典，鍵為學生錄音路徑，值為對應的教師錄音路徑
    """
    matched_pairs = {}
    
    # 分離教師和學生錄音
    teacher_files = [f for f in wav_files if 'Teacher' in f]
    student_files = [f for f in wav_files if 'Student' in f]
    
    # 為每個學生錄音尋找匹配的教師錄音
    for student_file in student_files:
        # 從文件名中提取課程、角色和話語編號
        filename = os.path.basename(student_file)
        match = re.match(r'(Lesson\d+)_(\w+)_Student\d+_utterance(\d+)\.wav', filename)
        
        if match:
            lesson, character, utterance = match.groups()
            
            # 構建對應的教師文件名模式
            teacher_pattern = f"{lesson}_{character}_Teacher_utterance{utterance}.wav"
            
            # 尋找匹配的教師文件
            for teacher_file in teacher_files:
                if teacher_pattern in teacher_file:
                    matched_pairs[student_file] = teacher_file
                    break
    
    return matched_pairs

def estimate_snr(audio_data):
    """
    估計信噪比(SNR)
    
    參數:
    - audio_data: 音頻數據
    
    返回:
    - snr: 估計的信噪比(dB)
    """
    # 使用前幾幀估計噪音水平
    noise_frames = min(int(len(audio_data) * 0.1), 2000)  # 使用前10%或前2000個樣本
    noise_power = np.mean(audio_data[:noise_frames]**2)
    
    # 估計信號功率
    signal_power = np.mean(audio_data**2)
    
    # 計算SNR
    if noise_power > 0 and signal_power > noise_power:
        snr = 10 * np.log10((signal_power - noise_power) / noise_power)
    else:
        snr = 0
    
    return snr

def calculate_pesq(reference, degraded, sr):
    """
    計算PESQ分數
    
    參數:
    - reference: 參考音頻
    - degraded: 待評估音頻
    - sr: 採樣率
    
    返回:
    - pesq_score: PESQ分數
    """
    try:
        from pypesq import pesq
        
        # 確保兩個音頻長度相同
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        # 正規化音量
        reference = reference / (np.max(np.abs(reference)) + 1e-10)
        degraded = degraded / (np.max(np.abs(degraded)) + 1e-10)
        
        # 計算PESQ
        pesq_score = pesq(reference, degraded, sr)
        
        return pesq_score
    except ImportError:
        print("警告: 未安裝pypesq庫，使用替代方法計算PESQ分數")
        return calculate_pesq_alternative(reference, degraded)

def calculate_pesq_alternative(reference, degraded):
    """
    使用替代方法計算類PESQ分數
    
    參數:
    - reference: 參考音頻
    - degraded: 待評估音頻
    
    返回:
    - pesq_like_score: 類PESQ分數
    """
    # 確保兩個音頻長度相同
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # 計算信噪比
    noise = degraded - reference
    signal_power = np.mean(reference**2)
    noise_power = np.mean(noise**2)
    
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = 100  # 極高的SNR
    
    # 計算相關性
    correlation = np.corrcoef(reference, degraded)[0, 1]
    
    # 計算頻譜距離
    ref_spec = np.abs(np.fft.fft(reference))
    deg_spec = np.abs(np.fft.fft(degraded))
    spec_dist = np.mean(np.abs(ref_spec - deg_spec) / (ref_spec + 1e-10))
    
    # 綜合評分 (將各指標映射到1.0-4.5的範圍，類似PESQ)
    # 這裡的權重需要通過實驗調整
    snr_score = min(max((snr - 5) / 35, 0), 1)  # 5dB->0, 40dB->1
    corr_score = max(correlation, 0)  # 0->0, 1->1
    spec_score = 1 - min(spec_dist, 1)  # 高距離->低分
    
    # 加權計算
    weights = {'snr': 0.4, 'corr': 0.4, 'spec': 0.2}
    quality_score = (
        weights['snr'] * snr_score +
        weights['corr'] * corr_score +
        weights['spec'] * spec_score
    )
    
    # 映射到PESQ範圍 (1.0-4.5)
    pesq_like_score = 1.0 + 3.5 * quality_score
    
    return pesq_like_score

def calculate_stoi(reference, degraded, sr):
    """
    計算STOI分數
    
    參數:
    - reference: 參考音頻
    - degraded: 待評估音頻
    - sr: 採樣率
    
    返回:
    - stoi_score: STOI分數
    """
    try:
        from pystoi import stoi
        
        # 確保兩個音頻長度相同
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        # 正規化音量
        reference = reference / (np.max(np.abs(reference)) + 1e-10)
        degraded = degraded / (np.max(np.abs(degraded)) + 1e-10)
        
        # 計算STOI
        stoi_score = stoi(reference, degraded, sr, extended=False)
        
        return stoi_score
    except ImportError:
        print("警告: 未安裝pystoi庫，使用替代方法計算STOI分數")
        return calculate_stoi_alternative(reference, degraded)

def calculate_stoi_alternative(reference, degraded):
    """
    使用替代方法計算類STOI分數
    
    參數:
    - reference: 參考音頻
    - degraded: 待評估音頻
    
    返回:
    - stoi_like_score: 類STOI分數
    """
    # 確保兩個音頻長度相同
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # 計算相關性
    correlation = np.corrcoef(reference, degraded)[0, 1]
    
    # 計算均方誤差
    mse = np.mean((reference - degraded)**2)
    
    # 計算信噪比
    noise = degraded - reference
    signal_power = np.mean(reference**2)
    noise_power = np.mean(noise**2)
    
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = 100  # 極高的SNR
    
    # 綜合評分 (映射到0-1的範圍，類似STOI)
    corr_score = max(correlation, 0)  # 0->0, 1->1
    mse_score = max(1 - mse * 10, 0)  # 高MSE->低分
    snr_score = min(max((snr - 5) / 35, 0), 1)  # 5dB->0, 40dB->1
    
    # 加權計算
    weights = {'corr': 0.5, 'mse': 0.3, 'snr': 0.2}
    stoi_like_score = (
        weights['corr'] * corr_score +
        weights['mse'] * mse_score +
        weights['snr'] * snr_score
    )
    
    return stoi_like_score

def evaluate_audio_quality(original_file, spectral_file, wiener_file, reference_file=None):
    """
    評估音頻質量
    
    參數:
    - original_file: 原始音頻文件路徑
    - spectral_file: 頻譜減法處理後的文件路徑
    - wiener_file: 維也納濾波器處理後的文件路徑
    - reference_file: 參考音頻文件路徑（如教師錄音）
    
    返回:
    - results: 包含各項評估指標的字典
    """
    # 讀取音頻文件
    original_audio, sr = librosa.load(original_file, sr=None, mono=True)
    spectral_audio, _ = librosa.load(spectral_file, sr=sr, mono=True)
    wiener_audio, _ = librosa.load(wiener_file, sr=sr, mono=True)
    
    # 如果有參考文件，讀取參考音頻
    if reference_file and os.path.exists(reference_file):
        reference_audio, _ = librosa.load(reference_file, sr=sr, mono=True)
    else:
        reference_audio = original_audio  # 如果沒有參考，使用原始音頻作為參考
    
    # 計算SNR
    original_snr = estimate_snr(original_audio)
    spectral_snr = estimate_snr(spectral_audio)
    wiener_snr = estimate_snr(wiener_audio)
    
    # 計算PESQ
    original_pesq = calculate_pesq(reference_audio, original_audio, sr)
    spectral_pesq = calculate_pesq(reference_audio, spectral_audio, sr)
    wiener_pesq = calculate_pesq(reference_audio, wiener_audio, sr)
    
    # 計算STOI
    original_stoi = calculate_stoi(reference_audio, original_audio, sr)
    spectral_stoi = calculate_stoi(reference_audio, spectral_audio, sr)
    wiener_stoi = calculate_stoi(reference_audio, wiener_audio, sr)
    
    # 構建結果字典
    results = {
        'original': {
            'snr': original_snr,
            'pesq': original_pesq,
            'stoi': original_stoi
        },
        'spectral_subtraction': {
            'snr': spectral_snr,
            'pesq': spectral_pesq,
            'stoi': spectral_stoi
        },
        'wiener_filter': {
            'snr': wiener_snr,
            'pesq': wiener_pesq,
            'stoi': wiener_stoi
        }
    }
    
    return results, (original_audio, spectral_audio, wiener_audio, sr)

def generate_visualizations(audio_data, file_id, output_dir):
    """
    為單個音頻生成視覺化
    
    參數:
    - audio_data: 元組 (原始音頻, 頻譜減法音頻, 維也納濾波器音頻, 採樣率)
    - file_id: 文件標識符
    - output_dir: 輸出目錄
    """
    original_audio, spectral_audio, wiener_audio, sr = audio_data
    
    # 創建輸出目錄
    os.makedirs(os.path.join(output_dir, 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'waveforms'), exist_ok=True)
    
    # 生成頻譜圖
    spectrogram_path = os.path.join(output_dir, 'spectrograms', f'{file_id}_spectrogram.png')
    plot_spectrogram_comparison(original_audio, spectral_audio, wiener_audio, sr, spectrogram_path)
    
    # 生成波形圖
    waveform_path = os.path.join(output_dir, 'waveforms', f'{file_id}_waveform.png')
    plot_waveform_comparison(original_audio, spectral_audio, wiener_audio, sr, waveform_path)

def generate_summary_report(results_df, output_file):
    """
    生成摘要報告
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - output_file: 輸出文件路徑
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("音檔質量評估摘要報告\n")
        f.write("=" * 50 + "\n\n")
        
        # 計算各方法的平均指標
        f.write("各方法平均指標:\n")
        f.write("-" * 50 + "\n")
        
        for method in ['original', 'spectral_subtraction', 'wiener_filter']:
            method_df = results_df[results_df['method'] == method]
            
            f.write(f"方法: {method}\n")
            f.write(f"  平均PESQ: {method_df['pesq'].mean():.2f}\n")
            f.write(f"  平均STOI: {method_df['stoi'].mean():.2f}\n")
            f.write(f"  平均SNR: {method_df['snr'].mean():.2f} dB\n")
            f.write("\n")
        
        # 合格率統計
        f.write("合格率統計:\n")
        f.write("-" * 50 + "\n")
        
        total_files = len(results_df['file_id'].unique())
        
        for method in ['original', 'spectral_subtraction', 'wiener_filter']:
            method_df = results_df[results_df['method'] == method]
            
            pesq_pass = method_df[method_df['pesq'] >= 3.0]
            stoi_pass = method_df[method_df['stoi'] >= 0.65]
            snr_pass = method_df[method_df['snr'] >= 20.0]
            
            f.write(f"方法: {method}\n")
            f.write(f"  PESQ合格率: {len(pesq_pass) / total_files * 100:.1f}%\n")
            f.write(f"  STOI合格率: {len(stoi_pass) / total_files * 100:.1f}%\n")
            f.write(f"  SNR合格率: {len(snr_pass) / total_files * 100:.1f}%\n")
            f.write("\n")
        
        # 方法比較
        f.write("方法比較:\n")
        f.write("-" * 50 + "\n")
        
        # 計算改進率
        original_pesq = results_df[results_df['method'] == 'original']['pesq'].mean()
        original_stoi = results_df[results_df['method'] == 'original']['stoi'].mean()
        original_snr = results_df[results_df['method'] == 'original']['snr'].mean()
        
        for method in ['spectral_subtraction', 'wiener_filter']:
            method_df = results_df[results_df['method'] == method]
            method_pesq = method_df['pesq'].mean()
            method_stoi = method_df['stoi'].mean()
            method_snr = method_df['snr'].mean()
            
            pesq_improvement = (method_pesq - original_pesq) / original_pesq * 100
            stoi_improvement = (method_stoi - original_stoi) / original_stoi * 100
            snr_improvement = (method_snr - original_snr) / original_snr * 100
            
            f.write(f"方法: {method}\n")
            f.write(f"  PESQ改進: {pesq_improvement:.1f}%\n")
            f.write(f"  STOI改進: {stoi_improvement:.1f}%\n")
            f.write(f"  SNR改進: {snr_improvement:.1f}%\n")
            f.write("\n")
        
        # 結論
        f.write("結論:\n")
        f.write("-" * 50 + "\n")
        
        # 根據平均指標決定最佳方法
        spectral_avg = results_df[results_df['method'] == 'spectral_subtraction'][['pesq', 'stoi', 'snr']].mean().mean()
        wiener_avg = results_df[results_df['method'] == 'wiener_filter'][['pesq', 'stoi', 'snr']].mean().mean()
        
        if spectral_avg > wiener_avg:
            best_method = "頻譜減法"
        else:
            best_method = "維也納濾波器"
        
        f.write(f"根據評估結果，{best_method}在整體上表現更好。\n\n")
        
        # 添加建議
        f.write("建議:\n")
        if best_method == "頻譜減法":
            f.write("1. 頻譜減法在處理語音噪聲方面表現較好，建議用於語音降噪。\n")
            f.write("2. 可以調整頻譜減法的beta參數以獲得更好的效果。\n")
        else:
            f.write("1. 維也納濾波器在保持語音清晰度方面表現較好，建議用於語音增強。\n")
            f.write("2. 可以調整維也納濾波器的噪聲估計方法以獲得更好的效果。\n")
    
    print(f"摘要報告已保存至: {output_file}")

def main(base_dir, output_dir):
    """
    主函數
    
    參數:
    - base_dir: 基礎目錄路徑
    - output_dir: 輸出目錄路徑
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed_audio', 'spectral_subtraction'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed_audio', 'wiener_filter'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results', 'visualizations'), exist_ok=True)
    
    # 搜尋所有.wav文件
    print("搜尋音檔...")
    wav_files = find_wav_files(base_dir)
    print(f"找到 {len(wav_files)} 個音檔")
    
    # 匹配教師和學生錄音
    print("匹配教師和學生錄音...")
    matched_pairs = match_teacher_student_recordings(wav_files)
    print(f"找到 {len(matched_pairs)} 對匹配的教師-學生錄音")
    
    # 處理音檔
    print("處理音檔...")
    student_files = list(matched_pairs.keys())
    
    # 批量處理音檔
    output_dir_spectral = os.path.join(output_dir, 'processed_audio', 'spectral_subtraction')
    output_dir_wiener = os.path.join(output_dir, 'processed_audio', 'wiener_filter')
    
    processed_files = batch_process_audio(student_files, output_dir_spectral, output_dir_wiener)
    
    # 評估音檔質量
    print("評估音檔質量...")
    results_data = []
    
    for i, (original_file, processed_paths) in enumerate(processed_files.items()):
        spectral_file = processed_paths['spectral']
        wiener_file = processed_paths['wiener']
        reference_file = matched_pairs.get(original_file)
        
        # 生成文件ID
        file_id = f"file_{i+1}"
        
        # 評估質量
        results, audio_data = evaluate_audio_quality(
            original_file, spectral_file, wiener_file, reference_file
        )
        
        # 添加到結果數據
        for method, metrics in results.items():
            results_data.append({
                'file_id': file_id,
                'original_file': original_file,
                'method': method,
                'pesq': metrics['pesq'],
                'stoi': metrics['stoi'],
                'snr': metrics['snr']
            })
        
        # 生成單個音檔的視覺化
        generate_visualizations(audio_data, file_id, os.path.join(output_dir, 'results'))
    
    # 創建結果DataFrame
    results_df = pd.DataFrame(results_data)
    
    # 保存結果到CSV
    csv_path = os.path.join(output_dir, 'results', 'metrics_report.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"評估結果已保存至: {csv_path}")
    
    # 生成視覺化
    print("生成視覺化...")
    vis_dir = os.path.join(output_dir, 'results', 'visualizations')
    
    # 指標對比圖
    plot_metrics_comparison(results_df, os.path.join(vis_dir, 'metrics_comparison.png'))
    
    # 方法比較圖
    plot_method_comparison(results_df, os.path.join(vis_dir, 'method_comparison.png'))
    
    # SNR比較圖
    plot_snr_comparison(results_df, os.path.join(vis_dir, 'snr_comparison.png'))
    
    # 生成摘要報告
    print("生成摘要報告...")
    summary_path = os.path.join(output_dir, 'results', 'summary_report.txt')
    generate_summary_report(results_df, summary_path)
    
    print("處理完成!")
    print(f"所有結果已保存至: {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "."  # 當前目錄
    
    output_dir = "audio_data_collection"
    
    main(base_dir, output_dir)
