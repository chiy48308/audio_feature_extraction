#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音檔格式評估工具
用於評估音檔的基本格式和質量指標
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

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

def check_audio_format(file_path):
    """
    檢查音頻格式
    
    參數:
    - file_path: 音頻文件路徑
    
    返回:
    - format_info: 包含格式信息的字典
    """
    try:
        # 檢查文件類型
        import subprocess
        result = subprocess.run(['file', file_path], capture_output=True, text=True)
        file_info = result.stdout
        
        # 檢查是否為WebM格式（學生錄音）
        if 'WebM' in file_info:
            # 使用mediainfo獲取WebM格式信息
            media_result = subprocess.run(['mediainfo', file_path], capture_output=True, text=True)
            media_info = media_result.stdout
            
            # 解析mediainfo輸出
            sample_rate = 48000  # 默認值
            bit_depth = 32       # 默認值
            channels = 1         # 默認值
            duration = 0         # 默認值
            
            # 從mediainfo輸出中提取信息
            import re
            sample_rate_match = re.search(r'Sampling rate\s+:\s+([\d\.]+)\s+kHz', media_info)
            if sample_rate_match:
                sample_rate = float(sample_rate_match.group(1)) * 1000
            
            bit_depth_match = re.search(r'Bit depth\s+:\s+(\d+)\s+bits', media_info)
            if bit_depth_match:
                bit_depth = int(bit_depth_match.group(1))
            
            channels_match = re.search(r'Channel\(s\)\s+:\s+(\d+)\s+channel', media_info)
            if channels_match:
                channels = int(channels_match.group(1))
            
            # 檢查是否符合學生錄音標準
            sample_rate_ok = abs(sample_rate - 48000) < 100  # 允許小誤差
            bit_depth_ok = bit_depth == 32
            channels_ok = channels == 1
            
            # 格式是否符合標準
            format_ok = sample_rate_ok and bit_depth_ok and channels_ok
            
            return {
                'file_path': file_path,
                'sample_rate': sample_rate,
                'bit_depth': f'{bit_depth}-bit Opus',
                'channels': channels,
                'duration': duration,
                'format_ok': format_ok,
                'file_type': 'WebM'
            }
        else:
            # 使用soundfile獲取音頻格式信息（教師錄音）
            info = sf.info(file_path)
            
            # 檢查取樣率
            sample_rate_ok = info.samplerate == 16000
            
            # 檢查位元深度 - 更新為PCM_32
            bit_depth_ok = info.subtype == 'PCM_32'
            
            # 檢查聲道數
            channels_ok = info.channels == 1
            
            # 格式是否符合標準
            format_ok = sample_rate_ok and bit_depth_ok and channels_ok
            
            return {
                'file_path': file_path,
                'sample_rate': info.samplerate,
                'bit_depth': '32-bit' if bit_depth_ok else info.subtype,
                'channels': info.channels,
                'duration': info.duration,
                'format_ok': format_ok,
                'file_type': 'WAV'
            }
    except Exception as e:
        print(f"檢查文件 {file_path} 時出錯: {str(e)}")
        return {
            'file_path': file_path,
            'sample_rate': None,
            'bit_depth': None,
            'channels': None,
            'duration': None,
            'format_ok': False
        }

def detect_silence(audio_data, sr, threshold_db=-40):
    """
    檢測靜音
    
    參數:
    - audio_data: 音頻數據
    - sr: 採樣率
    - threshold_db: 靜音閾值 (dB)
    
    返回:
    - silence_info: 包含靜音信息的字典
    """
    # 計算幀長度 (10ms)
    frame_length = int(sr * 0.01)
    hop_length = frame_length
    
    # 計算每幀的RMS能量
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 轉換為dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # 檢測靜音幀
    silence_frames = rms_db < threshold_db
    
    # 計算靜音比例
    silence_ratio = np.mean(silence_frames)
    
    # 檢測連續靜音段
    silence_segments = []
    in_silence = False
    start_frame = 0
    
    for i, is_silence in enumerate(silence_frames):
        if is_silence and not in_silence:
            # 靜音開始
            in_silence = True
            start_frame = i
        elif not is_silence and in_silence:
            # 靜音結束
            in_silence = False
            duration = (i - start_frame) * hop_length / sr
            silence_segments.append(duration)
    
    # 如果結束時仍在靜音中
    if in_silence:
        duration = (len(silence_frames) - start_frame) * hop_length / sr
        silence_segments.append(duration)
    
    # 計算最長靜音段
    max_silence_duration = max(silence_segments) if silence_segments else 0
    
    # 檢查是否符合標準
    silence_ratio_ok = silence_ratio < 0.3  # 靜音比例 < 30%
    max_silence_ok = max_silence_duration < 1.0  # 單段靜音 < 1秒
    silence_ok = silence_ratio_ok and max_silence_ok
    
    return {
        'silence_ratio': silence_ratio,
        'max_silence_duration': max_silence_duration,
        'silence_ok': silence_ok
    }

def check_volume_levels(audio_data):
    """
    檢查音量水平
    
    參數:
    - audio_data: 音頻數據
    
    返回:
    - volume_info: 包含音量信息的字典
    """
    # 計算RMS能量
    rms = np.sqrt(np.mean(audio_data**2))
    
    # 轉換為dBFS
    rms_dbfs = 20 * np.log10(rms) if rms > 0 else -100
    
    # 計算峰值
    peak = np.max(np.abs(audio_data))
    peak_dbfs = 20 * np.log10(peak) if peak > 0 else -100
    
    # 檢查是否符合標準
    rms_ok = rms_dbfs > -30  # RMS > -30 dBFS
    peak_ok = peak_dbfs < 0  # Peak < 0 dBFS
    volume_ok = rms_ok and peak_ok
    
    return {
        'rms_dbfs': rms_dbfs,
        'peak_dbfs': peak_dbfs,
        'volume_ok': volume_ok
    }

def check_amplitude_stability(audio_data, sr):
    """
    檢查音量穩定性
    
    參數:
    - audio_data: 音頻數據
    - sr: 採樣率
    
    返回:
    - stability_info: 包含穩定性信息的字典
    """
    # 計算幀長度 (100ms)
    frame_length = int(sr * 0.1)
    hop_length = frame_length
    
    # 計算每幀的RMS能量
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 計算RMS的標準差
    rms_std = np.std(rms)
    
    # 計算RMS的變異係數 (標準差/平均值)
    rms_mean = np.mean(rms)
    rms_cv = rms_std / rms_mean if rms_mean > 0 else 0
    
    # 檢查是否符合標準 (變異係數 < 0.5 表示相對穩定)
    stability_ok = rms_cv < 0.5
    
    return {
        'rms_std': rms_std,
        'rms_cv': rms_cv,
        'stability_ok': stability_ok
    }

def estimate_snr(audio_data):
    """
    估計信噪比(SNR)
    
    參數:
    - audio_data: 音頻數據
    
    返回:
    - snr_info: 包含SNR信息的字典
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
    
    # 檢查是否符合標準
    snr_ok = snr >= 20  # SNR ≥ 20 dB
    
    return {
        'snr': snr,
        'snr_ok': snr_ok
    }

def assess_audio_file(file_path):
    """
    評估單個音頻文件
    
    參數:
    - file_path: 音頻文件路徑
    
    返回:
    - assessment: 包含評估結果的字典
    """
    try:
        # 檢查音頻格式
        format_info = check_audio_format(file_path)
        
        # 如果格式檢查失敗，返回基本信息
        if not format_info['sample_rate']:
            return {
                'file_path': file_path,
                'assessment_ok': False,
                'error': '無法讀取音頻文件'
            }
        
        # 讀取音頻數據
        audio_data, sr = librosa.load(file_path, sr=None, mono=True)
        
        # 檢測靜音
        silence_info = detect_silence(audio_data, sr)
        
        # 檢查音量水平
        volume_info = check_volume_levels(audio_data)
        
        # 檢查音量穩定性
        stability_info = check_amplitude_stability(audio_data, sr)
        
        # 估計SNR
        snr_info = estimate_snr(audio_data)
        
        # 綜合評估
        assessment_ok = (
            format_info['format_ok'] and
            silence_info['silence_ok'] and
            volume_info['volume_ok'] and
            stability_info['stability_ok'] and
            snr_info['snr_ok']
        )
        
        # 構建結果字典
        assessment = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'sample_rate': format_info['sample_rate'],
            'bit_depth': format_info['bit_depth'],
            'channels': format_info['channels'],
            'duration': format_info['duration'],
            'silence_ratio': silence_info['silence_ratio'],
            'max_silence_duration': silence_info['max_silence_duration'],
            'rms_dbfs': volume_info['rms_dbfs'],
            'peak_dbfs': volume_info['peak_dbfs'],
            'rms_cv': stability_info['rms_cv'],
            'snr': snr_info['snr'],
            'format_ok': format_info['format_ok'],
            'silence_ok': silence_info['silence_ok'],
            'volume_ok': volume_info['volume_ok'],
            'stability_ok': stability_info['stability_ok'],
            'snr_ok': snr_info['snr_ok'],
            'assessment_ok': assessment_ok
        }
        
        return assessment
    except Exception as e:
        print(f"評估文件 {file_path} 時出錯: {str(e)}")
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'assessment_ok': False,
            'error': str(e)
        }

def batch_assess_audio(wav_files, output_dir):
    """
    批量評估音頻文件
    
    參數:
    - wav_files: 音頻文件路徑列表
    - output_dir: 輸出目錄
    
    返回:
    - results_df: 包含評估結果的DataFrame
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 評估結果列表
    results = []
    
    # 批量評估
    for file_path in tqdm(wav_files, desc="評估音檔"):
        assessment = assess_audio_file(file_path)
        results.append(assessment)
    
    # 創建DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存結果到CSV
    csv_path = os.path.join(output_dir, 'audio_format_assessment.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"評估結果已保存至: {csv_path}")
    
    return results_df

def generate_summary_report(results_df, output_dir):
    """
    生成摘要報告
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - output_dir: 輸出目錄
    """
    # 創建輸出目錄
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 計算合格率
    total_files = len(results_df)
    format_pass = len(results_df[results_df['format_ok'] == True])
    silence_pass = len(results_df[results_df['silence_ok'] == True])
    volume_pass = len(results_df[results_df['volume_ok'] == True])
    stability_pass = len(results_df[results_df['stability_ok'] == True])
    snr_pass = len(results_df[results_df['snr_ok'] == True])
    overall_pass = len(results_df[results_df['assessment_ok'] == True])
    
    # 生成摘要報告
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("音檔格式評估摘要報告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"評估音檔總數: {total_files}\n\n")
        
        f.write("合格率統計:\n")
        f.write("-" * 50 + "\n")
        f.write(f"錄音格式合格率: {format_pass / total_files * 100:.1f}%\n")
        f.write(f"靜音檢測合格率: {silence_pass / total_files * 100:.1f}%\n")
        f.write(f"音量範圍合格率: {volume_pass / total_files * 100:.1f}%\n")
        f.write(f"音量穩定性合格率: {stability_pass / total_files * 100:.1f}%\n")
        f.write(f"信噪比合格率: {snr_pass / total_files * 100:.1f}%\n")
        f.write(f"整體合格率: {overall_pass / total_files * 100:.1f}%\n\n")
        
        f.write("指標統計:\n")
        f.write("-" * 50 + "\n")
        f.write(f"取樣率: {results_df['sample_rate'].mean():.1f} Hz (標準: 16000 Hz)\n")
        f.write(f"靜音比例: {results_df['silence_ratio'].mean() * 100:.1f}% (標準: < 30%)\n")
        f.write(f"最長靜音段: {results_df['max_silence_duration'].mean():.2f} 秒 (標準: < 1秒)\n")
        f.write(f"RMS音量: {results_df['rms_dbfs'].mean():.1f} dBFS (標準: > -30 dBFS)\n")
        f.write(f"峰值音量: {results_df['peak_dbfs'].mean():.1f} dBFS (標準: < 0 dBFS)\n")
        f.write(f"音量變異係數: {results_df['rms_cv'].mean():.3f} (標準: < 0.5)\n")
        f.write(f"信噪比: {results_df['snr'].mean():.1f} dB (標準: ≥ 20 dB)\n\n")
        
        f.write("不合格原因分析:\n")
        f.write("-" * 50 + "\n")
        format_fail = total_files - format_pass
        silence_fail = total_files - silence_pass
        volume_fail = total_files - volume_pass
        stability_fail = total_files - stability_pass
        snr_fail = total_files - snr_pass
        
        f.write(f"錄音格式不合格: {format_fail} 個文件 ({format_fail / total_files * 100:.1f}%)\n")
        f.write(f"靜音檢測不合格: {silence_fail} 個文件 ({silence_fail / total_files * 100:.1f}%)\n")
        f.write(f"音量範圍不合格: {volume_fail} 個文件 ({volume_fail / total_files * 100:.1f}%)\n")
        f.write(f"音量穩定性不合格: {stability_fail} 個文件 ({stability_fail / total_files * 100:.1f}%)\n")
        f.write(f"信噪比不合格: {snr_fail} 個文件 ({snr_fail / total_files * 100:.1f}%)\n")
    
    print(f"摘要報告已保存至: {os.path.join(output_dir, 'summary_report.txt')}")
    
    # 生成視覺化
    generate_visualizations(results_df, output_dir)

def generate_visualizations(results_df, output_dir):
    """
    生成視覺化圖表
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - output_dir: 輸出目錄
    """
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    # 設置風格
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. 合格率柱狀圖
    plt.figure(figsize=(12, 6))
    
    categories = ['錄音格式', '靜音檢測', '音量範圍', '音量穩定性', '信噪比', '整體評估']
    pass_rates = [
        results_df['format_ok'].mean() * 100,
        results_df['silence_ok'].mean() * 100,
        results_df['volume_ok'].mean() * 100,
        results_df['stability_ok'].mean() * 100,
        results_df['snr_ok'].mean() * 100,
        results_df['assessment_ok'].mean() * 100
    ]
    
    bars = plt.bar(categories, pass_rates, color='skyblue')
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('各評估指標合格率')
    plt.ylabel('合格率 (%)')
    plt.ylim(0, 110)  # 設置y軸範圍，留出空間顯示標籤
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'pass_rates.png'), dpi=300)
    plt.close()
    
    # 2. 靜音比例分布
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['silence_ratio'] * 100, bins=20, kde=True)
    plt.axvline(x=30, color='r', linestyle='--', label='閾值 (30%)')
    plt.title('靜音比例分布')
    plt.xlabel('靜音比例 (%)')
    plt.ylabel('文件數量')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'silence_ratio_distribution.png'), dpi=300)
    plt.close()
    
    # 3. RMS音量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['rms_dbfs'], bins=20, kde=True)
    plt.axvline(x=-30, color='r', linestyle='--', label='閾值 (-30 dBFS)')
    plt.title('RMS音量分布')
    plt.xlabel('RMS音量 (dBFS)')
    plt.ylabel('文件數量')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'rms_distribution.png'), dpi=300)
    plt.close()
    
    # 4. 信噪比分布
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['snr'], bins=20, kde=True)
    plt.axvline(x=20, color='r', linestyle='--', label='閾值 (20 dB)')
    plt.title('信噪比分布')
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('文件數量')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'snr_distribution.png'), dpi=300)
    plt.close()
    
    # 5. 評估指標箱形圖
    plt.figure(figsize=(14, 10))
    
    # 靜音比例
    plt.subplot(2, 2, 1)
    sns.boxplot(y=results_df['silence_ratio'] * 100)
    plt.axhline(y=30, color='r', linestyle='--')
    plt.title('靜音比例')
    plt.ylabel('靜音比例 (%)')
    
    # RMS音量
    plt.subplot(2, 2, 2)
    sns.boxplot(y=results_df['rms_dbfs'])
    plt.axhline(y=-30, color='r', linestyle='--')
    plt.title('RMS音量')
    plt.ylabel('RMS音量 (dBFS)')
    
    # 音量變異係數
    plt.subplot(2, 2, 3)
    sns.boxplot(y=results_df['rms_cv'])
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('音量變異係數')
    plt.ylabel('變異係數')
    
    # 信噪比
    plt.subplot(2, 2, 4)
    sns.boxplot(y=results_df['snr'])
    plt.axhline(y=20, color='r', linestyle='--')
    plt.title('信噪比')
    plt.ylabel('信噪比 (dB)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'metrics_boxplot.png'), dpi=300)
    plt.close()
    
    # 6. 評估結果表格
    plt.figure(figsize=(12, 8))
    
    # 創建表格數據
    table_data = [
        ['評估指標', '標準閾值', '平均值', '合格率'],
        ['錄音格式', '16 kHz, 16-bit, 單聲道', f"{results_df['sample_rate'].mean():.0f} Hz", f"{results_df['format_ok'].mean() * 100:.1f}%"],
        ['靜音比例', '< 30%', f"{results_df['silence_ratio'].mean() * 100:.1f}%", f"{results_df['silence_ok'].mean() * 100:.1f}%"],
        ['最長靜音段', '< 1秒', f"{results_df['max_silence_duration'].mean():.2f}秒", '-'],
        ['RMS音量', '> -30 dBFS', f"{results_df['rms_dbfs'].mean():.1f} dBFS", f"{results_df['volume_ok'].mean() * 100:.1f}%"],
        ['峰值音量', '< 0 dBFS', f"{results_df['peak_dbfs'].mean():.1f} dBFS", '-'],
        ['音量穩定性', '變異係數 < 0.5', f"{results_df['rms_cv'].mean():.3f}", f"{results_df['stability_ok'].mean() * 100:.1f}%"],
        ['信噪比', '≥ 20 dB', f"{results_df['snr'].mean():.1f} dB", f"{results_df['snr_ok'].mean() * 100:.1f}%"],
        ['整體評估', '所有指標合格', '-', f"{results_df['assessment_ok'].mean() * 100:.1f}%"]
    ]
    
    # 創建表格
    table = plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # 設置表格樣式
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表頭
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('darkblue')
        elif j == 0:  # 第一列
            cell.set_text_props(weight='bold')
            cell.set_facecolor('lightblue')
        elif i % 2 == 0:  # 偶數行
            cell.set_facecolor('whitesmoke')
    
    plt.axis('off')
    plt.title('音檔質量評估結果摘要', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'assessment_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"視覺化圖表已保存至: {vis_dir}")

def main(base_dir, output_dir):
    """
    主函數
    
    參數:
    - base_dir: 基礎目錄路徑
    - output_dir: 輸出目錄路徑
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 搜尋所有.wav文件
    print("搜尋音檔...")
    wav_files = find_wav_files(base_dir)
    print(f"找到 {len(wav_files)} 個音檔")
    
    # 批量評估
    print("評估音檔...")
    results_df = batch_assess_audio(wav_files, output_dir)
    
    # 生成摘要報告
    print("生成摘要報告...")
    generate_summary_report(results_df, output_dir)
    
    print("評估完成!")
    print(f"所有結果已保存至: {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "."  # 當前目錄
    
    output_dir = "audio_data_collection/format_assessment_results"
    
    main(base_dir, output_dir)
