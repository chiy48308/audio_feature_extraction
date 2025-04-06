import os
import json
import csv
import numpy as np
import librosa
from typing import Dict, List, Any, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def get_audio_files(directory: str) -> List[str]:
    """獲取目錄中的所有音頻文件"""
    try:
        audio_files = [os.path.join(directory, f) for f in os.listdir(directory)
                      if f.endswith(('.wav', '.mp3'))]
        return audio_files
    except FileNotFoundError:
        print(f"錯誤：找不到目錄 {directory}")
        return []
    except Exception as e:
        print(f"讀取目錄時出錯：{str(e)}")
        return []

def extract_features(audio_file: str) -> Optional[Dict[str, Any]]:
    """從音頻文件中提取特徵"""
    try:
        # 讀取音頻文件，重採樣到 16kHz 以加快處理
        y, sr = librosa.load(audio_file, sr=16000, duration=10.0)  # 限制處理時長為 10 秒
        
        # 簡化的預處理
        y = librosa.util.normalize(y)
        
        # 提取 MFCC 特徵，使用更高效的參數
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr,
            n_mfcc=13,
            n_fft=400,
            hop_length=160,
            window='hamming'
        )
        
        # 只計算一階 delta 特徵
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_combined = np.vstack([mfcc, mfcc_delta])
        
        mfcc_mean = float(np.mean(mfcc_combined))
        mfcc_std = float(np.std(mfcc_combined))
        mfcc_stability = abs(mfcc_std) < 30
        
        # 使用更高效的基頻提取參數
        f0, voiced_flag, _ = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=400,
            hop_length=160
        )
        f0_missing_rate = float(np.sum(np.isnan(f0)) / len(f0))
        f0_quality = f0_missing_rate < 0.5
        
        # 計算能量特徵
        energy = np.sum(y**2)
        energy_mean = float(np.mean(energy))
        energy_std = float(np.std(y**2))
        energy_stability = energy_std < energy_mean * 0.5
        
        # 計算過零率
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=400,
            hop_length=160
        )
        zcr_mean = float(np.mean(zcr))
        zcr_rationality = 0.05 <= zcr_mean <= 0.5
        
        feature_integrity = not (
            np.isnan(mfcc_mean) or 
            np.isnan(mfcc_std) or 
            np.isnan(energy_mean) or 
            np.isnan(energy_std) or 
            np.isnan(zcr_mean)
        )
        
        return {
            'file': os.path.basename(audio_file),
            'mfcc_mean': float(mfcc_mean),
            'mfcc_std': float(mfcc_std),
            'mfcc_stability': bool(mfcc_stability),
            'f0_missing_rate': float(f0_missing_rate),
            'f0_quality': bool(f0_quality),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'energy_stability': bool(energy_stability),
            'zcr_mean': float(zcr_mean),
            'zcr_rationality': bool(zcr_rationality),
            'feature_integrity': bool(feature_integrity)
        }
    except Exception as e:
        print(f"提取特徵時出錯 {audio_file}: {str(e)}")
        return None

def process_audio_file(audio_file: str) -> Optional[Dict[str, Any]]:
    """處理單個音頻文件的包裝函數"""
    try:
        return extract_features(audio_file)
    except Exception as e:
        print(f"處理文件 {audio_file} 時出錯: {str(e)}")
        return None

def calculate_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """計算特徵統計摘要"""
    if not results:
        return {}
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {}
    
    # 提取各個特徵的值
    mfcc_means = [float(r['mfcc_mean']) for r in valid_results]
    mfcc_stds = [float(r['mfcc_std']) for r in valid_results]
    f0_missing_rates = [float(r['f0_missing_rate']) for r in valid_results]
    energy_means = [float(r['energy_mean']) for r in valid_results]
    energy_stds = [float(r['energy_std']) for r in valid_results]
    zcr_means = [float(r['zcr_mean']) for r in valid_results]
    
    # 計算布爾值特徵的統計
    mfcc_stability_rate = float(sum(1 for r in valid_results if r['mfcc_stability'] is True) / len(valid_results))
    f0_quality_rate = float(sum(1 for r in valid_results if r['f0_quality'] is True) / len(valid_results))
    energy_stability_rate = float(sum(1 for r in valid_results if r['energy_stability'] is True) / len(valid_results))
    zcr_rationality_rate = float(sum(1 for r in valid_results if r['zcr_rationality'] is True) / len(valid_results))
    feature_integrity_rate = float(sum(1 for r in valid_results if r['feature_integrity'] is True) / len(valid_results))
    
    # 計算統計數據
    summary = {
        'file_count': int(len(valid_results)),
        'mfcc_mean_range': f"{float(min(mfcc_means)):.3f} to {float(max(mfcc_means)):.3f}",
        'mfcc_std_range': f"{float(min(mfcc_stds)):.3f} to {float(max(mfcc_stds)):.3f}",
        'mfcc_stability_rate': f"{mfcc_stability_rate * 100:.2f}%",
        'f0_missing_rate_avg': f"{float(sum(f0_missing_rates) / len(f0_missing_rates)) * 100:.2f}%",
        'f0_quality_rate': f"{f0_quality_rate * 100:.2f}%",
        'energy_mean_range': f"{float(min(energy_means)):.2e} to {float(max(energy_means)):.2e}",
        'energy_std_range': f"{float(min(energy_stds)):.2e} to {float(max(energy_stds)):.2e}",
        'energy_stability_rate': f"{energy_stability_rate * 100:.2f}%",
        'zcr_mean_range': f"{float(min(zcr_means)):.3f} to {float(max(zcr_means)):.3f}",
        'zcr_rationality_rate': f"{zcr_rationality_rate * 100:.2f}%",
        'feature_integrity_rate': f"{feature_integrity_rate * 100:.2f}%"
    }
    
    return summary

def main():
    """主函數"""
    # 設定輸入和輸出路徑
    input_dir = 'organized_audio/student'
    output_dir = 'feature_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取所有音頻文件
    audio_files = get_audio_files(input_dir)
    if not audio_files:
        print("未找到音頻文件")
        return
    
    print(f"開始處理 {len(audio_files)} 個音頻文件...")
    
    # 使用多進程處理
    num_processes = max(1, cpu_count() - 1)  # 保留一個核心給系統
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_audio_file, audio_files),
            total=len(audio_files),
            desc="提取特徵"
        ))
    
    # 過濾掉 None 結果
    results = [r for r in results if r is not None]
    
    if not results:
        print("沒有成功提取任何特徵")
        return
    
    # 計算並保存統計摘要
    summary = calculate_summary_statistics(results)
    summary_file = os.path.join(output_dir, 'feature_evaluation_summary_student.csv')
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['指標', '值'])
        for key, value in summary.items():
            writer.writerow([key, value])
    
    # 保存詳細結果
    detailed_file = os.path.join(output_dir, 'feature_evaluation_detailed_student.json')
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"特徵提取完成。結果已保存到 {output_dir} 目錄")

if __name__ == '__main__':
    main() 