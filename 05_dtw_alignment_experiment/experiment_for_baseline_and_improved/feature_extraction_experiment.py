import os
import numpy as np
import librosa
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
from datetime import datetime
from tqdm import tqdm

class BaselineFeatureExtractor:
    def __init__(self, sr=22050, frame_length=2048, hop_length=512):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def extract_features(self, audio_path: str) -> Dict:
        """基準版本的特徵提取方法"""
        # 加載音頻
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # 1. 基頻特徵 (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # 2. MFCC特徵
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 3. 能量特徵
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, 
                                hop_length=self.hop_length)
        
        # 4. 過零率
        zcr = librosa.feature.zero_crossing_rate(y)
        
        return {
            'f0': f0.tolist(),
            'voiced_flag': voiced_flag.tolist(),
            'mfcc': mfcc.tolist(),
            'rms': rms.tolist(),
            'zcr': zcr.tolist(),
            'start_points': [],  # 基準版本不計算切分點
            'end_points': [],
            'alignments': []
        }

class ImprovedFeatureExtractor:
    def __init__(self, sr=22050, frame_length=2048, hop_length=512,
                 n_mfcc=13, n_mels=128, f0_min=50, f0_max=1000):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    def preprocess_audio(self, y: np.ndarray) -> np.ndarray:
        """改進的音頻預處理"""
        # 1. 預加重
        y_pre = librosa.effects.preemphasis(y)
        
        # 2. 去噪
        y_denoised = librosa.decompose.nn_filter(y_pre,
                                               aggregate=np.median,
                                               metric='cosine')
        
        # 3. 音量歸一化
        y_normalized = librosa.util.normalize(y_denoised)
        
        return y_normalized
    
    def extract_features(self, audio_path: str) -> Dict:
        """改進版本的特徵提取方法"""
        # 加載音頻
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # 預處理
        y_processed = self.preprocess_audio(y)
        
        # 1. 增強的基頻檢測
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_processed,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # 2. 改進的MFCC特徵
        mel_spec = librosa.feature.melspectrogram(
            y=y_processed, 
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfcc = librosa.feature.mfcc(
            S=mel_spec_db,
            n_mfcc=self.n_mfcc
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 3. 多維度能量特徵
        rms = librosa.feature.rms(
            y=y_processed,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # 4. 頻譜特徵
        spec_cent = librosa.feature.spectral_centroid(
            y=y_processed,
            sr=sr,
            hop_length=self.hop_length
        )
        spec_bw = librosa.feature.spectral_bandwidth(
            y=y_processed,
            sr=sr,
            hop_length=self.hop_length
        )
        
        # 5. 語音活動檢測（VAD）
        zcr = librosa.feature.zero_crossing_rate(y_processed)
        voiced_segments = librosa.effects.split(
            y_processed,
            top_db=30,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # 計算切分點
        start_points = [seg[0]/sr for seg in voiced_segments]
        end_points = [seg[1]/sr for seg in voiced_segments]
        
        # 計算對齊點（這裡使用語音段的中點作為對齊點）
        alignments = [(start + end)/2 for start, end in zip(start_points, end_points)]
        
        return {
            'f0': f0.tolist(),
            'voiced_flag': voiced_flag.tolist(),
            'mfcc': mfcc.tolist(),
            'mfcc_delta': mfcc_delta.tolist(),
            'mfcc_delta2': mfcc_delta2.tolist(),
            'rms': rms.tolist(),
            'spectral_centroid': spec_cent.tolist(),
            'spectral_bandwidth': spec_bw.tolist(),
            'zcr': zcr.tolist(),
            'start_points': start_points,
            'end_points': end_points,
            'alignments': alignments
        }

def process_directory(input_dir: str, output_dir: str, method: str):
    """處理整個目錄的音頻文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 選擇特徵提取器
    if method == 'baseline':
        extractor = BaselineFeatureExtractor()
    else:
        extractor = ImprovedFeatureExtractor()
    
    results = []
    audio_files = list(input_path.glob('*.wav'))
    
    for audio_file in tqdm(audio_files, desc=f"處理{method}版本"):
        try:
            start_time = time.time()
            features = extractor.extract_features(str(audio_file))
            processing_time = time.time() - start_time
            
            result = {
                'file_name': audio_file.name,
                'processing_time': processing_time,
                **features
            }
            results.append(result)
            
        except Exception as e:
            print(f"處理文件 {audio_file} 時出錯: {str(e)}")
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{method}_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return output_file

def main():
    base_dir = Path("/Users/chris/Desktop/數據紀錄日誌-處理/05_dtw_alignment_experiment")
    experiment_dir = base_dir / "experiment_for_baseline_and_improved"
    
    # 處理教師音頻
    print("\n處理教師音頻...")
    teacher_audio_dir = base_dir / "preprocess_teacher_audio"
    process_directory(str(teacher_audio_dir), 
                     str(experiment_dir), 
                     'baseline')
    process_directory(str(teacher_audio_dir), 
                     str(experiment_dir), 
                     'improved')
    
    # 處理學生音頻
    print("\n處理學生音頻...")
    student_audio_dir = base_dir / "preprocess_student_audio"
    process_directory(str(student_audio_dir), 
                     str(experiment_dir), 
                     'baseline')
    process_directory(str(student_audio_dir), 
                     str(experiment_dir), 
                     'improved')
    
    print("\n實驗完成！結果已保存到", str(experiment_dir))

if __name__ == "__main__":
    main() 