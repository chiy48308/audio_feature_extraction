import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
import webrtcvad
import noisereduce as nr
import pyloudnorm as pyln

class AudioProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_directories()
        self.vad = webrtcvad.Vad(self.config['vad']['aggressiveness'])
        self.meter = pyln.Meter(self.config['audio']['sample_rate'])
    
    def setup_directories(self):
        """設置輸出目錄"""
        base_dir = Path(self.config['output']['base_dir'])
        for subdir in self.config['output']['subdirs'].values():
            (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def process_audio(self, audio_path):
        """處理單個音訊檔案，順序：正規化 -> 降噪 -> VAD"""
        try:
            # 讀取音訊
            audio, sr = librosa.load(str(audio_path), 
                                   sr=self.config['audio']['sample_rate'],
                                   mono=True)
            
            # 1. 音量正規化
            normalized = self.normalize_volume(audio)
            
            # 2. 降噪處理
            denoised = self.apply_noise_reduction(normalized)
            
            # 3. VAD 處理
            vad_segments = self.apply_vad(denoised)
            
            # 保存結果
            output_base = Path(self.config['output']['base_dir'])
            filename = Path(audio_path).stem
            
            # 保存處理後的音訊
            output_path = output_base / f"{filename}_processed.wav"
            sf.write(
                str(output_path),
                denoised,  # 保存最終處理後的音訊（包含正規化和降噪）
                self.config['audio']['sample_rate']
            )
            
            # 計算音量（處理短音訊的情況）
            try:
                loudness = self.meter.integrated_loudness(normalized)
            except ValueError:
                loudness = None
            
            # 返回處理結果
            return {
                'filename': filename,
                'vad_segments': vad_segments,
                'loudness': loudness,
                'duration': len(audio) / self.config['audio']['sample_rate'],
                'status': 'success'
            }
        except Exception as e:
            print(f"警告：處理檔案 {audio_path} 時發生錯誤：{str(e)}")
            return {
                'filename': Path(audio_path).stem,
                'vad_segments': [],
                'loudness': None,
                'duration': None,
                'status': 'error',
                'error_message': str(e)
            }
    
    def apply_noise_reduction(self, audio):
        """應用降噪處理"""
        if self.config['noise_reduction']['method'] == 'wiener':
            return nr.reduce_noise(
                y=audio,
                sr=self.config['audio']['sample_rate'],
                stationary=self.config['noise_reduction']['stationary'],
                prop_decrease=self.config['noise_reduction']['prop_decrease']
            )
        else:  # spectral subtraction
            return nr.reduce_noise(
                y=audio,
                sr=self.config['audio']['sample_rate'],
                n_fft=self.config['noise_reduction']['n_fft'],
                win_length=self.config['noise_reduction']['win_length'],
                hop_length=self.config['noise_reduction']['hop_length']
            )
    
    def apply_vad(self, audio):
        """應用語音活動檢測"""
        frame_duration = self.config['vad']['frame_duration']
        frame_size = int(self.config['audio']['sample_rate'] * frame_duration / 1000)
        frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=frame_size)
        
        segments = []
        for i, frame in enumerate(frames.T):
            is_speech = self.vad.is_speech(
                (frame * 32768).astype(np.int16).tobytes(),
                self.config['audio']['sample_rate']
            )
            if is_speech:
                start = i * frame_duration / 1000
                end = (i + 1) * frame_duration / 1000
                segments.append((start, end))
        
        # 合併相近的片段
        merged_segments = []
        if segments:
            current_start, current_end = segments[0]
            for start, end in segments[1:]:
                if start - current_end <= self.config['vad']['min_silence_duration']:
                    current_end = end
                else:
                    if (current_end - current_start) >= self.config['vad']['min_speech_duration']:
                        merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            if (current_end - current_start) >= self.config['vad']['min_speech_duration']:
                merged_segments.append((current_start, current_end))
        
        return merged_segments
    
    def normalize_volume(self, audio):
        """音量正規化"""
        try:
            loudness = self.meter.integrated_loudness(audio)
            normalized = pyln.normalize.loudness(
                audio,
                loudness,
                self.config['volume']['reference_level']
            )
        except ValueError as e:
            # 如果音訊太短，直接返回原始音訊
            print(f"警告：音訊太短，跳過音量正規化")
            normalized = audio
        return normalized

def main():
    # 設置處理器
    processor = AudioProcessor('config/experiment_config.yaml')
    
    # 處理老師音檔
    teacher_results = []
    teacher_dir = Path('organized_audio/teacher')
    for audio_file in tqdm(list(teacher_dir.glob('*.wav')), desc='處理老師音檔'):
        result = processor.process_audio(audio_file)
        result['speaker_type'] = 'teacher'
        teacher_results.append(result)
    
    # 處理學生音檔
    student_results = []
    student_dir = Path('organized_audio/student')
    for audio_file in tqdm(list(student_dir.glob('*.wav')), desc='處理學生音檔'):
        result = processor.process_audio(audio_file)
        result['speaker_type'] = 'student'
        student_results.append(result)
    
    # 合併結果並保存
    all_results = pd.DataFrame(teacher_results + student_results)
    all_results.to_csv(
        Path(processor.config['output']['base_dir']) / 'processing_results.csv',
        index=False,
        encoding='utf-8'
    )

if __name__ == '__main__':
    main() 