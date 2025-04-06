import os
import sys
import shutil
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

def process_audio_file(input_file, output_file):
    # 1. 讀取音頻
    audio, sr = librosa.load(input_file, sr=None)
    
    # 2. 音量校正
    audio_rms = np.sqrt(np.mean(audio**2))
    target_rms = 0.1  # 目標RMS值
    gain = target_rms / (audio_rms + 1e-6)
    normalized_audio = audio * gain
    
    # 3. 噪音消除
    # 使用頻譜減法進行簡單的噪音消除
    S = librosa.stft(normalized_audio)
    mag = np.abs(S)
    phase = np.angle(S)
    
    # 估計噪音profile（使用前幾幀）
    noise_profile = np.mean(mag[:, :10], axis=1, keepdims=True)
    
    # 頻譜減法
    mag_reduced = np.maximum(mag - noise_profile, 0)
    
    # 重建信號
    S_reduced = mag_reduced * np.exp(1j * phase)
    audio_denoised = librosa.istft(S_reduced)
    
    # 4. VAD檢測（簡單的能量閾值方法）
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    energy = librosa.feature.rms(y=audio_denoised, frame_length=frame_length, hop_length=hop_length)
    energy_threshold = np.mean(energy) * 0.5
    
    # 應用VAD mask
    frames_mask = energy > energy_threshold
    audio_vad = np.zeros_like(audio_denoised)
    
    for i, is_speech in enumerate(frames_mask[0]):
        if is_speech:
            start = i * hop_length
            end = start + frame_length
            if end <= len(audio_denoised):
                audio_vad[start:end] = audio_denoised[start:end]
    
    # 5. 保存處理後的音頻
    sf.write(output_file, audio_vad, sr)
    
    # 6. 提取特徵（作為示例，我們提取MFCC特徵）
    mfccs = librosa.feature.mfcc(y=audio_vad, sr=sr, n_mfcc=13)
    feature_file = output_file.replace('.wav', '_features.npy')
    np.save(feature_file, mfccs)

def process_directory(input_dir, output_dir):
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理所有.wav文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                # 保持相同的目錄結構
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # 確保輸出文件的目錄存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 處理音頻文件
                try:
                    print(f"處理文件: {input_path}")
                    process_audio_file(input_path, output_path)
                    
                    # 複製對應的文本文件（如果存在）
                    txt_file = input_path.replace('.wav', '.txt')
                    if os.path.exists(txt_file):
                        shutil.copy2(txt_file, output_path.replace('.wav', '.txt'))
                except Exception as e:
                    print(f"處理文件 {input_path} 時發生錯誤: {str(e)}")

if __name__ == "__main__":
    # 處理學生音頻
    process_directory('student_audio', 'preprocess_student_audio')
    
    # 處理老師音頻
    process_directory('teacher_audio', 'preprocess_teacher_audio')
    
    print("音頻處理完成！") 