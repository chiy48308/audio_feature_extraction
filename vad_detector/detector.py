import torch
import numpy as np
from speechbrain.pretrained import VAD
from pathlib import Path
from typing import List, Dict, Union, Tuple

class VADDetector:
    def __init__(self,
                 window_size: int = 3,
                 threshold: float = 0.3,
                 min_duration: float = 0.1,
                 max_merge_gap: float = 0.3):
        """
        初始化VAD檢測器
        
        Args:
            window_size: 滑動窗口大小
            threshold: 語音檢測閾值
            min_duration: 最小語音片段長度（秒）
            max_merge_gap: 最大合併間隔（秒）
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_duration = min_duration
        self.max_merge_gap = max_merge_gap
        
        # 預加載模型
        self.model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty")

    def process_file(self, audio_path: Union[str, Path]) -> Dict:
        """
        處理單個音頻文件
        
        Args:
            audio_path: 音頻文件路徑
            
        Returns:
            Dict: 包含檢測結果的字典
        """
        # 加載音頻
        audio_path = str(audio_path)
        signal = self.model.load_audio(audio_path)
        
        # 檢測語音區段
        speech_probs = self.model.get_speech_prob_file(audio_path)
        
        # 使用向量化操作處理概率值
        speech_regions = self._detect_speech_regions(speech_probs)
        
        return {
            'speech_regions': speech_regions,
            'speech_probs': speech_probs.numpy(),
            'audio_length': len(signal) / self.model.sample_rate
        }

    def process_directory(self, directory: Union[str, Path]) -> Dict[str, Dict]:
        """
        批量處理目錄中的音頻文件
        
        Args:
            directory: 音頻文件目錄
            
        Returns:
            Dict: 檔案名稱到處理結果的映射
        """
        directory = Path(directory)
        results = {}
        
        for audio_file in directory.glob('*.wav'):
            try:
                results[audio_file.name] = self.process_file(audio_file)
            except Exception as e:
                print(f'處理文件 {audio_file} 時發生錯誤: {str(e)}')
                
        return results

    def _detect_speech_regions(self, speech_probs: torch.Tensor) -> List[Tuple[float, float]]:
        """
        檢測語音區段
        
        Args:
            speech_probs: 語音概率值
            
        Returns:
            List[Tuple[float, float]]: 語音區段的起止時間列表
        """
        # 使用滑動平均平滑概率值
        kernel = np.ones(self.window_size) / self.window_size
        smoothed_probs = np.convolve(speech_probs.numpy(), kernel, mode='same')
        
        # 使用向量化操作找到語音區段
        speech_frames = smoothed_probs > self.threshold
        changes = np.diff(speech_frames.astype(int))
        start_frames = np.where(changes == 1)[0] + 1
        end_frames = np.where(changes == -1)[0] + 1
        
        if len(start_frames) == 0:
            return []
            
        # 處理邊界情況
        if speech_frames[0]:
            start_frames = np.concatenate(([0], start_frames))
        if speech_frames[-1]:
            end_frames = np.concatenate((end_frames, [len(speech_frames)]))
            
        # 轉換為時間
        frame_duration = 0.01  # 10ms per frame
        regions = [(start * frame_duration, end * frame_duration)
                  for start, end in zip(start_frames, end_frames)
                  if (end - start) * frame_duration >= self.min_duration]
        
        # 合併接近的區段
        return self._merge_regions(regions)

    def _merge_regions(self, regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        合併接近的語音區段
        
        Args:
            regions: 語音區段列表
            
        Returns:
            List[Tuple[float, float]]: 合併後的語音區段列表
        """
        if not regions:
            return []
            
        merged = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= self.max_merge_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
                
        return merged 