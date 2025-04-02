import os
import json
import numpy as np
import librosa
import time
from scipy.signal import find_peaks, medfilt
from typing import Dict, Tuple, List
import traceback
import sys
import logging
import torch
import torchaudio
import glob
import soundfile as sf
from scipy import ndimage

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BaselineVADProcessor:
    def __init__(self):
        """初始化VAD處理器"""
        # 預加載模型以減少運行時延遲
        self.model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty")
        self.sample_rate = 16000
        self.hop_length = 160  # 10ms
        self.min_silence_duration = 0.1
        self.min_speech_duration = 0.1
        self.output_dir = "vad_results"
        
        # 預計算常用值以減少運行時計算
        self.window_size = 3  # 減小窗口大小
        self.threshold = 0.3
        self.margin = 0.025  # 減小邊界擴展
        
        # 預分配緩衝區
        self.prob_buffer = None
        self.audio_buffer = None

    def load_model(self):
        """加載VAD模型"""
        try:
            from speechbrain.pretrained import VAD
            self.model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir="pretrained_models/vad-crdnn-libriparty"
            )
            self.model.to(self.device)
            logging.info("成功加載VAD模型")
        except Exception as e:
            logging.error(f"加載VAD模型時出錯: {str(e)}")
            logging.error(traceback.format_exc())
            self.model = None
        
    def calculate_metrics(self, audio_path: str, reference_path: str = None) -> Dict:
        """計算音頻文件的VAD指標"""
        try:
            # 加載音頻
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 確保音頻是單聲道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 檢測語音區域
            speech_regions = self._detect_speech_regions(waveform, sample_rate)
            
            # 計算基本統計信息
            total_duration = waveform.shape[1] / sample_rate
            total_speech_duration = sum(end - start for start, end in speech_regions)
            total_silence_duration = total_duration - total_speech_duration
            speech_percentage = (total_speech_duration / total_duration * 100) if total_duration > 0 else 0
            
            # 計算處理延遲
            processing_delay = self._calculate_processing_delay(waveform, sample_rate)
            
            # 計算SNR
            snr = self.calculate_snr(waveform.numpy().flatten(), speech_regions)
            
            # 構建結果字典
            result = {
                "speech_regions": speech_regions,
                "statistics": {
                    "total_duration": total_duration,
                    "speech_duration": total_speech_duration,
                    "silence_duration": total_silence_duration,
                    "speech_percentage": speech_percentage,
                    "number_of_segments": len(speech_regions)
                },
                "metrics": {
                    "processing_delay": processing_delay,
                    "snr": snr
                },
                "method": "baseline"
            }
            
            # 如果有參考數據，添加準確度指標
            if reference_path and os.path.exists(reference_path):
                try:
                    with open(reference_path, 'r', encoding='utf-8') as f:
                        reference_data = json.load(f)
                    
                    file_name = os.path.basename(audio_path)
                    if file_name in reference_data:
                        ref_regions = reference_data[file_name]
                        accuracy_metrics = self._calculate_accuracy_metrics(speech_regions, ref_regions)
                        result["metrics"].update(accuracy_metrics)
                except Exception as e:
                    logging.error(f"讀取參考數據時出錯: {str(e)}")
            
            return result
            
        except Exception as e:
            logging.error(f"計算指標時發生錯誤: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """使用RMS正規化音頻"""
        target_rms = 0.1
        current_rms = np.sqrt(np.mean(y ** 2))
        if current_rms > 0:
            y = y * (target_rms / current_rms)
        return y
    
    def calculate_snr(self, audio_data, speech_segments):
        """計算信噪比（SNR）。
        
        Args:
            audio_data: 音頻數據
            speech_segments: 語音片段列表，每個元素為(start_time, end_time)元組
            
        Returns:
            float: SNR值（分貝）
        """
        try:
            # 1. 音頻預處理
            audio_data = audio_data.astype(np.float32)
            # 正規化到 [-1, 1]
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 2. 使用短時能量分析
            frame_length = int(0.025 * self.sample_rate)  # 25ms 窗口
            frame_step = int(0.010 * self.sample_rate)    # 10ms 步長
            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=frame_step)
            
            # 3. 應用漢明窗
            window = np.hamming(frame_length)
            frames = frames * window[:, np.newaxis]
            
            # 4. 計算每幀能量
            energy = np.sum(frames ** 2, axis=0)
            
            # 5. 動態閾值設置
            # 使用能量分佈的10%分位數作為閾值
            energy_threshold = np.percentile(energy, 10)
            
            # 6. 分離語音和噪音能量
            speech_energy = []
            noise_energy = []
            
            for i in range(len(energy)):
                time = i * frame_step / self.sample_rate
                is_speech = False
                
                for start, end in speech_segments:
                    if start <= time <= end:
                        is_speech = True
                        break
                
                if energy[i] > energy_threshold:  # 只考慮超過閾值的能量
                    if is_speech:
                        speech_energy.append(energy[i])
                    else:
                        noise_energy.append(energy[i])
            
            # 7. 計算平均能量
            if len(speech_energy) > 0 and len(noise_energy) > 0:
                avg_speech_energy = np.mean(speech_energy)
                avg_noise_energy = np.mean(noise_energy)
                
                # 確保不會出現零能量
                avg_speech_energy = max(avg_speech_energy, 1e-6)
                avg_noise_energy = max(avg_noise_energy, 1e-6)
                
                # 計算SNR
                snr = 10 * np.log10(avg_speech_energy / avg_noise_energy)
                
                # 限制SNR的範圍
                snr = max(min(snr, 40), -20)  # 限制在 -20dB 到 40dB 之間
                return snr
            
            return -20.0  # 如果無法計算，返回最小SNR值
            
        except Exception as e:
            print(f"計算SNR時發生錯誤: {str(e)}")
            return -20.0  # 發生錯誤時返回最小SNR值
    
    def _detect_speech_regions(self, audio_data: torch.Tensor, sr: int) -> List[Tuple[float, float]]:
        """使用優化的語音區段檢測"""
        try:
            if self.model is None:
                raise ValueError("VAD模型未正確加載")
            
            # 重採樣（如果需要）
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio_data = resampler(audio_data)
            
            # 使用批處理進行預測
            speech_prob = self.model.get_speech_prob_chunk(audio_data)
            
            # 使用向量化操作替代循環
            speech_prob = speech_prob.numpy().squeeze()
            
            # 使用快速卷積進行平滑
            weights = np.ones(self.window_size) / self.window_size
            speech_prob_smooth = np.convolve(speech_prob, weights, mode='same')
            
            # 使用向量化操作進行閾值處理
            speech_frames = speech_prob_smooth > self.threshold
            
            # 使用向量化操作找到語音區段
            changes = np.diff(speech_frames.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            
            # 處理邊界情況
            if speech_frames[0]:
                starts = np.concatenate(([0], starts))
            if speech_frames[-1]:
                ends = np.concatenate((ends, [len(speech_frames)]))
            
            # 轉換為時間並應用最小持續時間約束
            min_frames = int(self.min_speech_duration * self.sample_rate / self.hop_length)
            speech_regions = []
            
            for start, end in zip(starts, ends):
                if end - start >= min_frames:
                    # 添加邊界但避免額外計算
                    start_time = max(0, (start - 1) * self.hop_length / self.sample_rate)
                    end_time = min(len(audio_data[0])/self.sample_rate, 
                                 (end + 1) * self.hop_length / self.sample_rate)
                    speech_regions.append((start_time, end_time))
            
            # 合併接近的區段
            if not speech_regions:
                return []
            
            merged = []
            current_start, current_end = speech_regions[0]
            
            for start, end in speech_regions[1:]:
                if start - current_end <= self.min_silence_duration:
                    current_end = end
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged.append((current_start, current_end))
            
            return merged
            
        except Exception as e:
            print(f"語音區段檢測時發生錯誤: {str(e)}")
            return []
    
    def _calculate_accuracy_metrics(self, speech_regions: List[Tuple[float, float]], ref_regions: List[Tuple[float, float]]) -> Dict:
        """計算準確率相關指標"""
        try:
            # 計算重疊區域
            total_overlap = 0
            for pred_start, pred_end in speech_regions:
                for ref_start, ref_end in ref_regions:
                    overlap_start = max(pred_start, ref_start)
                    overlap_end = min(pred_end, ref_end)
                    if overlap_end > overlap_start:
                        total_overlap += overlap_end - overlap_start
            
            # 計算總語音時長
            total_pred_duration = sum(end - start for start, end in speech_regions)
            total_ref_duration = sum(end - start for start, end in ref_regions)
            
            # 計算指標
            precision = total_overlap / total_pred_duration if total_pred_duration > 0 else 0
            recall = total_overlap / total_ref_duration if total_ref_duration > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'reference_duration': total_ref_duration,
                'predicted_duration': total_pred_duration,
                'overlap_duration': total_overlap
            }
            
        except Exception as e:
            logging.error(f"計算準確率指標時出錯: {str(e)}")
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'reference_duration': 0,
                'predicted_duration': 0,
                'overlap_duration': 0
            }
    
    def _calculate_processing_delay(self, audio_data: torch.Tensor, sr: int) -> float:
        """優化的處理延遲計算"""
        start_time = time.time()
        _ = self._detect_speech_regions(audio_data, sr)
        return (time.time() - start_time) * 1000  # 轉換為毫秒

    def process_audio_files(self, audio_dir: str, reference_path: str = None) -> Tuple[int, List[str]]:
        """處理目錄中的所有音頻文件"""
        try:
            # 確保輸出目錄存在並有寫入權限
            os.makedirs(self.output_dir, exist_ok=True)
            os.chmod(self.output_dir, 0o755)  # 設置目錄權限
            logging.info(f"輸出目錄已創建/確認: {self.output_dir}")
            
            # 檢查音頻目錄是否存在
            if not os.path.exists(audio_dir):
                raise FileNotFoundError(f"音頻目錄不存在: {audio_dir}")
            
            # 獲取所有WAV文件
            wav_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                        if f.endswith('.wav')]
            
            if not wav_files:
                logging.warning(f"在 {audio_dir} 中未找到WAV文件")
                return 0, []
            
            print(f"找到 {len(wav_files)} 個音檔待處理\n")
            logging.info(f"找到 {len(wav_files)} 個音檔待處理")
            
            success_count = 0
            failed_files = []
            
            # 處理每個文件
            for i, audio_file in enumerate(wav_files, 1):
                try:
                    print(f"\n處理進度: [{i}/{len(wav_files)}] {os.path.basename(audio_file)}")
                    logging.info(f"開始處理: {os.path.basename(audio_file)}")
                    
                    # 檢查輸入文件是否存在
                    if not os.path.exists(audio_file):
                        raise FileNotFoundError(f"找不到音頻文件: {audio_file}")
                    
                    # 計算指標
                    result = self.calculate_metrics(audio_file, reference_path)
                    if result is None:
                        raise Exception("指標計算失敗")
                    
                    # 保存結果到JSON文件
                    output_file = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_vad.json")
                    try:
                        # 確保輸出目錄存在
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        # 寫入結果
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=4, ensure_ascii=False)
                        
                        # 設置文件權限
                        os.chmod(output_file, 0o644)
                        
                        # 驗證文件是否成功創建和寫入
                        if not os.path.exists(output_file):
                            raise Exception(f"結果文件未創建: {output_file}")
                        
                        if os.path.getsize(output_file) == 0:
                            raise Exception(f"結果文件為空: {output_file}")
                            
                        logging.info(f"結果已保存到: {output_file}")
                        
                    except Exception as e:
                        raise Exception(f"保存結果文件時出錯: {str(e)}")
                    
                    success_count += 1
                    print(f"✓ 成功處理: {os.path.basename(audio_file)}")
                    logging.info(f"成功處理: {os.path.basename(audio_file)}")
                    
                except Exception as e:
                    error_msg = f"處理 {os.path.basename(audio_file)} 時出錯: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    failed_files.append(os.path.basename(audio_file))
                    print(f"✗ 處理失敗: {os.path.basename(audio_file)}")
            
            # 保存處理摘要
            summary = {
                'total_files': len(wav_files),
                'success_count': success_count,
                'failed_count': len(failed_files),
                'failed_files': failed_files
            }
            
            summary_path = os.path.join(self.output_dir, 'summary.json')
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=4, ensure_ascii=False)
                os.chmod(summary_path, 0o644)  # 設置文件權限
                logging.info(f"處理摘要已保存到: {summary_path}")
            except Exception as e:
                logging.error(f"保存處理摘要時出錯: {str(e)}")
            
            print(f"\n處理完成!")
            print(f"成功: {success_count}/{len(wav_files)}")
            print(f"失敗: {len(failed_files)}/{len(wav_files)}")
            
            return success_count, failed_files
            
        except Exception as e:
            logging.error(f"處理音頻文件時發生錯誤: {str(e)}")
            logging.error(traceback.format_exc())
            return 0, []

    def process_and_evaluate(self, audio_file, reference_file=None):
        """處理音頻文件並評估結果。
        
        Args:
            audio_file: 音頻文件路徑
            reference_file: 參考文件路徑（可選）
            
        Returns:
            dict: 包含處理結果和評估指標的字典
        """
        try:
            # 加載音頻
            waveform, sr = torchaudio.load(audio_file)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重採樣到目標採樣率
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 檢測語音區域
            speech_regions = self._detect_speech_regions(waveform, self.sample_rate)
            
            # 計算 SNR
            audio_data = waveform.numpy().flatten()
            snr = self.calculate_snr(audio_data, speech_regions)
            
            # 計算處理延遲
            processing_delay = self._calculate_processing_delay(waveform, self.sample_rate)
            
            # 準備結果字典
            result = {
                "file_name": os.path.basename(audio_file),
                "speech_regions": speech_regions,
                "metrics": {
                    "snr": float(snr),
                    "processing_delay": float(processing_delay)
                }
            }
            
            # 如果有參考文件，計算準確度指標
            if reference_file and os.path.exists(reference_file):
                try:
                    with open(reference_file, 'r', encoding='utf-8') as f:
                        reference_data = json.load(f)
                    
                    file_name = os.path.basename(audio_file)
                    if file_name in reference_data:
                        ref_regions = reference_data[file_name]
                        accuracy_metrics = self._calculate_accuracy_metrics(speech_regions, ref_regions)
                        result["metrics"].update(accuracy_metrics)
                except Exception as e:
                    logging.error(f"讀取參考數據時出錯: {str(e)}")
            
            # 保存結果
            output_file = os.path.join(
                self.output_dir,
                os.path.splitext(os.path.basename(audio_file))[0] + "_vad_result.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result
            
        except Exception as e:
            logging.error(f"處理文件 {audio_file} 時出錯: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def process_directory(self):
        """處理目錄中的所有音頻文件。"""
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 獲取所有音頻文件
        audio_files = glob.glob(os.path.join(self.audio_dir, '*.wav'))
        print(f"\n找到 {len(audio_files)} 個音檔待處理\n")
        
        # 處理每個文件
        for i, audio_file in enumerate(audio_files, 1):
            base_name = os.path.basename(audio_file)
            print(f"\n處理進度: [{i}/{len(audio_files)}] {base_name}")
            
            # 構建參考標註文件路徑
            reference_file = os.path.join(self.reference_dir, base_name.replace('.wav', '.json'))
            
            # 處理並評估
            result = self.process_and_evaluate(audio_file, reference_file)
            
            if result is not None:
                # 保存結果
                output_file = os.path.join(self.output_dir, base_name.replace('.wav', '.json'))
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"✓ 成功處理: {base_name}")
            else:
                print(f"✗ 處理失敗: {base_name}")
            
        print("\n處理完成!")
        print(f"成功: {len(glob.glob(os.path.join(self.output_dir, '*.json')))}/{len(audio_files)}")
        print(f"失敗: {len(audio_files) - len(glob.glob(os.path.join(self.output_dir, '*.json')))}/{len(audio_files)}")

    def process_audio(self, audio_file):
        try:
            # 讀取音頻文件
            audio_data, sample_rate = sf.read(audio_file)
            
            # 確保音頻是單聲道
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # 使用VAD模型檢測語音
            speech_prob = self.model.get_speech_prob_chunk(audio_data)
            speech_segments = self.model.get_speech_segments_from_probs(speech_prob)
            
            # 使用改進的SNR計算方法
            snr = self.calculate_snr(audio_data, speech_segments)
            
            # 計算統計信息
            total_duration = len(audio_data) / sample_rate
            speech_duration = sum(end - start for start, end in speech_segments)
            silence_duration = total_duration - speech_duration
            
            # 返回結果
            return {
                'audio_file': os.path.basename(audio_file),
                'speech_segments': speech_segments.tolist(),
                'speech_probs': speech_prob.tolist(),
                'stats': {
                    'total_duration': total_duration,
                    'speech_duration': speech_duration,
                    'silence_duration': silence_duration,
                    'snr': float(snr)
                }
            }
            
        except Exception as e:
            print(f"處理音頻文件時發生錯誤: {str(e)}")
            return None

if __name__ == "__main__":
    # 設置參數
    audio_dir = "02_audio_noise_reducer_experiment/output"
    ground_truth_path = "ground_truth.json"
    
    # 創建處理器實例
    processor = BaselineVADProcessor()
    
    # 處理音頻文件
    success_count, failed_files = processor.process_audio_files(
        audio_dir=audio_dir,
        reference_path=ground_truth_path
    )
    
    # 顯示最終結果
    if success_count > 0:
        logging.info("處理完成！")
        if failed_files:
            logging.warning(f"有 {len(failed_files)} 個文件處理失敗")
            for failed_file in failed_files:
                logging.warning(f"失敗的文件: {failed_file}")
    else:
        logging.error("處理失敗：沒有成功處理任何文件")
        sys.exit(1) 