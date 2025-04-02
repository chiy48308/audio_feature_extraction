import os
import json
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import VAD
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score
import librosa

class ImprovedVADProcessor:
    def __init__(self):
        self.vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty")
        self.min_silence_duration = 0.1  # 進一步降低最小靜音持續時間
        self.merge_threshold = 0.1  # 進一步降低合併閾值
        self.speech_threshold = 0.2  # 進一步降低語音閾值
        self.window_size = 64  # 增加窗口大小
        self.boundary_extension = 0.1  # 增加邊界擴展時間（秒）

    def normalize_volume(self, audio):
        # 使用RMS歸一化
        if np.all(audio == 0):
            return audio
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 0.1
        gain = target_rms / (rms + 1e-6)
        return audio * gain

    def calculate_snr(self, audio, sr):
        # 使用短時能量估計信號和噪音
        frame_length = int(0.025 * sr)  # 25ms幀
        hop_length = int(0.010 * sr)    # 10ms步長
        
        # 計算短時能量
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 使用百分位數區分信號和噪音
        noise_threshold = np.percentile(energy, 5)  # 降低噪音閾值
        signal_threshold = np.percentile(energy, 95)  # 提高信號閾值
        
        # 計算信噪比
        signal_energy = np.mean(energy[energy > signal_threshold]**2)
        noise_energy = np.mean(energy[energy < noise_threshold]**2)
        
        if noise_energy == 0:
            return 100.0  # 當沒有噪音時返回高SNR
        
        snr = 10 * np.log10(signal_energy / (noise_energy + 1e-10))
        return max(0, min(snr, 100))  # 限制SNR範圍

    def generate_reference_labels(self, audio, sr, smoothed_prob):
        # 計算多個特徵
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        # 1. 短時能量
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        energy = np.repeat(energy, len(smoothed_prob) // len(energy))
        
        # 2. 過零率
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = np.repeat(zcr, len(smoothed_prob) // len(zcr))
        
        # 3. 頻譜質心
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        cent = np.repeat(cent, len(smoothed_prob) // len(cent))
        
        # 4. 頻譜帶寬
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        bandwidth = np.repeat(bandwidth, len(smoothed_prob) // len(bandwidth))
        
        # 使用多特徵組合生成標籤
        energy_threshold = np.mean(energy) * 0.3  # 降低能量閾值
        zcr_threshold = np.mean(zcr) * 1.5
        cent_threshold = np.mean(cent) * 0.8
        bandwidth_threshold = np.mean(bandwidth) * 0.8
        
        # 組合特徵
        y_true = np.zeros(len(smoothed_prob))
        for i in range(len(smoothed_prob)):
            if (energy[i] > energy_threshold and 
                (zcr[i] > zcr_threshold or 
                 cent[i] > cent_threshold or 
                 bandwidth[i] > bandwidth_threshold)):
                y_true[i] = 1
        
        # 平滑標籤
        window = np.ones(32) / 32
        y_true = np.convolve(y_true, window, mode='same')
        y_true = (y_true > 0.3).astype(int)
        
        return y_true

    def process_audio(self, audio_path, output_path):
        try:
            # 加載音頻
            audio, sr = torchaudio.load(audio_path)
            audio = audio.squeeze().numpy()
            
            # 音量歸一化
            audio = self.normalize_volume(audio)
            
            # 計算SNR
            snr = self.calculate_snr(audio, sr)
            
            # VAD處理
            start_time = time.time()
            speech_prob = self.vad.get_speech_prob_file(audio_path).numpy()
            
            # 使用滑動平均平滑概率
            window = np.ones(self.window_size) / self.window_size
            smoothed_prob = np.convolve(speech_prob, window, mode='same')
            
            # 自適應閾值
            dynamic_threshold = np.mean(smoothed_prob) * 0.6  # 降低閾值係數
            threshold = min(max(dynamic_threshold, 0.15), self.speech_threshold)  # 降低最小閾值
            
            # 檢測語音區域
            speech_regions = []
            in_speech = False
            speech_start = 0
            prev_end = 0
            
            for i, prob in enumerate(smoothed_prob):
                if not in_speech and prob > threshold:
                    speech_start = max(0, i - int(self.boundary_extension * len(smoothed_prob)))
                    in_speech = True
                elif in_speech and prob < threshold:
                    speech_end = min(len(smoothed_prob), i + int(self.boundary_extension * len(smoothed_prob)))
                    
                    # 檢查最小持續時間
                    duration = (speech_end - speech_start) / len(smoothed_prob)
                    if duration >= self.min_silence_duration:
                        # 合併接近的區域
                        if speech_regions and (speech_start - prev_end) / len(smoothed_prob) < self.merge_threshold:
                            speech_regions[-1][1] = speech_end
                        else:
                            speech_regions.append([speech_start, speech_end])
                        prev_end = speech_end
                    in_speech = False
            
            # 處理最後一個區域
            if in_speech:
                speech_end = len(smoothed_prob)
                if (speech_end - speech_start) / len(smoothed_prob) >= self.min_silence_duration:
                    if speech_regions and (speech_start - prev_end) / len(smoothed_prob) < self.merge_threshold:
                        speech_regions[-1][1] = speech_end
                    else:
                        speech_regions.append([speech_start, speech_end])
            
            # 計算處理延遲
            processing_delay = (time.time() - start_time) * 1000  # 轉換為毫秒
            
            # 生成預測標籤
            y_pred = np.zeros(len(smoothed_prob))
            for start, end in speech_regions:
                y_pred[start:end] = 1
            
            # 生成參考標籤
            y_true = self.generate_reference_labels(audio, sr, smoothed_prob)
            
            # 計算指標
            accuracy = accuracy_score(y_true, y_pred) * 100
            recall = recall_score(y_true, y_pred, zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # 計算切分準確率
            total_segments = len(speech_regions)
            correct_segments = sum(1 for region in speech_regions if any(y_true[region[0]:region[1]]))
            segmentation_accuracy = (correct_segments / total_segments * 100) if total_segments > 0 else 0
            
            # 計算RMSE
            true_changes = np.where(np.diff(y_true) != 0)[0]
            pred_changes = np.where(np.diff(y_pred) != 0)[0]
            if len(true_changes) > 0 and len(pred_changes) > 0:
                # 對齊邊界點
                true_times = true_changes / sr
                pred_times = pred_changes / sr
                
                # 計算每個預測邊界到最近真實邊界的距離
                min_distances = []
                for pred_time in pred_times:
                    distances = np.abs(true_times - pred_time)
                    min_distances.append(np.min(distances))
                
                rmse = np.sqrt(np.mean(np.array(min_distances)**2)) * 1000  # 轉換為毫秒
            else:
                rmse = float('inf')
            
            # 保存結果
            result = {
                "filename": os.path.basename(audio_path),
                "speech_regions": [[float(start)/len(smoothed_prob), float(end)/len(smoothed_prob)] for start, end in speech_regions],
                "accuracy": accuracy,
                "recall": recall,
                "f1_score": f1,
                "processing_delay": processing_delay,
                "threshold": threshold,
                "snr": snr,
                "rmse": rmse,
                "segmentation_accuracy": segmentation_accuracy
            }
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            return True, "處理成功"
            
        except Exception as e:
            return False, f"處理失敗: {str(e)}"

if __name__ == "__main__":
    processor = ImprovedVADProcessor()
    audio_dir = "../02_audio_noise_reducer_experiment/output"
    output_dir = "vad_experiment_results/improved"
    
    if not os.path.exists(audio_dir):
        print(f"目錄不存在: {audio_dir}")
        exit(1)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    total_files = len(audio_files)
    print(f"找到 {total_files} 個音檔待處理")
    
    success_count = 0
    error_count = 0
    error_log = []
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"\n處理進度: [{i}/{total_files}] {audio_file}")
            audio_path = os.path.join(audio_dir, audio_file)
            
            if processor.process_audio(audio_path, os.path.join(output_dir, os.path.splitext(audio_file)[0] + "_vad.json")):
                success_count += 1
                print(f"✓ 成功處理: {audio_file}")
            else:
                error_count += 1
                error_msg = f"✗ 處理失敗: {audio_file}"
                print(error_msg)
                error_log.append(error_msg)
                
        except Exception as e:
            error_count += 1
            error_msg = f"✗ 處理出錯: {audio_file}, 錯誤: {str(e)}"
            print(error_msg)
            error_log.append(error_msg)
            
    print(f"\n處理完成!")
    print(f"成功: {success_count}/{total_files}")
    print(f"失敗: {error_count}/{total_files}")
    
    if error_log:
        print("\n錯誤日誌:")
        for error in error_log:
            print(error) 