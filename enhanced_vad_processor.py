import torch
import torchaudio
import json
import os
from speechbrain.pretrained import VAD

class EnhancedVADProcessor:
    def __init__(self, output_dir="vad_results"):
        self.vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", 
                                   savedir="pretrained_models/vad-crdnn-libriparty")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vad.eval()
        self.vad.to(self.device)

    def process_audio(self, audio_path):
        """處理單個音頻文件"""
        try:
            # 讀取音頻文件以獲取總時長
            waveform, sample_rate = torchaudio.load(audio_path)
            total_duration = waveform.shape[1] / sample_rate

            # 使用 get_speech_segments 方法直接獲取語音段
            speech_segments = self.vad.get_speech_segments(audio_path)
            
            # 將語音段轉換為所需的格式
            segments = []
            speech_duration = 0
            
            for segment in speech_segments:
                start_time = segment[0]
                end_time = segment[1]
                segments.append({
                    "start": round(float(start_time), 3),
                    "end": round(float(end_time), 3)
                })
                speech_duration += (end_time - start_time)

            # 計算語音百分比
            speech_percentage = (speech_duration / total_duration) * 100 if total_duration > 0 else 0

            return {
                "file_name": os.path.basename(audio_path),
                "total_duration": round(float(total_duration), 3),
                "speech_duration": round(float(speech_duration), 3),
                "speech_percentage": round(float(speech_percentage), 2),
                "speech_segments": segments
            }

        except Exception as e:
            print(f"處理失敗: {os.path.basename(audio_path)}, 錯誤: {str(e)}")
            return None

    def process_directory(self, audio_dir):
        """處理目錄中的所有音頻文件"""
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print(f"找到 {len(audio_files)} 個音檔待處理")
        
        results = []
        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            result = self.process_audio(audio_path)
            if result is not None:
                results.append(result)
        
        # 保存結果
        output_file = os.path.join(self.output_dir, "vad_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

if __name__ == "__main__":
    processor = EnhancedVADProcessor()
    processor.process_directory("processed_audio")