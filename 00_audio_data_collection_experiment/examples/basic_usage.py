from vad_detector import VADDetector

# 初始化檢測器
detector = VADDetector(
    window_size=3,
    threshold=0.3,
    min_duration=0.1,
    max_merge_gap=0.3
)

# 處理單個文件
result = detector.process_file('path/to/audio.wav')
print('語音區段:', result['speech_regions'])

# 批量處理目錄
results = detector.process_directory('path/to/audio/directory')
for filename, result in results.items():
    print(f'文件 {filename} 的語音區段:', result['speech_regions']) 