import os
import json
from collections import defaultdict
import glob

def find_matching_pairs():
    # 基本路徑設定
    student_dir = "../02_audio_noise_reducer_experiment/output"
    teacher_base_dir = "../03_audio_vad_experiment/datasets"
    
    # 獲取所有學生音檔
    student_files = [f for f in os.listdir(student_dir) if f.endswith('.wav')]
    
    # 獲取所有session目錄
    session_dirs = [d for d in os.listdir(teacher_base_dir) 
                   if os.path.isdir(os.path.join(teacher_base_dir, d)) and d.startswith('session_')]
    
    # 存儲所有配對
    matching_pairs = defaultdict(dict)
    
    # 遍歷每個session目錄
    for session_dir in session_dirs:
        teacher_recordings_dir = os.path.join(teacher_base_dir, session_dir, "teacher_recordings")
        if not os.path.exists(teacher_recordings_dir):
            continue
            
        # 獲取該session中的所有教師音檔
        teacher_files = [f for f in os.listdir(teacher_recordings_dir) 
                        if f.endswith('.wav') and 'Teacher' in f]
        
        # 處理每個教師音檔
        for teacher_file in teacher_files:
            # 解析教師音檔名稱
            parts = teacher_file.split('_')
            if len(parts) < 4:
                continue
                
            lesson = parts[0]      # Lesson01
            teacher = parts[1]     # Pete/Anna
            utterance = parts[3].split('.')[0]  # utterance01
            
            # 構建對應的學生音檔名稱模式
            student_pattern = f"{lesson}_{teacher}_Student*_{utterance}"
            
            # 在學生音檔中尋找匹配
            for student_file in student_files:
                if student_file.startswith(f"{lesson}_{teacher}_Student") and utterance in student_file:
                    # 解析學生ID
                    student_parts = student_file.split('_')
                    student = student_parts[2]  # Student01
                    
                    key = f"{session_dir}_{lesson}_{teacher}_{student}_{utterance}"
                    matching_pairs[key] = {
                        'session': session_dir,
                        'lesson': lesson,
                        'teacher': teacher,
                        'student': student,
                        'utterance': utterance,
                        'student_file': os.path.join(student_dir, student_file),
                        'teacher_file': os.path.join(teacher_recordings_dir, teacher_file)
                    }
    
    return matching_pairs

def main():
    # 找出配對
    matching_pairs = find_matching_pairs()
    
    # 輸出結果
    print(f"\n找到 {len(matching_pairs)} 組配對的音檔：")
    
    # 按session分組顯示
    by_session = defaultdict(list)
    for key, info in matching_pairs.items():
        by_session[info['session']].append((key, info))
    
    for session in sorted(by_session.keys()):
        print(f"\n會話: {session}")
        print("-" * 130)
        print("{:<15} {:<10} {:<10} {:<10} {:<15} {:<35} {:<35}".format(
            "課程", "教師", "學生", "發音編號", "錄音時間",
            "學生音檔", "教師音檔"
        ))
        print("-" * 130)
        
        for key, info in sorted(by_session[session]):
            print("{:<15} {:<10} {:<10} {:<10} {:<15} {:<35} {:<35}".format(
                info['lesson'],
                info['teacher'],
                info['student'],
                info['utterance'],
                session.replace('session_', ''),
                os.path.basename(info['student_file']),
                os.path.basename(info['teacher_file'])
            ))
    
    # 保存結果
    output_file = 'matching_pairs.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matching_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果已保存到 {output_file}")

if __name__ == "__main__":
    main() 