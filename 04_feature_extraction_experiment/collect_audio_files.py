import os
import shutil
from pathlib import Path

# 獲取腳本所在目錄的絕對路徑
script_dir = Path(__file__).parent.absolute()

# 設定基礎路徑
base_path = script_dir / 'audio_datasets'
output_path = script_dir / 'organized_audio'

# 創建輸出目錄
teacher_output = output_path / 'teacher'
student_output = output_path / 'student'
teacher_output.mkdir(parents=True, exist_ok=True)
student_output.mkdir(parents=True, exist_ok=True)

# 收集所有音檔
def collect_audio_files():
    teacher_files = []
    student_files = []
    
    # 遍歷所有會話目錄
    for session_dir in base_path.glob('session_*'):
        if not session_dir.is_dir():
            continue
            
        # 收集老師音檔
        teacher_path = session_dir / 'teacher_recordings'
        if teacher_path.exists():
            for audio_file in teacher_path.glob('*.wav'):
                teacher_files.append(audio_file)
        
        # 收集學生音檔
        student_path = session_dir / 'student_recordings'
        if student_path.exists():
            for audio_file in student_path.glob('*.wav'):
                student_files.append(audio_file)
    
    return teacher_files, student_files

def copy_files(files, target_dir):
    for file in files:
        # 使用會話ID和原始檔名組合作為新檔名
        session_id = file.parent.parent.name
        new_name = f"{session_id}_{file.name}"
        shutil.copy2(file, target_dir / new_name)
        print(f"已複製: {new_name}")

if __name__ == '__main__':
    print(f"基礎路徑: {base_path}")
    teacher_files, student_files = collect_audio_files()
    
    print(f"\n找到 {len(teacher_files)} 個老師音檔")
    print(f"找到 {len(student_files)} 個學生音檔\n")
    
    print("開始複製老師音檔...")
    copy_files(teacher_files, teacher_output)
    
    print("\n開始複製學生音檔...")
    copy_files(student_files, student_output)
    
    print("\n音檔整理完成！") 