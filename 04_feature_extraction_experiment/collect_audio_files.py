import os
from pathlib import Path
import shutil
from tqdm import tqdm

def collect_audio_files():
    """收集並組織音頻文件"""
    # 設置目錄路徑
    base_dir = Path(__file__).parent / 'audio_datasets'
    organized_dir = Path(__file__).parent.parent / 'organized_audio'
    student_dir = organized_dir / 'student'
    teacher_dir = organized_dir / 'teacher'
    
    # 創建目錄
    student_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    
    # 遞歸搜索所有session目錄
    session_dirs = list(base_dir.glob('session_*'))
    
    if not session_dirs:
        print("錯誤：未找到任何會話目錄")
        return
    
    print(f"\n找到 {len(session_dirs)} 個會話目錄")
    
    # 收集音頻文件
    student_files = []
    teacher_files = []
    
    for session_dir in session_dirs:
        # 搜索所有WAV文件
        wav_files = list(session_dir.rglob('*.wav'))
        for wav_file in wav_files:
            if 'Student' in wav_file.name:
                student_files.append(wav_file)
            elif 'Teacher' in wav_file.name:
                teacher_files.append(wav_file)
    
    print(f"找到 {len(student_files)} 個學生語音文件")
    print(f"找到 {len(teacher_files)} 個教師語音文件")
    
    # 複製學生語音文件
    print("\n正在複製學生語音文件...")
    for src_file in tqdm(student_files):
        dst_file = student_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"已複製: {src_file.name}")
    
    # 複製教師語音文件
    print("\n正在複製教師語音文件...")
    for src_file in tqdm(teacher_files):
        dst_file = teacher_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"已複製: {src_file.name}")
    
    print("\n音頻文件組織完成！")
    print(f"學生語音文件已保存至：{student_dir}")
    print(f"教師語音文件已保存至：{teacher_dir}")

if __name__ == '__main__':
    collect_audio_files() 