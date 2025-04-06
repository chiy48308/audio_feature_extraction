#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音檔質量分析腳本
用於分析所有實驗會話中的音頻文件，評估其質量指標
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 導入音頻格式評估模塊
from audio_data_collection.audio_format_assessment import (
    find_wav_files,
    assess_audio_file,
    batch_assess_audio,
    generate_summary_report
)

def analyze_by_session(results_df):
    """
    按會話分析音頻質量
    
    參數:
    - results_df: 包含評估結果的DataFrame
    
    返回:
    - session_stats: 包含各會話統計信息的DataFrame
    """
    # 從文件路徑中提取會話ID
    results_df['session_id'] = results_df['file_path'].apply(
        lambda x: x.split('session_')[1].split('/')[0] if 'session_' in x else 'unknown'
    )
    
    # 按會話分組計算統計信息
    session_stats = results_df.groupby('session_id').agg({
        'format_ok': 'mean',
        'silence_ok': 'mean',
        'volume_ok': 'mean',
        'stability_ok': 'mean',
        'snr_ok': 'mean',
        'assessment_ok': 'mean',
        'silence_ratio': 'mean',
        'max_silence_duration': 'mean',
        'rms_dbfs': 'mean',
        'peak_dbfs': 'mean',
        'rms_cv': 'mean',
        'snr': 'mean',
        'file_path': 'count'
    }).reset_index()
    
    # 重命名列
    session_stats = session_stats.rename(columns={
        'format_ok': '錄音格式合格率',
        'silence_ok': '靜音檢測合格率',
        'volume_ok': '音量範圍合格率',
        'stability_ok': '音量穩定性合格率',
        'snr_ok': '信噪比合格率',
        'assessment_ok': '整體合格率',
        'silence_ratio': '平均靜音比例',
        'max_silence_duration': '平均最長靜音段',
        'rms_dbfs': '平均RMS音量',
        'peak_dbfs': '平均峰值音量',
        'rms_cv': '平均音量變異係數',
        'snr': '平均信噪比',
        'file_path': '文件數量'
    })
    
    # 將合格率轉換為百分比
    percentage_columns = [
        '錄音格式合格率', '靜音檢測合格率', '音量範圍合格率',
        '音量穩定性合格率', '信噪比合格率', '整體合格率'
    ]
    for col in percentage_columns:
        session_stats[col] = session_stats[col] * 100
    
    # 將靜音比例轉換為百分比
    session_stats['平均靜音比例'] = session_stats['平均靜音比例'] * 100
    
    return session_stats

def analyze_by_role(results_df):
    """
    按角色(教師/學生)分析音頻質量
    
    參數:
    - results_df: 包含評估結果的DataFrame
    
    返回:
    - role_stats: 包含各角色統計信息的DataFrame
    """
    # 從文件路徑中提取角色
    results_df['role'] = results_df['file_path'].apply(
        lambda x: 'teacher' if 'teacher_recordings' in x else 'student' if 'student_recordings' in x else 'unknown'
    )
    
    # 按角色分組計算統計信息
    role_stats = results_df.groupby('role').agg({
        'format_ok': 'mean',
        'silence_ok': 'mean',
        'volume_ok': 'mean',
        'stability_ok': 'mean',
        'snr_ok': 'mean',
        'assessment_ok': 'mean',
        'silence_ratio': 'mean',
        'max_silence_duration': 'mean',
        'rms_dbfs': 'mean',
        'peak_dbfs': 'mean',
        'rms_cv': 'mean',
        'snr': 'mean',
        'file_path': 'count'
    }).reset_index()
    
    # 重命名列
    role_stats = role_stats.rename(columns={
        'role': '角色',
        'format_ok': '錄音格式合格率',
        'silence_ok': '靜音檢測合格率',
        'volume_ok': '音量範圍合格率',
        'stability_ok': '音量穩定性合格率',
        'snr_ok': '信噪比合格率',
        'assessment_ok': '整體合格率',
        'silence_ratio': '平均靜音比例',
        'max_silence_duration': '平均最長靜音段',
        'rms_dbfs': '平均RMS音量',
        'peak_dbfs': '平均峰值音量',
        'rms_cv': '平均音量變異係數',
        'snr': '平均信噪比',
        'file_path': '文件數量'
    })
    
    # 將合格率轉換為百分比
    percentage_columns = [
        '錄音格式合格率', '靜音檢測合格率', '音量範圍合格率',
        '音量穩定性合格率', '信噪比合格率', '整體合格率'
    ]
    for col in percentage_columns:
        role_stats[col] = role_stats[col] * 100
    
    # 將靜音比例轉換為百分比
    role_stats['平均靜音比例'] = role_stats['平均靜音比例'] * 100
    
    # 將角色翻譯為中文
    role_stats['角色'] = role_stats['角色'].map({
        'teacher': '教師',
        'student': '學生',
        'unknown': '未知'
    })
    
    return role_stats

def generate_additional_visualizations(results_df, output_dir):
    """
    生成額外的視覺化圖表
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - output_dir: 輸出目錄
    """
    # 創建輸出目錄
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 設置風格
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 從文件路徑中提取會話ID和角色
    results_df['session_id'] = results_df['file_path'].apply(
        lambda x: x.split('session_')[1].split('/')[0] if 'session_' in x else 'unknown'
    )
    results_df['role'] = results_df['file_path'].apply(
        lambda x: '教師' if 'teacher_recordings' in x else '學生' if 'student_recordings' in x else '未知'
    )
    
    # 1. 按會話比較合格率
    session_stats = analyze_by_session(results_df)
    
    plt.figure(figsize=(15, 8))
    metrics = ['錄音格式合格率', '靜音檢測合格率', '音量範圍合格率', '音量穩定性合格率', '信噪比合格率']
    
    # 轉換為長格式
    session_long = pd.melt(
        session_stats, 
        id_vars=['session_id'], 
        value_vars=metrics,
        var_name='指標',
        value_name='合格率'
    )
    
    # 繪製按會話的合格率比較圖
    sns.barplot(x='session_id', y='合格率', hue='指標', data=session_long)
    plt.title('各會話音頻質量指標合格率比較')
    plt.xlabel('會話ID')
    plt.ylabel('合格率 (%)')
    plt.xticks(rotation=45)
    plt.legend(title='質量指標')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'session_pass_rates.png'), dpi=300)
    plt.close()
    
    # 2. 按角色比較合格率
    role_stats = analyze_by_role(results_df)
    
    plt.figure(figsize=(12, 6))
    
    # 轉換為長格式
    role_long = pd.melt(
        role_stats, 
        id_vars=['角色'], 
        value_vars=metrics,
        var_name='指標',
        value_name='合格率'
    )
    
    # 繪製按角色的合格率比較圖
    sns.barplot(x='指標', y='合格率', hue='角色', data=role_long)
    plt.title('教師與學生音頻質量指標合格率比較')
    plt.xlabel('質量指標')
    plt.ylabel('合格率 (%)')
    plt.legend(title='角色')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'role_pass_rates.png'), dpi=300)
    plt.close()
    
    # 3. 按會話比較SNR
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='session_id', y='snr', data=results_df)
    plt.axhline(y=20, color='r', linestyle='--', label='閾值 (20 dB)')
    plt.title('各會話信噪比(SNR)分布')
    plt.xlabel('會話ID')
    plt.ylabel('信噪比 (dB)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'session_snr_distribution.png'), dpi=300)
    plt.close()
    
    # 4. 按角色比較SNR
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='role', y='snr', data=results_df)
    plt.axhline(y=20, color='r', linestyle='--', label='閾值 (20 dB)')
    plt.title('教師與學生信噪比(SNR)分布')
    plt.xlabel('角色')
    plt.ylabel('信噪比 (dB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'role_snr_distribution.png'), dpi=300)
    plt.close()
    
    # 5. 按會話比較靜音比例
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='session_id', y='silence_ratio', data=results_df)
    plt.axhline(y=0.3, color='r', linestyle='--', label='閾值 (30%)')
    plt.title('各會話靜音比例分布')
    plt.xlabel('會話ID')
    plt.ylabel('靜音比例')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'session_silence_distribution.png'), dpi=300)
    plt.close()
    
    # 6. 按角色比較靜音比例
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='role', y='silence_ratio', data=results_df)
    plt.axhline(y=0.3, color='r', linestyle='--', label='閾值 (30%)')
    plt.title('教師與學生靜音比例分布')
    plt.xlabel('角色')
    plt.ylabel('靜音比例')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'role_silence_distribution.png'), dpi=300)
    plt.close()
    
    # 7. 按會話比較RMS音量
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='session_id', y='rms_dbfs', data=results_df)
    plt.axhline(y=-30, color='r', linestyle='--', label='閾值 (-30 dBFS)')
    plt.title('各會話RMS音量分布')
    plt.xlabel('會話ID')
    plt.ylabel('RMS音量 (dBFS)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'session_rms_distribution.png'), dpi=300)
    plt.close()
    
    # 8. 按角色比較RMS音量
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='role', y='rms_dbfs', data=results_df)
    plt.axhline(y=-30, color='r', linestyle='--', label='閾值 (-30 dBFS)')
    plt.title('教師與學生RMS音量分布')
    plt.xlabel('角色')
    plt.ylabel('RMS音量 (dBFS)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'role_rms_distribution.png'), dpi=300)
    plt.close()
    
    # 9. 按會話比較音量穩定性
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='session_id', y='rms_cv', data=results_df)
    plt.axhline(y=0.5, color='r', linestyle='--', label='閾值 (0.5)')
    plt.title('各會話音量穩定性分布')
    plt.xlabel('會話ID')
    plt.ylabel('音量變異係數')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'session_stability_distribution.png'), dpi=300)
    plt.close()
    
    # 10. 按角色比較音量穩定性
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='role', y='rms_cv', data=results_df)
    plt.axhline(y=0.5, color='r', linestyle='--', label='閾值 (0.5)')
    plt.title('教師與學生音量穩定性分布')
    plt.xlabel('角色')
    plt.ylabel('音量變異係數')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'role_stability_distribution.png'), dpi=300)
    plt.close()
    
    print(f"額外視覺化圖表已保存至: {vis_dir}")

def generate_detailed_report(results_df, session_stats, role_stats, output_file):
    """
    生成詳細報告
    
    參數:
    - results_df: 包含評估結果的DataFrame
    - session_stats: 包含各會話統計信息的DataFrame
    - role_stats: 包含各角色統計信息的DataFrame
    - output_file: 輸出文件路徑
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("音檔質量詳細分析報告\n")
        f.write("=" * 50 + "\n\n")
        
        # 總體統計
        total_files = len(results_df)
        format_pass = len(results_df[results_df['format_ok'] == True])
        silence_pass = len(results_df[results_df['silence_ok'] == True])
        volume_pass = len(results_df[results_df['volume_ok'] == True])
        stability_pass = len(results_df[results_df['stability_ok'] == True])
        snr_pass = len(results_df[results_df['snr_ok'] == True])
        overall_pass = len(results_df[results_df['assessment_ok'] == True])
        
        f.write("1. 總體統計\n")
        f.write("-" * 50 + "\n")
        f.write(f"評估音檔總數: {total_files}\n\n")
        
        f.write("合格率統計:\n")
        f.write(f"錄音格式合格率: {format_pass / total_files * 100:.1f}%\n")
        f.write(f"靜音檢測合格率: {silence_pass / total_files * 100:.1f}%\n")
        f.write(f"音量範圍合格率: {volume_pass / total_files * 100:.1f}%\n")
        f.write(f"音量穩定性合格率: {stability_pass / total_files * 100:.1f}%\n")
        f.write(f"信噪比合格率: {snr_pass / total_files * 100:.1f}%\n")
        f.write(f"整體合格率: {overall_pass / total_files * 100:.1f}%\n\n")
        
        f.write("指標統計:\n")
        f.write(f"取樣率: {results_df['sample_rate'].mean():.1f} Hz (標準: 16000 Hz)\n")
        f.write(f"靜音比例: {results_df['silence_ratio'].mean() * 100:.1f}% (標準: < 30%)\n")
        f.write(f"最長靜音段: {results_df['max_silence_duration'].mean():.2f} 秒 (標準: < 1秒)\n")
        f.write(f"RMS音量: {results_df['rms_dbfs'].mean():.1f} dBFS (標準: > -30 dBFS)\n")
        f.write(f"峰值音量: {results_df['peak_dbfs'].mean():.1f} dBFS (標準: < 0 dBFS)\n")
        f.write(f"音量變異係數: {results_df['rms_cv'].mean():.3f} (標準: < 0.5)\n")
        f.write(f"信噪比: {results_df['snr'].mean():.1f} dB (標準: ≥ 20 dB)\n\n")
        
        # 按會話分析
        f.write("2. 按會話分析\n")
        f.write("-" * 50 + "\n")
        
        for _, row in session_stats.iterrows():
            session_id = row['session_id']
            f.write(f"會話ID: {session_id}\n")
            f.write(f"文件數量: {row['文件數量']:.0f}\n")
            f.write(f"錄音格式合格率: {row['錄音格式合格率']:.1f}%\n")
            f.write(f"靜音檢測合格率: {row['靜音檢測合格率']:.1f}%\n")
            f.write(f"音量範圍合格率: {row['音量範圍合格率']:.1f}%\n")
            f.write(f"音量穩定性合格率: {row['音量穩定性合格率']:.1f}%\n")
            f.write(f"信噪比合格率: {row['信噪比合格率']:.1f}%\n")
            f.write(f"整體合格率: {row['整體合格率']:.1f}%\n")
            f.write(f"平均靜音比例: {row['平均靜音比例']:.1f}%\n")
            f.write(f"平均最長靜音段: {row['平均最長靜音段']:.2f} 秒\n")
            f.write(f"平均RMS音量: {row['平均RMS音量']:.1f} dBFS\n")
            f.write(f"平均峰值音量: {row['平均峰值音量']:.1f} dBFS\n")
            f.write(f"平均音量變異係數: {row['平均音量變異係數']:.3f}\n")
            f.write(f"平均信噪比: {row['平均信噪比']:.1f} dB\n\n")
        
        # 按角色分析
        f.write("3. 按角色分析\n")
        f.write("-" * 50 + "\n")
        
        for _, row in role_stats.iterrows():
            role = row['角色']
            f.write(f"角色: {role}\n")
            f.write(f"文件數量: {row['文件數量']:.0f}\n")
            f.write(f"錄音格式合格率: {row['錄音格式合格率']:.1f}%\n")
            f.write(f"靜音檢測合格率: {row['靜音檢測合格率']:.1f}%\n")
            f.write(f"音量範圍合格率: {row['音量範圍合格率']:.1f}%\n")
            f.write(f"音量穩定性合格率: {row['音量穩定性合格率']:.1f}%\n")
            f.write(f"信噪比合格率: {row['信噪比合格率']:.1f}%\n")
            f.write(f"整體合格率: {row['整體合格率']:.1f}%\n")
            f.write(f"平均靜音比例: {row['平均靜音比例']:.1f}%\n")
            f.write(f"平均最長靜音段: {row['平均最長靜音段']:.2f} 秒\n")
            f.write(f"平均RMS音量: {row['平均RMS音量']:.1f} dBFS\n")
            f.write(f"平均峰值音量: {row['平均峰值音量']:.1f} dBFS\n")
            f.write(f"平均音量變異係數: {row['平均音量變異係數']:.3f}\n")
            f.write(f"平均信噪比: {row['平均信噪比']:.1f} dB\n\n")
        
        # 結論與建議
        f.write("4. 結論與建議\n")
        f.write("-" * 50 + "\n")
        
        # 找出最佳和最差的會話
        best_session = session_stats.loc[session_stats['整體合格率'].idxmax()]
        worst_session = session_stats.loc[session_stats['整體合格率'].idxmin()]
        
        f.write("最佳表現會話:\n")
        f.write(f"會話ID: {best_session['session_id']}\n")
        f.write(f"整體合格率: {best_session['整體合格率']:.1f}%\n\n")
        
        f.write("最差表現會話:\n")
        f.write(f"會話ID: {worst_session['session_id']}\n")
        f.write(f"整體合格率: {worst_session['整體合格率']:.1f}%\n\n")
        
        # 主要問題分析
        problem_counts = {
            '錄音格式問題': total_files - format_pass,
            '靜音問題': total_files - silence_pass,
            '音量問題': total_files - volume_pass,
            '穩定性問題': total_files - stability_pass,
            '信噪比問題': total_files - snr_pass
        }
        
        # 按問題數量排序
        sorted_problems = sorted(problem_counts.items(), key=lambda x: x[1], reverse=True)
        
        f.write("主要問題分析:\n")
        for problem, count in sorted_problems:
            if count > 0:
                f.write(f"{problem}: {count} 個文件 ({count / total_files * 100:.1f}%)\n")
        
        f.write("\n建議:\n")
        
        # 根據主要問題提供建議
        if problem_counts['錄音格式問題'] > 0:
            f.write("1. 確保所有錄音設備設置為 16 kHz, 16-bit, 單聲道格式。\n")
        
        if problem_counts['靜音問題'] > 0:
            f.write("2. 減少錄音中的靜音段，確保靜音比例不超過30%，單段靜音不超過1秒。\n")
        
        if problem_counts['音量問題'] > 0:
            f.write("3. 調整錄音音量，確保RMS音量高於-30 dBFS，但峰值不超過0 dBFS。\n")
        
        if problem_counts['穩定性問題'] > 0:
            f.write("4. 保持說話音量的穩定性，避免音量忽大忽小。\n")
        
        if problem_counts['信噪比問題'] > 0:
            f.write("5. 減少環境噪音，確保錄音環境安靜，或使用降噪技術處理音頻。\n")
    
    print(f"詳細報告已保存至: {output_file}")

def main(base_dir, output_dir):
    """
    主函數
    
    參數:
    - base_dir: 基礎目錄路徑
    - output_dir: 輸出目錄路徑
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 搜尋所有.wav文件
    print("搜尋音檔...")
    wav_files = find_wav_files(base_dir)
    print(f"找到 {len(wav_files)} 個音檔")
    
    # 批量評估
    print("評估音檔...")
    results_df = batch_assess_audio(wav_files, output_dir)
    
    # 生成摘要報告
    print("生成摘要報告...")
    generate_summary_report(results_df, output_dir)
    
    # 按會話分析
    print("按會話分析...")
    session_stats = analyze_by_session(results_df)
    session_stats.to_csv(os.path.join(output_dir, 'session_stats.csv'), index=False)
    
    # 按角色分析
    print("按角色分析...")
    role_stats = analyze_by_role(results_df)
    role_stats.to_csv(os.path.join(output_dir, 'role_stats.csv'), index=False)
    
    # 生成額外視覺化
    print("生成額外視覺化...")
    generate_additional_visualizations(results_df, output_dir)
    
    # 生成詳細報告
    print("生成詳細報告...")
    generate_detailed_report(
        results_df,
        session_stats,
        role_stats,
        os.path.join(output_dir, 'detailed_report.txt')
    )
    
    print("分析完成!")
    print(f"所有結果已保存至: {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "."  # 當前目錄
    
    output_dir = "audio_analysis_results"
    
    main(base_dir, output_dir)
