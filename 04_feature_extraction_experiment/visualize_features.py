import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_prepare_data():
    """載入並準備數據"""
    # 讀取原始數據
    df = pd.read_csv('feature_evaluation/feature_evaluation.csv')
    
    # 從文件名中提取類別（教師/學生）
    df['category'] = df['filename'].apply(lambda x: 'student' if 'Student' in x else 'teacher')
    
    # 將類別轉換為中文
    df['category'] = df['category'].map({'teacher': '教師', 'student': '學生'})
    
    # 將布爾值轉換為數值
    bool_columns = ['mfcc_stability', 'f0_quality', 'energy_stability', 'zcr_validity', 'feature_complete']
    for col in bool_columns:
        df[col] = df[col].astype(float)
    
    return df

def generate_summary_files(df):
    """生成摘要文件"""
    # 學生數據摘要
    student_df = df[df['category'] == '學生']
    student_summary = {
        'file_count': len(student_df),
        'mfcc_stability_rate': f"{(student_df['mfcc_stability'].mean() * 100):.2f}%",
        'f0_quality_rate': f"{(student_df['f0_quality'].mean() * 100):.2f}%",
        'energy_stability_rate': f"{(student_df['energy_stability'].mean() * 100):.2f}%",
        'zcr_rationality_rate': f"{(student_df['zcr_validity'].mean() * 100):.2f}%",
        'feature_integrity_rate': f"{(student_df['feature_complete'].mean() * 100):.2f}%"
    }
    
    # 將摘要保存為CSV
    pd.DataFrame([student_summary]).to_csv('feature_evaluation/feature_evaluation_summary_student.csv', index=False)
    
    # 教師數據摘要
    teacher_df = df[df['category'] == '教師']
    teacher_summary = {
        'file_count': len(teacher_df),
        'mfcc_stability_rate': f"{(teacher_df['mfcc_stability'].mean() * 100):.2f}%",
        'f0_quality_rate': f"{(teacher_df['f0_quality'].mean() * 100):.2f}%",
        'energy_stability_rate': f"{(teacher_df['energy_stability'].mean() * 100):.2f}%",
        'zcr_rationality_rate': f"{(teacher_df['zcr_validity'].mean() * 100):.2f}%",
        'feature_integrity_rate': f"{(teacher_df['feature_complete'].mean() * 100):.2f}%"
    }
    
    # 將摘要保存為CSV
    pd.DataFrame([teacher_summary]).to_csv('feature_evaluation/feature_evaluation_summary_teacher.csv', index=False)

def plot_feature_distributions(df):
    """繪製特徵分佈圖"""
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('教師和學生語音特徵分佈對比', fontsize=16)
    
    # MFCC穩定性分佈
    sns.barplot(x='category', y='mfcc_stability', data=df, ax=axes[0,0])
    axes[0,0].set_title('MFCC穩定性分佈')
    axes[0,0].set_xlabel('類別')
    axes[0,0].set_ylabel('穩定性率')
    
    # F0質量分佈
    sns.barplot(x='category', y='f0_quality', data=df, ax=axes[0,1])
    axes[0,1].set_title('F0質量分佈')
    axes[0,1].set_xlabel('類別')
    axes[0,1].set_ylabel('質量率')
    
    # 能量穩定性分佈
    sns.barplot(x='category', y='energy_stability', data=df, ax=axes[1,0])
    axes[1,0].set_title('能量穩定性分佈')
    axes[1,0].set_xlabel('類別')
    axes[1,0].set_ylabel('穩定性率')
    
    # ZCR合理性分佈
    sns.barplot(x='category', y='zcr_validity', data=df, ax=axes[1,1])
    axes[1,1].set_title('過零率合理性分佈')
    axes[1,1].set_xlabel('類別')
    axes[1,1].set_ylabel('合理性率')
    
    plt.tight_layout()
    plt.savefig('feature_evaluation/feature_distributions.png', dpi=300)
    plt.close()

def plot_feature_correlations(df):
    """繪製特徵相關性熱圖"""
    # 選擇數值型特徵
    numeric_features = ['mfcc_stability', 'f0_quality', 'energy_stability', 'zcr_validity']
    
    # 分別計算教師和學生的相關性
    teacher_corr = df[df['category'] == '教師'][numeric_features].corr()
    student_corr = df[df['category'] == '學生'][numeric_features].corr()
    
    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 繪製教師相關性熱圖
    sns.heatmap(teacher_corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('教師語音特徵相關性')
    
    # 繪製學生相關性熱圖
    sns.heatmap(student_corr, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('學生語音特徵相關性')
    
    plt.tight_layout()
    plt.savefig('feature_evaluation/feature_correlations.png', dpi=300)
    plt.close()

def plot_feature_statistics(df):
    """繪製特徵統計圖"""
    # 計算每個類別的統計數據
    stats = df.groupby('category').agg({
        'mfcc_stability': ['mean', 'std'],
        'f0_quality': ['mean', 'std'],
        'energy_stability': ['mean', 'std'],
        'zcr_validity': ['mean', 'std']
    }).round(4)
    
    # 保存統計數據
    stats.to_csv('feature_evaluation/feature_statistics.csv')
    
    # 創建統計圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('教師和學生語音特徵統計對比', fontsize=16)
    
    features = ['mfcc_stability', 'f0_quality', 'energy_stability', 'zcr_validity']
    titles = ['MFCC穩定性', 'F0質量', '能量穩定性', '過零率合理性']
    
    for i, (feature, title) in enumerate(zip(features, titles)):
        ax = axes[i//2, i%2]
        means = stats[feature]['mean']
        stds = stats[feature]['std']
        
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(means.index)
        ax.set_title(title)
        ax.set_ylabel('比率')
        
    plt.tight_layout()
    plt.savefig('feature_evaluation/feature_statistics.png', dpi=300)
    plt.close()

def main():
    # 載入數據
    df = load_and_prepare_data()
    
    # 生成摘要文件
    generate_summary_files(df)
    
    # 繪製分佈圖
    plot_feature_distributions(df)
    
    # 繪製相關性圖
    plot_feature_correlations(df)
    
    # 繪製統計圖
    plot_feature_statistics(df)
    
    print("可視化結果已保存至feature_evaluation目錄")

if __name__ == '__main__':
    main() 