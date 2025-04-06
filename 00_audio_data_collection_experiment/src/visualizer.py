import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os

class Visualizer:
    def __init__(self, output_dir: str = 'visualization_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_cv_distribution(self,
                           baseline_cv: List[float],
                           treatment_cv: List[float],
                           save_path: str = None):
        """繪製 CV 分布直方圖"""
        plt.figure(figsize=(10, 6))
        plt.hist(baseline_cv, alpha=0.5, label='原始組', bins=30)
        plt.hist(treatment_cv, alpha=0.5, label='控制組', bins=30)
        plt.xlabel('變異係數 (CV)')
        plt.ylabel('頻率')
        plt.title('音量變異係數分布比較')
        plt.legend()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            plt.close()
        else:
            plt.show()
    
    def plot_error_comparison(self,
                            metrics: Dict,
                            save_path: str = None):
        """繪製模型誤差比較圖"""
        metrics_df = pd.DataFrame({
            '指標': ['MAE', 'RMSE', 'Kappa'],
            '原始組': [
                metrics['baseline']['mae'],
                metrics['baseline']['rmse'],
                metrics['baseline']['kappa']
            ],
            '控制組': [
                metrics['treatment']['mae'],
                metrics['treatment']['rmse'],
                metrics['treatment']['kappa']
            ]
        })
        
        plt.figure(figsize=(10, 6))
        metrics_df.plot(x='指標', kind='bar', rot=0)
        plt.title('模型評估指標比較')
        plt.ylabel('數值')
        plt.legend(title='')
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            plt.close()
        else:
            plt.show()
    
    def plot_cv_improvement(self,
                          cv_values: List[float],
                          save_path: str = None):
        """繪製 CV 改善率折線圖"""
        plt.figure(figsize=(10, 6))
        
        # 計算累積改善率
        improvements = np.sort(cv_values)
        percentiles = np.arange(len(improvements)) / len(improvements) * 100
        
        plt.plot(percentiles, improvements)
        plt.xlabel('百分位數')
        plt.ylabel('CV 改善率 (%)')
        plt.title('音量變異係數改善率分布')
        plt.grid(True)
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path))
            plt.close()
        else:
            plt.show() 