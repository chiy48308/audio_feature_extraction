import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from typing import Dict, Tuple, List
import logging
from scipy import stats

class ModelTrainer:
    def __init__(self):
        self.setup_logging()
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, features_list: List[Dict]) -> pd.DataFrame:
        """將特徵列表轉換為 DataFrame"""
        df = pd.DataFrame(features_list)
        return df
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """訓練模型並返回評估結果"""
        # 資料分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 訓練模型
        self.model.fit(X_train, y_train)
        
        # 預測
        y_pred = self.model.predict(X_test)
        
        # 計算評估指標
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'kappa': cohen_kappa_score(
                y_test.round(),
                y_pred.round(),
                weights='quadratic'
            )
        }
        
        # 進行統計檢定
        t_stat, p_value = stats.ttest_ind(y_test, y_pred)
        metrics['t_stat'] = t_stat
        metrics['p_value'] = p_value
        
        return metrics
    
    def compare_models(self, 
                      baseline_features: pd.DataFrame,
                      treatment_features: pd.DataFrame,
                      scores: np.ndarray) -> Dict:
        """比較基準組和處理組的模型表現"""
        baseline_metrics = self.train_model(baseline_features, scores)
        treatment_metrics = self.train_model(treatment_features, scores)
        
        comparison = {
            'baseline': baseline_metrics,
            'treatment': treatment_metrics,
            'improvement': {
                metric: (treatment_metrics[metric] - baseline_metrics[metric])
                for metric in baseline_metrics.keys()
            }
        }
        
        return comparison 