
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ComprehensiveEvaluator:
    def __init__(self, config):
        self.config = config
        self.time_horizons = config['evaluation']['time_horizons']
        self.success_thresholds = config['evaluation']['success_thresholds']
    
    def calculate_returns(self, stock_data, periods):
        """複数期間の収益率計算"""
        returns = {}
        for period in periods:
            returns[f'{period}d_return'] = (
                stock_data['Close'].shift(-period) / stock_data['Close'] - 1
            )
        return pd.DataFrame(returns)
    
    def define_success_labels(self, returns):
        """成功ラベルの定義"""
        labels = {}
        
        for horizon in self.time_horizons:
            threshold = self.success_thresholds[str(horizon)]
            labels[f'{horizon}d_success'] = returns[f'{horizon}d_return'] > threshold
        
        return pd.DataFrame(labels)
    
    def calculate_risk_metrics(self, returns):
        """リスク指標の計算"""
        metrics = {}
        
        # シャープレシオ
        risk_free_rate = self.config['trading']['risk_free_rate']
        metrics['sharpe_ratio'] = (returns.mean() - risk_free_rate) / returns.std()
        
        # 最大ドローダウン
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # カルマーレシオ
        annual_return = returns.mean() * 252
        metrics['calmar_ratio'] = annual_return / abs(metrics['max_drawdown'])
        
        return metrics
    
    def evaluate_predictions(self, predictions, actual_data):
        """予測結果の総合評価"""
        results = {}
        
        for horizon in self.time_horizons:
            actual_returns = self.calculate_returns(actual_data, [horizon])
            
            # 成功率
            success_rate = (
                actual_returns[f'{horizon}d_return'] > 
                self.success_thresholds[str(horizon)]
            ).mean()
            
            # 平均収益率
            avg_return = actual_returns[f'{horizon}d_return'].mean()
            
            # リスク指標
            risk_metrics = self.calculate_risk_metrics(actual_returns[f'{horizon}d_return'])
            
            results[f'{horizon}d'] = {
                'success_rate': success_rate,
                'avg_return': avg_return,
                **risk_metrics
            }
        
        return results
