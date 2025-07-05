
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

class FailureLearner:
    def __init__(self, config):
        self.config = config
        self.failure_patterns = {}
    
    def classify_failure(self, stock_data, prediction, actual):
        """失敗パターンの分類"""
        if actual < -0.2:  # 20%以上の下落
            features = self.extract_failure_features(stock_data)
            pattern = self.identify_failure_pattern(features)
            
            self.failure_patterns.setdefault(pattern, []).append({
                'stock_data': stock_data,
                'prediction': prediction,
                'actual': actual,
                'features': features
            })
            
            return pattern
        return None
    
    def extract_failure_features(self, stock_data):
        """失敗時の特徴量抽出"""
        features = {}
        
        # ボラティリティ
        features['volatility'] = stock_data['Close'].pct_change().std()
        
        # 出来高異常
        features['volume_anomaly'] = (
            stock_data['Volume'].iloc[-1] / stock_data['Volume'].rolling(20).mean().iloc[-1]
        )
        
        # 価格モメンタム
        features['momentum'] = (
            stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-20] - 1
        )
        
        return features
    
    def contrastive_loss(self, anchor, positive, negative, margin=1.0):
        """対照学習損失"""
        pos_distance = torch.norm(anchor - positive, dim=1)
        neg_distance = torch.norm(anchor - negative, dim=1)
        
        loss = torch.clamp(pos_distance - neg_distance + margin, min=0.0)
        return loss.mean()
