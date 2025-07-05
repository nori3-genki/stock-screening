
import pandas as pd
import numpy as np
import talib

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
    
    def create_technical_features(self, data):
        """テクニカル指標の作成"""
        features = pd.DataFrame(index=data.index)
        
        # 移動平均
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        features['ema_12'] = data['Close'].ewm(span=12).mean()
        features['ema_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        features['rsi'] = talib.RSI(data['Close'].values, timeperiod=14)
        
        # MACD
        macd, signal, hist = talib.MACD(data['Close'].values)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # ボリンジャーバンド
        bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'].values)
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        
        return features
    
    def create_fundamental_features(self, financial_data):
        """ファンダメンタル指標の作成"""
        features = {}
        
        info = financial_data.get('info', {})
        features['pe_ratio'] = info.get('forwardPE', None)
        features['pb_ratio'] = info.get('priceToBook', None)
        features['debt_to_equity'] = info.get('debtToEquity', None)
        features['roe'] = info.get('returnOnEquity', None)
        features['roa'] = info.get('returnOnAssets', None)
        
        return features
