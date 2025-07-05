#!/usr/bin/env python3
"""
AI投資判断システム - 包括的セットアップスクリプト
Version: 2.0
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AIInvestmentSetup:
    """AI投資判断システムのセットアップとセッティング"""
    
    def __init__(self, base_dir: str = "ai_investment_system"):
        self.base_dir = Path(base_dir)
        self.config = {}
        self.setup_logging()
        
    def setup_logging(self):
        """ログシステムの設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('setup.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_directory_structure(self):
        """プロジェクトのディレクトリ構造を作成"""
        directories = [
            'data/raw',
            'data/processed',
            'data/features',
            'models',
            'config',
            'scripts',
            'notebooks',
            'tests',
            'logs',
            'results',
            'backtest',
            'deployment'
        ]
        
        for dir_path in directories:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {full_path}")
    
    def install_dependencies(self):
        """必要なライブラリをインストール"""
        requirements = [
            'pandas>=1.5.0',
            'numpy>=1.21.0',
            'scikit-learn>=1.1.0',
            'torch>=1.12.0',
            'transformers>=4.21.0',
            'yfinance>=0.1.87',
            'ta-lib>=0.4.25',
            'plotly>=5.11.0',
            'streamlit>=1.12.0',
            'jupyter>=1.0.0',
            'requests>=2.28.0',
            'beautifulsoup4>=4.11.0',
            'selenium>=4.5.0',
            'schedule>=1.1.0',
            'python-dotenv>=0.19.0',
            'psycopg2-binary>=2.9.0',
            'sqlalchemy>=1.4.0',
            'fastapi>=0.85.0',
            'uvicorn>=0.18.0',
            'redis>=4.3.0',
            'celery>=5.2.0'
        ]
        
        self.logger.info("Installing dependencies...")
        with open(self.base_dir / 'requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 
                str(self.base_dir / 'requirements.txt')
            ])
            self.logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
    
    def create_config_files(self):
        """設定ファイルの作成"""
        
        # メイン設定ファイル
        main_config = {
            "system": {
                "name": "AI Investment System",
                "version": "2.0",
                "timezone": "Asia/Tokyo",
                "log_level": "INFO"
            },
            "data": {
                "sources": ["yahoo", "alpha_vantage", "quandl"],
                "update_frequency": "daily",
                "historical_period": "5y",
                "cache_duration": 3600
            },
            "model": {
                "type": "ensemble",
                "models": ["lstm", "transformer", "xgboost"],
                "training_window": 252,
                "validation_split": 0.2,
                "test_split": 0.1
            },
            "trading": {
                "risk_free_rate": 0.02,
                "max_position_size": 0.1,
                "stop_loss": 0.05,
                "take_profit": 0.15,
                "rebalance_frequency": "monthly"
            },
            "evaluation": {
                "time_horizons": [21, 63, 126, 252],
                "success_thresholds": {
                    "21": 0.05,
                    "63": 0.10,
                    "126": 0.15,
                    "252": 0.20
                },
                "risk_metrics": ["sharpe", "calmar", "max_drawdown"]
            }
        }
        
        # 失敗パターン設定
        failure_config = {
            "patterns": {
                "high_volatility_crash": {
                    "threshold": -0.2,
                    "lookback_window": 30,
                    "features": ["volatility", "volume", "momentum"]
                },
                "earnings_disappointment": {
                    "threshold": -0.15,
                    "lookback_window": 5,
                    "features": ["eps_surprise", "revenue_growth"]
                },
                "sector_rotation": {
                    "threshold": -0.1,
                    "lookback_window": 21,
                    "features": ["sector_momentum", "relative_strength"]
                },
                "liquidity_crisis": {
                    "threshold": -0.25,
                    "lookback_window": 7,
                    "features": ["bid_ask_spread", "volume_ratio"]
                }
            },
            "learning": {
                "contrastive_margin": 1.0,
                "failure_weight": 0.3,
                "contrastive_weight": 0.2
            }
        }
        
        # 設定ファイルの保存
        with open(self.base_dir / 'config/main_config.json', 'w') as f:
            json.dump(main_config, f, indent=2)
        
        with open(self.base_dir / 'config/failure_config.json', 'w') as f:
            json.dump(failure_config, f, indent=2)
        
        # 環境変数テンプレート
        env_template = """
# API Keys
ALPHA_VANTAGE_API_KEY=your_api_key_here
QUANDL_API_KEY=your_api_key_here
FRED_API_KEY=your_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost/investment_db
REDIS_URL=redis://localhost:6379

# Trading APIs
BROKER_API_KEY=your_broker_api_key
BROKER_SECRET_KEY=your_broker_secret_key

# Notification
SLACK_WEBHOOK_URL=your_slack_webhook_url
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
"""
        
        with open(self.base_dir / '.env.template', 'w') as f:
            f.write(env_template)
        
        self.logger.info("Configuration files created")
    
    def create_core_modules(self):
        """コアモジュールの作成"""
        
        # データ収集モジュール
        data_collector = '''
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fetch_stock_data(self, symbol, period="1y"):
        """株価データの取得"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_financial_data(self, symbol):
        """財務データの取得"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            return {"info": info, "financials": financials}
        except Exception as e:
            self.logger.error(f"Error fetching financial data for {symbol}: {e}")
            return None
'''
        
        # 特徴量エンジニアリングモジュール
        feature_engineer = '''
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
'''
        
        # 失敗パターン学習モジュール
        failure_learner = '''
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
'''
        
        # モジュールファイルの保存
        modules = {
            'data_collector.py': data_collector,
            'feature_engineer.py': feature_engineer,
            'failure_learner.py': failure_learner
        }
        
        for filename, content in modules.items():
            with open(self.base_dir / 'scripts' / filename, 'w') as f:
                f.write(content)
        
        self.logger.info("Core modules created")
    
    def create_evaluation_system(self):
        """評価システムの作成"""
        
        evaluator_code = '''
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
'''
        
        with open(self.base_dir / 'scripts/evaluator.py', 'w') as f:
            f.write(evaluator_code)
        
        self.logger.info("Evaluation system created")
    
    def create_sample_data(self):
        """サンプルデータの作成"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, period='2y')
                data.to_csv(self.base_dir / 'data/raw' / f'{symbol}.csv')
                self.logger.info(f"Downloaded sample data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error downloading {symbol}: {e}")
    
    def create_startup_script(self):
        """起動スクリプトの作成"""
        
        startup_code = '''#!/usr/bin/env python3
"""
AI投資判断システム - 起動スクリプト
"""

import sys
import json
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from scripts.data_collector import DataCollector
from scripts.feature_engineer import FeatureEngineer
from scripts.failure_learner import FailureLearner
from scripts.evaluator import ComprehensiveEvaluator

def load_config():
    """設定ファイルの読み込み"""
    with open('config/main_config.json', 'r') as f:
        main_config = json.load(f)
    
    with open('config/failure_config.json', 'r') as f:
        failure_config = json.load(f)
    
    return main_config, failure_config

def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("AI投資判断システムを起動します...")
    
    # 設定読み込み
    main_config, failure_config = load_config()
    
    # モジュール初期化
    data_collector = DataCollector(main_config)
    feature_engineer = FeatureEngineer(main_config)
    failure_learner = FailureLearner(failure_config)
    evaluator = ComprehensiveEvaluator(main_config)
    
    logger.info("システムの初期化が完了しました")
    logger.info("データ収集、特徴量エンジニアリング、失敗パターン学習が利用可能です")

if __name__ == "__main__":
    main()
'''
        
        with open(self.base_dir / 'main.py', 'w') as f:
            f.write(startup_code)
        
        # 実行権限を付与
        os.chmod(self.base_dir / 'main.py', 0o755)
        
        self.logger.info("Startup script created")
    
    def create_documentation(self):
        """ドキュメントの作成"""
        
        readme_content = '''# AI投資判断システム v2.0

包括的な株式投資判断支援システム

## 特徴

- 多時間軸での収益予測
- 失敗パターンの学習と回避
- リスク調整後評価
- 対照学習による精度向上
- 包括的なバックテスト機能

## セットアップ

1. 依存関係のインストール:
   ```bash
   pip install -r requirements.txt
   ```

2. 環境変数の設定:
   ```bash
   cp .env.template .env
   # .envファイルを編集してAPIキーを設定
   ```

3. システムの起動:
   ```bash
   python main.py
   ```

## ディレクトリ構成

```
ai_investment_system/
├── data/                # データディレクトリ
│   ├── raw/            # 生データ
│   ├── processed/      # 処理済みデータ
│   └── features/       # 特徴量データ
├── models/             # 学習済みモデル
├── config/             # 設定ファイル
├── scripts/            # Pythonスクリプト
├── notebooks/          # Jupyter notebooks
├── tests/              # テストファイル
├── logs/               # ログファイル
├── results/            # 結果データ
├── backtest/           # バックテスト結果
└── deployment/         # デプロイメント用ファイル
```

## 使用方法

### データ収集
```python
from scripts.data_collector import DataCollector

collector = DataCollector(config)
data = collector.fetch_stock_data('AAPL', '1y')
```

### 特徴量作成
```python
from scripts.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(config)
features = engineer.create_technical_features(data)
```

### 失敗パターン学習
```python
from scripts.failure_learner import FailureLearner

learner = FailureLearner(config)
pattern = learner.classify_failure(data, prediction, actual)
```

### 評価
```python
from scripts.evaluator import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(config)
results = evaluator.evaluate_predictions(predictions, actual_data)
```

## 設定

### 評価時間軸
- 21日（1ヶ月）
- 63日（3ヶ月）
- 126日（6ヶ月）
- 252日（1年）

### 成功基準
- 1ヶ月：5%以上の収益
- 3ヶ月：10%以上の収益
- 6ヶ月：15%以上の収益
- 1年：20%以上の収益

### 失敗パターン
- 高ボラティリティクラッシュ
- 決算ディスアポイント
- セクターローテーション
- 流動性危機

## ライセンス

MIT License
'''
        
        with open(self.base_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        self.logger.info("Documentation created")
    
    def run_setup(self):
        """セットアップの実行"""
        self.logger.info("AI投資判断システムのセットアップを開始します...")
        
        try:
            self.create_directory_structure()
            self.install_dependencies()
            self.create_config_files()
            self.create_core_modules()
            self.create_evaluation_system()
            self.create_sample_data()
            self.create_startup_script()
            self.create_documentation()
            
            self.logger.info("=" * 50)
            self.logger.info("セットアップが完了しました！")
            self.logger.info("=" * 50)
            self.logger.info(f"プロジェクトディレクトリ: {self.base_dir.absolute()}")
            self.logger.info("次のステップ:")
            self.logger.info("1. .env.templateを.envにコピーしてAPIキーを設定")
            self.logger.info("2. python main.pyでシステムを起動")
            self.logger.info("3. notebooks/ディレクトリでJupyter notebookを使用")
            
        except Exception as e:
            self.logger.error(f"セットアップ中にエラーが発生しました: {e}")
            raise

if __name__ == "__main__":
    setup = AIInvestmentSetup()
    setup.run_setup()
