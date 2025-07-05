# AI投資判断システム v2.0

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
