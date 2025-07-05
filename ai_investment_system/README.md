# AI       f V X e   v2.0

  I Ȋ          f x   V X e  

##     

-      Ԏ  ł̎  v \  
-    s p ^ [   ̊w K Ɖ  
-    X N      ]  
-  ΏƊw K ɂ 鐸 x    
-   I ȃo b N e X g @ \

##  Z b g A b v

1.  ˑ  ֌W ̃C   X g [  :
   ```bash
   pip install -r requirements.txt
   ```

2.    ϐ  ̐ݒ :
   ```bash
   cp .env.template .env
   # .env t @ C    ҏW    API L [  ݒ 
   ```

3.  V X e   ̋N  :
   ```bash
   python main.py
   ```

##  f B   N g   \  

```
ai_investment_system/
       data/                #  f [ ^ f B   N g  
            raw/            #    f [ ^
            processed/      #      ς݃f [ ^
            features/       #      ʃf [ ^
       models/             #  w K ς݃  f  
       config/             #  ݒ t @ C  
       scripts/            # Python X N   v g
       notebooks/          # Jupyter notebooks
       tests/              #  e X g t @ C  
       logs/               #    O t @ C  
       results/            #    ʃf [ ^
       backtest/           #  o b N e X g    
       deployment/         #  f v   C     g p t @ C  
```

##  g p   @

###  f [ ^   W
```python
from scripts.data_collector import DataCollector

collector = DataCollector(config)
data = collector.fetch_stock_data('AAPL', '1y')
```

###      ʍ쐬
```python
from scripts.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(config)
features = engineer.create_technical_features(data)
```

###    s p ^ [   w K
```python
from scripts.failure_learner import FailureLearner

learner = FailureLearner(config)
pattern = learner.classify_failure(data, prediction, actual)
```

###  ]  
```python
from scripts.evaluator import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(config)
results = evaluator.evaluate_predictions(predictions, actual_data)
```

##  ݒ 

###  ]     Ԏ 
- 21   i1     j
- 63   i3     j
- 126   i6     j
- 252   i1 N j

###      
- 1     F5% ȏ ̎  v
- 3     F10% ȏ ̎  v
- 6     F15% ȏ ̎  v
- 1 N F20% ȏ ̎  v

###    s p ^ [  
-    {   e B   e B N   b V  
-    Z f B X A | C   g
-  Z N ^ [   [ e [ V    
-         @

##    C Z   X

MIT License
