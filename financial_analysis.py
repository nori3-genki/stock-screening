 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """
    包括的な財務分析クラス
    """
    
    def __init__(self, api_key: str = None):
        """
        初期化
        
        Args:
            api_key: Alpha Vantage API キー（環境変数から取得も可能）
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            logger.warning("API キーが設定されていません。環境変数 ALPHA_VANTAGE_API_KEY を設定してください。")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.rate_limit_calls = 0
        self.rate_limit_reset = time.time()
        
        # 日本の祝日・休場日（簡易版）
        self.japan_holidays = [
            "2024-01-01", "2024-01-08", "2024-02-11", "2024-02-12", "2024-02-23",
            "2024-03-20", "2024-04-29", "2024-05-03", "2024-05-04", "2024-05-05",
            "2024-05-06", "2024-07-15", "2024-08-11", "2024-08-12", "2024-09-16",
            "2024-09-22", "2024-09-23", "2024-10-14", "2024-11-03", "2024-11-04",
            "2024-11-23", "2024-12-30", "2024-12-31"
        ]
    
    def _handle_rate_limit(self):
        """
        APIレート制限の管理
        """
        current_time = time.time()
        if current_time - self.rate_limit_reset > 60:  # 1分でリセット
            self.rate_limit_calls = 0
            self.rate_limit_reset = current_time
        
        if self.rate_limit_calls >= 5:  # 1分間に5回まで
            sleep_time = 60 - (current_time - self.rate_limit_reset)
            if sleep_time > 0:
                logger.info(f"レート制限により {sleep_time:.1f} 秒待機中...")
                time.sleep(sleep_time)
                self.rate_limit_calls = 0
                self.rate_limit_reset = time.time()
    
    def _make_api_request(self, params: Dict) -> Optional[Dict]:
        """
        API リクエストの実行（エラーハンドリング付き）
        
        Args:
            params: APIパラメータ
            
        Returns:
            APIレスポンス または None
        """
        if not self.api_key:
            logger.error("API キーが設定されていません")
            return None
        
        self._handle_rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # APIエラーチェック
            if 'Error Message' in data:
                logger.error(f"API エラー: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"API 制限: {data['Note']}")
                time.sleep(60)  # 1分待機
                return self._make_api_request(params)
            
            self.rate_limit_calls += 1
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ネットワークエラー: {e}")
            return None
        except ValueError as e:
            logger.error(f"JSON解析エラー: {e}")
            return None
    
    def _format_japanese_ticker(self, ticker: str) -> str:
        """
        日本株のティッカーシンボルを正規化
        
        Args:
            ticker: 元のティッカーシンボル
            
        Returns:
            正規化されたティッカーシンボル
        """
        # 日本株の場合、.T を追加
        if ticker.isdigit() and len(ticker) == 4:
            return f"{ticker}.T"
        return ticker
    
    def calculate_financial_ratios(self, overview_data: Dict) -> Dict:
        """
        財務比率を計算
        
        Args:
            overview_data: 企業概要データ
            
        Returns:
            財務比率の辞書
        """
        try:
            ratios = {}
            
            # 収益性指標
            ratios['PER'] = self._safe_float(overview_data.get('PERatio'))
            ratios['PBR'] = self._safe_float(overview_data.get('PriceToBookRatio'))
            ratios['ROE'] = self._safe_float(overview_data.get('ReturnOnEquityTTM'))
            ratios['ROA'] = self._safe_float(overview_data.get('ReturnOnAssetsTTM'))
            ratios['利益率'] = self._safe_float(overview_data.get('ProfitMargin'))
            
            # 成長性指標
            ratios['売上成長率'] = self._safe_float(overview_data.get('RevenueGrowthTTM'))
            ratios['EPS成長率'] = self._safe_float(overview_data.get('EPSGrowthTTM'))
            
            # 財務健全性指標
            ratios['負債比率'] = self._safe_float(overview_data.get('DebtToEquityRatio'))
            ratios['流動比率'] = self._safe_float(overview_data.get('CurrentRatio'))
            
            # 配当関連
            ratios['配当利回り'] = self._safe_float(overview_data.get('DividendYield'))
            ratios['配当性向'] = self._safe_float(overview_data.get('PayoutRatio'))
            
            # バリュエーション
            ratios['PEG'] = self._safe_float(overview_data.get('PEGRatio'))
            ratios['PSR'] = self._safe_float(overview_data.get('PriceToSalesRatio'))
            
            # 追加指標
            ratios['Beta'] = self._safe_float(overview_data.get('Beta'))
            ratios['52週高値'] = self._safe_float(overview_data.get('52WeekHigh'))
            ratios['52週安値'] = self._safe_float(overview_data.get('52WeekLow'))
            
            return ratios
            
        except Exception as e:
            logger.error(f"財務比率計算エラー: {e}")
            return {}
    
    def _safe_float(self, value) -> Optional[float]:
        """
        安全な浮動小数点変換
        
        Args:
            value: 変換する値
            
        Returns:
            浮動小数点数 または None
        """
        if value is None or value == 'None' or value == '-':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def analyze_income_statement_trend(self, income_data: Dict) -> Dict:
        """
        損益計算書の推移分析
        
        Args:
            income_data: 損益計算書データ
            
        Returns:
            分析結果
        """
        try:
            if 'annualReports' not in income_data:
                logger.error("年次レポートデータが見つかりません")
                return {}
            
            annual_reports = income_data['annualReports']
            if not annual_reports:
                logger.error("年次レポートが空です")
                return {}
            
            # データフレームに変換
            df = pd.DataFrame(annual_reports)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding')
            
            # 数値データの変換
            numeric_columns = [
                'totalRevenue', 'grossProfit', 'operatingIncome', 'netIncome',
                'ebitda', 'operatingExpenses', 'researchAndDevelopment'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 成長率の計算
            growth_rates = {}
            for col in numeric_columns:
                if col in df.columns and len(df[col].dropna()) > 1:
                    growth_rates[f'{col}_growth'] = df[col].pct_change().mean()
            
            # 利益率の計算
            margins = {}
            if 'totalRevenue' in df.columns and 'grossProfit' in df.columns:
                margins['gross_margin'] = (df['grossProfit'] / df['totalRevenue']).mean()
            
            if 'totalRevenue' in df.columns and 'operatingIncome' in df.columns:
                margins['operating_margin'] = (df['operatingIncome'] / df['totalRevenue']).mean()
            
            if 'totalRevenue' in df.columns and 'netIncome' in df.columns:
                margins['net_margin'] = (df['netIncome'] / df['totalRevenue']).mean()
            
            # トレンド分析
            trend_analysis = {
                'データ期間': f"{df['fiscalDateEnding'].min().strftime('%Y-%m-%d')} - {df['fiscalDateEnding'].max().strftime('%Y-%m-%d')}",
                '年数': len(df),
                '成長率': growth_rates,
                '利益率': margins,
                '最新年度': df.iloc[-1].to_dict() if not df.empty else {},
                'データフレーム': df
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"損益計算書分析エラー: {e}")
            return {}
    
    def analyze_balance_sheet_trend(self, balance_sheet_data: Dict) -> Dict:
        """
        貸借対照表の推移分析
        
        Args:
            balance_sheet_data: 貸借対照表データ
            
        Returns:
            分析結果
        """
        try:
            if 'annualReports' not in balance_sheet_data:
                logger.error("年次レポートデータが見つかりません")
                return {}
            
            annual_reports = balance_sheet_data['annualReports']
            if not annual_reports:
                logger.error("年次レポートが空です")
                return {}
            
            # データフレームに変換
            df = pd.DataFrame(annual_reports)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding')
            
            # 数値データの変換
            numeric_columns = [
                'totalAssets', 'totalLiabilities', 'totalShareholderEquity',
                'currentAssets', 'currentLiabilities', 'longTermDebt',
                'cash', 'inventory', 'totalDebt'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 財務比率の計算
            ratios = {}
            
            # 流動比率
            if 'currentAssets' in df.columns and 'currentLiabilities' in df.columns:
                ratios['current_ratio'] = (df['currentAssets'] / df['currentLiabilities']).mean()
            
            # 負債比率
            if 'totalLiabilities' in df.columns and 'totalAssets' in df.columns:
                ratios['debt_ratio'] = (df['totalLiabilities'] / df['totalAssets']).mean()
            
            # 自己資本比率
            if 'totalShareholderEquity' in df.columns and 'totalAssets' in df.columns:
                ratios['equity_ratio'] = (df['totalShareholderEquity'] / df['totalAssets']).mean()
            
            # 長期負債比率
            if 'longTermDebt' in df.columns and 'totalAssets' in df.columns:
                ratios['long_term_debt_ratio'] = (df['longTermDebt'] / df['totalAssets']).mean()
            
            analysis_result = {
                'データ期間': f"{df['fiscalDateEnding'].min().strftime('%Y-%m-%d')} - {df['fiscalDateEnding'].max().strftime('%Y-%m-%d')}",
                '年数': len(df),
                '財務比率': ratios,
                '最新年度': df.iloc[-1].to_dict() if not df.empty else {},
                'データフレーム': df
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"貸借対照表分析エラー: {e}")
            return {}
    
    def analyze_cash_flow_trend(self, cash_flow_data: Dict) -> Dict:
        """
        キャッシュフロー計算書の推移分析
        
        Args:
            cash_flow_data: キャッシュフローデータ
            
        Returns:
            分析結果
        """
        try:
            if 'annualReports' not in cash_flow_data:
                logger.error("年次レポートデータが見つかりません")
                return {}
            
            annual_reports = cash_flow_data['annualReports']
            if not annual_reports:
                logger.error("年次レポートが空です")
                return {}
            
            # データフレームに変換
            df = pd.DataFrame(annual_reports)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding')
            
            # 数値データの変換
            numeric_columns = [
                'operatingCashflow', 'cashflowFromInvestment', 'cashflowFromFinancing',
                'netIncome', 'depreciation', 'capitalExpenditures', 'freeCashflow'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # フリーキャッシュフローの計算（データにない場合）
            if 'freeCashflow' not in df.columns or df['freeCashflow'].isna().all():
                if 'operatingCashflow' in df.columns and 'capitalExpenditures' in df.columns:
                    df['freeCashflow'] = df['operatingCashflow'] - df['capitalExpenditures']
            
            # キャッシュフロー分析
            cf_analysis = {}
            
            # 営業CFの安定性
            if 'operatingCashflow' in df.columns:
                cf_analysis['operating_cf_mean'] = df['operatingCashflow'].mean()
                cf_analysis['operating_cf_std'] = df['operatingCashflow'].std()
                cf_analysis['operating_cf_growth'] = df['operatingCashflow'].pct_change().mean()
            
            # フリーキャッシュフローの推移
            if 'freeCashflow' in df.columns:
                cf_analysis['free_cf_mean'] = df['freeCashflow'].mean()
                cf_analysis['free_cf_growth'] = df['freeCashflow'].pct_change().mean()
            
            # 営業CFと純利益の関係
            if 'operatingCashflow' in df.columns and 'netIncome' in df.columns:
                cf_analysis['cf_to_ni_ratio'] = (df['operatingCashflow'] / df['netIncome']).mean()
            
            analysis_result = {
                'データ期間': f"{df['fiscalDateEnding'].min().strftime('%Y-%m-%d')} - {df['fiscalDateEnding'].max().strftime('%Y-%m-%d')}",
                '年数': len(df),
                'キャッシュフロー分析': cf_analysis,
                '最新年度': df.iloc[-1].to_dict() if not df.empty else {},
                'データフレーム': df
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"キャッシュフロー分析エラー: {e}")
            return {}
    
    def get_company_overview(self, symbol: str) -> Optional[Dict]:
        """
        企業概要データの取得
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            企業概要データ
        """
        symbol = self._format_japanese_ticker(symbol)
        
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        return self._make_api_request(params)
    
    def get_income_statement(self, symbol: str) -> Optional[Dict]:
        """
        損益計算書データの取得
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            損益計算書データ
        """
        symbol = self._format_japanese_ticker(symbol)
        
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol
        }
        
        return self._make_api_request(params)
    
    def get_balance_sheet(self, symbol: str) -> Optional[Dict]:
        """
        貸借対照表データの取得
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            貸借対照表データ
        """
        symbol = self._format_japanese_ticker(symbol)
        
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol
        }
        
        return self._make_api_request(params)
    
    def get_cash_flow(self, symbol: str) -> Optional[Dict]:
        """
        キャッシュフローデータの取得
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            キャッシュフローデータ
        """
        symbol = self._format_japanese_ticker(symbol)
        
        params = {
            'function': 'CASH_FLOW',
            'symbol': symbol
        }
        
        return self._make_api_request(params)
    
    def create_financial_dashboard(self, symbol: str, save_path: str = None) -> bool:
        """
        財務ダッシュボードの作成
        
        Args:
            symbol: 銘柄コード
            save_path: 保存パス（Noneの場合は表示のみ）
            
        Returns:
            成功フラグ
        """
        try:
            logger.info(f"財務ダッシュボードを作成中: {symbol}")
            
            # データ取得
            overview = self.get_company_overview(symbol)
            income_statement = self.get_income_statement(symbol)
            balance_sheet = self.get_balance_sheet(symbol)
            cash_flow = self.get_cash_flow(symbol)
            
            if not overview:
                logger.error("企業概要データの取得に失敗しました")
                return False
            
            # 財務比率の計算
            ratios = self.calculate_financial_ratios(overview)
            
            # 図表の作成
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{overview.get("Name", symbol)} 財務分析ダッシュボード', fontsize=16, fontweight='bold')
            
            # 1. 財務比率の可視化
            if ratios:
                ratio_names = []
                ratio_values = []
                for name, value in ratios.items():
                    if value is not None and not np.isnan(value):
                        ratio_names.append(name)
                        ratio_values.append(value)
                
                if ratio_names:
                    axes[0, 0].barh(ratio_names[:10], ratio_values[:10])
                    axes[0, 0].set_title('主要財務比率')
                    axes[0, 0].set_xlabel('値')
                else:
                    axes[0, 0].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[0, 0].transAxes)
            
            # 2. 損益計算書の推移
            if income_statement:
                income_analysis = self.analyze_income_statement_trend(income_statement)
                if 'データフレーム' in income_analysis:
                    df = income_analysis['データフレーム']
                    if 'totalRevenue' in df.columns and not df['totalRevenue'].isna().all():
                        axes[0, 1].plot(df['fiscalDateEnding'], df['totalRevenue'], marker='o', label='売上高')
                        axes[0, 1].set_title('売上高推移')
                        axes[0, 1].set_xlabel('年度')
                        axes[0, 1].set_ylabel('売上高')
                        axes[0, 1].legend()
                        axes[0, 1].tick_params(axis='x', rotation=45)
                    else:
                        axes[0, 1].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[0, 1].transAxes)
                else:
                    axes[0, 1].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # 3. 貸借対照表の推移
            if balance_sheet:
                balance_analysis = self.analyze_balance_sheet_trend(balance_sheet)
                if 'データフレーム' in balance_analysis:
                    df = balance_analysis['データフレーム']
                    if 'totalAssets' in df.columns and not df['totalAssets'].isna().all():
                        axes[0, 2].plot(df['fiscalDateEnding'], df['totalAssets'], marker='o', label='総資産')
                        axes[0, 2].set_title('総資産推移')
                        axes[0, 2].set_xlabel('年度')
                        axes[0, 2].set_ylabel('総資産')
                        axes[0, 2].legend()
                        axes[0, 2].tick_params(axis='x', rotation=45)
                    else:
                        axes[0, 2].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[0, 2].transAxes)
                else:
                    axes[0, 2].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[0, 2].transAxes)
            
            # 4. キャッシュフロー推移
            if cash_flow:
                cf_analysis = self.analyze_cash_flow_trend(cash_flow)
                if 'データフレーム' in cf_analysis:
                    df = cf_analysis['データフレーム']
                    if 'operatingCashflow' in df.columns and not df['operatingCashflow'].isna().all():
                        axes[1, 0].plot(df['fiscalDateEnding'], df['operatingCashflow'], marker='o', label='営業CF')
                        if 'freeCashflow' in df.columns and not df['freeCashflow'].isna().all():
                            axes[1, 0].plot(df['fiscalDateEnding'], df['freeCashflow'], marker='s', label='フリーCF')
                        axes[1, 0].set_title('キャッシュフロー推移')
                        axes[1, 0].set_xlabel('年度')
                        axes[1, 0].set_ylabel('キャッシュフロー')
                        axes[1, 0].legend()
                        axes[1, 0].tick_params(axis='x', rotation=45)
                    else:
                        axes[1, 0].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[1, 0].transAxes)
                else:
                    axes[1, 0].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # 5. 収益性指標の比較
            profitability_metrics = ['ROE', 'ROA', '利益率']
            prof_values = [ratios.get(metric, 0) for metric in profitability_metrics if ratios.get(metric) is not None]
            prof_labels = [metric for metric in profitability_metrics if ratios.get(metric) is not None]
            
            if prof_values:
                axes[1, 1].bar(prof_labels, prof_values)
                axes[1, 1].set_title('収益性指標')
                axes[1, 1].set_ylabel('比率')
            else:
                axes[1, 1].text(0.5, 0.5, 'データなし', ha='center', va='center', transform=axes[1, 1].transAxes)
            
            # 6. 企業基本情報
            info_text = f"""
            銘柄コード: {symbol}
            企業名: {overview.get('Name', 'N/A')}
            セクター: {overview.get('Sector', 'N/A')}
            業界: {overview.get('Industry', 'N/A')}
            市場: {overview.get('Exchange', 'N/A')}
            時価総額: {overview.get('MarketCapitalization', 'N/A')}
            従業員数: {overview.get('FullTimeEmployees', 'N/A')}
            """
            
            axes[1, 2].text(0.05, 0.95, info_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('企業基本情報')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ダッシュボードを保存しました: {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {e}")
            return False
    
    def generate_financial_report(self, symbol: str, save_path: str = None) -> str:
        """
        財務レポートの生成
        
        Args:
            symbol: 銘柄コード
            save_path: 保存パス（Noneの場合は文字列で返す）
            
        Returns:
            レポート内容
        """
        try:
            logger.info(f"財務レポートを生成中: {symbol}")
            
            # データ取得
            overview = self.get_company_overview(symbol)
            income_statement = self.get_income_statement(symbol)
            balance_sheet = self.get_balance_sheet(symbol)
            cash_flow = self.get_cash_flow(symbol)
            
            if not overview:
                logger.error("企業概要データの取得に失敗しました")
                return "データの取得に失敗しました"
            
            # 分析実行
            ratios = self.calculate_financial_ratios(overview)
            income_analysis = self.analyze_income_statement_trend(income_statement) if income_statement else {}
            balance_analysis = self.analyze_balance_sheet_trend(balance_sheet) if balance_sheet else {}
            cf_analysis = self.analyze_cash_flow_trend(cash_flow) if cash_flow else {}
            
            # レポート生成
            report = f"""
# 財務分析レポート

## 企業概要
- 銘柄コード: {symbol}
- 企業名: {overview.get('Name', 'N/A')}
- セクター: {overview.get('Sector', 'N/A')}
- 業界: {overview.get('Industry', 'N/A')}
- 市場: {overview.get('Exchange', 'N/A')}
- 時価総額: {overview.get('MarketCapitalization', 'N/A')}
- 従業員数: {overview.get('FullTimeEmployees', 'N/A')}

## 財務比率分析
"""
            
            if ratios:
                report += "\n### 収益性指標\n"
                if ratios.get('PER'):
                    report += f"- PER: {ratios['PER']:.2f}\n"
                if ratios.get('ROE'):
                    report += f"- ROE: {ratios['ROE']:.2%}\n"
                if ratios.get('ROA'):
                    report += f"- ROA: {ratios['ROA']:.2%}\n"
                if ratios.get('利益率'):
                    report += f"- 利益率: {ratios['利益率']:.2%}\n"
                
                report += "\n### 成長性指標\n"
                if ratios.get('売上成長率'):
                    report += f"- 売上成長率: {ratios['売上成長率']:.2%}\n"
                if ratios.get('EPS成長率'):
                    report += f"- EPS成長率: {ratios['EPS成長率']:.2%}\n"
                
                report += "\n### 財務健全性指標\n"
                if ratios.get('流動比率'):
                    report += f"- 流動比率: {ratios['流動比率']:.2f}\n"
                if ratios.get('負債比率'):
                    report += f"- 負債比率: {ratios['負債比率']:.2f}\n"
                
                report += "\n### バリュエーション指標\n"
                if ratios.get('PBR'):
                    report += f"- PBR: {ratios['PBR']:.2f}\n"
                if ratios.get('PEG'):
                    report += f"- PEG: {ratios['PEG']:.2f}\n"
                if ratios.get('PSR'):
                    report += f"- PSR: {ratios['PSR']:.2f}\n"
                
                report += "\n### 配当関連\n"
                if ratios.get('配当利回り'):
                    report += f"- 配当利回り: {ratios['配当利回り']:.2%}\n"
                if ratios.get('配当性向'):
                    report += f"- 配当性向: {ratios['配当性向']:.2%}\n"
            
            # 損益計算書分析
            if income_analysis:
                report += "\n## 損益計算書分析\n"
                if 'データ期間' in income_analysis:
                    report += f"- 分析期間: {income_analysis['データ期間']}\n"
                if '年数' in income_analysis:
                    report += f"- 分析年数: {income_analysis['年数']}年\n"
                
                if '成長率' in income_analysis:
                    report += "\n### 成長率分析\n"
                    for metric, growth in income_analysis['成長率'].items():
                        if growth is not None and not np.isnan(growth):
                            report += f"- {metric}: {growth:.2%}\n"
                
                if '利益率' in income_analysis:
                    report += "\n### 利益率分析\n"
                    for metric, margin in income_analysis['利益率'].items():
                        if margin is not None and not np.isnan(margin):
                            report += f"- {metric}: {margin:.2%}\n"
            
            # 貸借対照表分析
            if balance_analysis:
                report += "\n## 貸借対照表分析\n"
                if 'データ期間' in balance_analysis:
                    report += f"- 分析期間: {balance_analysis['データ期間']}\n"
                
                if '財務比率' in balance_analysis:
                    report += "\n### 財務比率\n"
                    for metric, ratio in balance_analysis['財務比率'].items():
                        if ratio is not None and not np.isnan(ratio):
                            report += f"- {metric}: {ratio:.2f}\n"
            
            # キャッシュフロー分析
            if cf_analysis:
                report += "\n## キャッシュフロー分析\n"
                if 'データ期間' in cf_analysis:
                    report += f"- 分析期間: {cf_analysis['データ期間']}\n"
                
                if 'キャッシュフロー分析' in cf_analysis:
                    report += "\n### キャッシュフロー指標\n"
                    cf_metrics = cf_analysis['キャッシュフロー分析']
                    for metric, value in cf_metrics.items():
                        if value is not None and not np.isnan(value):
                            if 'growth' in metric:
                                report += f"- {metric}: {value:.2%}\n"
                            elif 'ratio' in metric:
                                report += f"- {metric}: {value:.2f}\n"
                            else:
                                report += f"- {metric}: {value:,.0f}\n"
            
            # 投資判断
            report += "\n## 投資判断の参考情報\n"
            
            # PERによる判断
            if ratios.get('PER'):
                per = ratios['PER']
                if per < 10:
                    report += "- PER: 割安水準（10倍未満）\n"
                elif per < 20:
                    report += "- PER: 適正水準（10-20倍）\n"
                else:
                    report += "- PER: 割高水準（20倍超）\n"
            
            # ROEによる判断
            if ratios.get('ROE'):
                roe = ratios['ROE']
                if roe > 0.15:
                    report += "- ROE: 優良水準（15%超）\n"
                elif roe > 0.10:
                    report += "- ROE: 良好水準（10-15%）\n"
                else:
                    report += "- ROE: 改善余地あり（10%未満）\n"
            
            # 配当利回りによる判断
            if ratios.get('配当利回り'):
                div_yield = ratios['配当利回り']
                if div_yield > 0.04:
                    report += "- 配当利回り: 高配当（4%超）\n"
                elif div_yield > 0.02:
                    report += "- 配当利回り: 標準的（2-4%）\n"
                else:
                    report += "- 配当利回り: 低配当（2%未満）\n"
            
            # 注意事項
            report += "\n## 注意事項\n"
            report += "- 本レポートは過去の財務データに基づく分析です\n"
            report += "- 投資判断は自己責任で行ってください\n"
            report += "- 市場環境や業界動向も考慮することをお勧めします\n"
            report += "- データは取得時点のものであり、最新情報は各種情報源で確認してください\n"
            
            report += f"\n---\n生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"レポートを保存しました: {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return f"レポート生成中にエラーが発生しました: {e}"
    
    def is_market_open(self, market: str = "US") -> bool:
        """
        市場開場時間の判定
        
        Args:
            market: 市場（"US", "JP"）
            
        Returns:
            開場中かどうか
        """
        try:
            now = datetime.now()
            
            if market == "US":
                # 米国市場（EST）
                # 平日 9:30-16:00 EST
                if now.weekday() >= 5:  # 土日
                    return False
                
                # 簡易的な時間判定（タイムゾーン考慮は別途実装）
                return 9 <= now.hour <= 16
                
            elif market == "JP":
                # 日本市場（JST）
                # 平日 9:00-15:00 JST（11:30-12:30は昼休み）
                if now.weekday() >= 5:  # 土日
                    return False
                
                # 祝日チェック
                date_str = now.strftime('%Y-%m-%d')
                if date_str in self.japan_holidays:
                    return False
                
                # 取引時間チェック
                if 9 <= now.hour < 11 or 12 < now.hour < 15:
                    return True
                elif now.hour == 11 and now.minute < 30:
                    return True
                elif now.hour == 12 and now.minute > 30:
                    return True
                
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"市場開場時間判定エラー: {e}")
            return False
    
    def compare_companies(self, symbols: List[str], save_path: str = None) -> bool:
        """
        複数企業の比較分析
        
        Args:
            symbols: 比較する銘柄コードのリスト
            save_path: 保存パス
            
        Returns:
            成功フラグ
        """
        try:
            logger.info(f"企業比較分析を開始: {symbols}")
            
            companies_data = {}
            
            # 各企業のデータを取得
            for symbol in symbols:
                overview = self.get_company_overview(symbol)
                if overview:
                    companies_data[symbol] = {
                        'overview': overview,
                        'ratios': self.calculate_financial_ratios(overview)
                    }
            
            if not companies_data:
                logger.error("比較データが取得できませんでした")
                return False
            
            # 比較チャートの作成
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('企業比較分析', fontsize=16, fontweight='bold')
            
            # 1. PER比較
            pers = []
            company_names = []
            for symbol, data in companies_data.items():
                per = data['ratios'].get('PER')
                if per is not None:
                    pers.append(per)
                    company_names.append(symbol)
            
            if pers:
                axes[0, 0].bar(company_names, pers)
                axes[0, 0].set_title('PER比較')
                axes[0, 0].set_ylabel('PER')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. ROE比較
            roes = []
            company_names_roe = []
            for symbol, data in companies_data.items():
                roe = data['ratios'].get('ROE')
                if roe is not None:
                    roes.append(roe * 100)  # パーセント表示
                    company_names_roe.append(symbol)
            
            if roes:
                axes[0, 1].bar(company_names_roe, roes)
                axes[0, 1].set_title('ROE比較')
                axes[0, 1].set_ylabel('ROE (%)')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. 配当利回り比較
            div_yields = []
            company_names_div = []
            for symbol, data in companies_data.items():
                div_yield = data['ratios'].get('配当利回り')
                if div_yield is not None:
                    div_yields.append(div_yield * 100)  # パーセント表示
                    company_names_div.append(symbol)
            
            if div_yields:
                axes[1, 0].bar(company_names_div, div_yields)
                axes[1, 0].set_title('配当利回り比較')
                axes[1, 0].set_ylabel('配当利回り (%)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. 時価総額比較
            market_caps = []
            company_names_mc = []
            for symbol, data in companies_data.items():
                market_cap = data['overview'].get('MarketCapitalization')
                if market_cap and market_cap != 'None':
                    try:
                        market_caps.append(float(market_cap) / 1e9)  # 10億単位
                        company_names_mc.append(symbol)
                    except:
                        pass
            
            if market_caps:
                axes[1, 1].bar(company_names_mc, market_caps)
                axes[1, 1].set_title('時価総額比較')
                axes[1, 1].set_ylabel('時価総額 (10億)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"比較チャートを保存しました: {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            logger.error(f"企業比較分析エラー: {e}")
            return False


# 使用例
def main():
    """
    メイン実行例
    """
    # APIキーを設定（環境変数から読み込み）
    analyzer = FinancialAnalyzer()
    
    # 使用例1: 単一企業の分析
    symbol = "AAPL"  # Apple
    print(f"=== {symbol} の財務分析 ===")
    
    # 企業概要の取得
    overview = analyzer.get_company_overview(symbol)
    if overview:
        print(f"企業名: {overview.get('Name')}")
        print(f"セクター: {overview.get('Sector')}")
        print(f"時価総額: {overview.get('MarketCapitalization')}")
    
    # 財務比率の計算
    if overview:
        ratios = analyzer.calculate_financial_ratios(overview)
        print("\n主要財務指標:")
        for name, value in ratios.items():
            if value is not None:
                print(f"  {name}: {value}")
    
    # ダッシュボードの作成
    analyzer.create_financial_dashboard(symbol, f"{symbol}_dashboard.png")
    
    # 財務レポートの生成
    report = analyzer.generate_financial_report(symbol, f"{symbol}_report.md")
    print("\n=== 財務レポート ===")
    print(report[:500] + "...")  # 最初の500文字を表示
    
    # 使用例2: 複数企業の比較
    print("\n=== 企業比較分析 ===")
    symbols_to_compare = ["AAPL", "GOOGL", "MSFT"]
    analyzer.compare_companies(symbols_to_compare, "company_comparison.png")
    
    # 使用例3: 日本株の分析
    print("\n=== 日本株分析 ===")
    jp_symbol = "7203"  # トヨタ自動車
    jp_overview = analyzer.get_company_overview(jp_symbol)
    if jp_overview:
        print(f"企業名: {jp_overview.get('Name')}")
        analyzer.create_financial_dashboard(jp_symbol, f"{jp_symbol}_dashboard.png")
    
    # 市場開場時間の確認
    print(f"\n米国市場開場中: {analyzer.is_market_open('US')}")
    print(f"日本市場開場中: {analyzer.is_market_open('JP')}")


if __name__ == "__main__":
    main()
