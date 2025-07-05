
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fetch_stock_data(self, symbol, period="1y"):
        """�����f�[�^�̎擾"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_financial_data(self, symbol):
        """�����f�[�^�̎擾"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            return {"info": info, "financials": financials}
        except Exception as e:
            self.logger.error(f"Error fetching financial data for {symbol}: {e}")
            return None
