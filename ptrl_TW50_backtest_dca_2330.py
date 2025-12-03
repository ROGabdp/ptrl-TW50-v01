# -*- coding: utf-8 -*-
"""
Pro Trader RL - DCA Backtest for 2330.TW
Features:
- Target: 2330.TW Only
- Capital: Start with 600k (or 0 + 600k injection). Adds 600k every year.
- Strategy:
    - Benchmark: DCA Buy every 20 days (approx 50k).
    - AI:
        - Regular: DCA Buy every 20 days (approx 50k).
        - Aggressive: If Close <= 60MA AND AI Buy Signal -> Invest ALL remaining yearly budget.
        - Sell: Only if AI Prob Diff > 0.85. No Stop Loss.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from stable_baselines3 import PPO

# --- 0. 環境與 GPU 設定 ---
def setup_environment():
    if torch.cuda.is_available():
        print(f"✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️ CUDA not available. Using CPU.")
        device = "cpu"

    PROJECT_PATH = os.getcwd()
    MODELS_PATH = os.path.join(PROJECT_PATH, 'models_paper') 
    RESULTS_PATH = os.path.join(PROJECT_PATH, 'results_paper')
    DATA_PATH = os.path.join(PROJECT_PATH, 'data')

    for path in [MODELS_PATH, RESULTS_PATH, DATA_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    return PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device

# --- 1. 資料下載 (2330.TW Only) ---
def fetch_data(data_path, start_date="2000-01-01"):
    tickers = ["2330.TW", "^TWII"]
    print(f"開始下載資料 (Start: {start_date})...")
    data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, threads=True, progress=False)
    
    clean_data = {}
    if "^TWII" in data.columns.levels[0]:
        market_index = data["^TWII"].copy()
        if isinstance(market_index.columns, pd.MultiIndex):
            market_index.columns = market_index.columns.get_level_values(0)
        clean_data["^TWII"] = market_index

    t = "2330.TW"
    try:
        df = data[t].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        df = df.dropna()
        clean_data[t] = df
    except Exception as e:
        print(f"Error downloading {t}: {e}")
        
    return clean_data

# --- 2. 特徵工程 ---
FEATURE_COLS = [
    'Norm_Close', 'Norm_Open', 'Norm_High', 'Norm_Low',
    'Norm_DC_Lower',
    'Norm_HA_Open', 'Norm_HA_High', 'Norm_HA_Low', 'Norm_HA_Close',
    'Norm_SuperTrend_1', 'Norm_SuperTrend_2',
    'Norm_RSI', 'Norm_MFI',
    'Norm_ATR_Change',
    'Norm_RS_Ratio',
    'RS_ROC_5', 'RS_ROC_10', 'RS_ROC_20', 'RS_ROC_60', 'RS_ROC_120'
]

def calculate_heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = [df['Open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha_close.iloc[i-1]) / 2)
    ha_open = pd.Series(ha_open, index=df.index)
    ha_high = pd.concat([df['High'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['Low'], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({'HA_open': ha_open, 'HA_high': ha_high, 'HA_low': ha_low, 'HA_close': ha_close})

def calculate_supertrend(df, length=14, multiplier=3.0):
    high = df['High']; low = df['Low']; close = df['Close']
    atr = AverageTrueRange(high, low, close, window=length).average_true_range().fillna(method='bfill')
    hl2 = (high + low) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    final_upperband = basic_upperband.copy(); final_lowerband = basic_lowerband.copy()
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        if basic_upperband.iloc[i] < final_upperband.iloc[i-1] or close.iloc[i-1] > final_upperband.iloc[i-1]:
            final_upperband.iloc[i] = basic_upperband.iloc[i]
        else: final_upperband.iloc[i] = final_upperband.iloc[i-1]
        if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1] or close.iloc[i-1] < final_lowerband.iloc[i-1]:
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
        else: final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
        if close.iloc[i] > final_upperband.iloc[i-1]: trend[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i-1]: trend[i] = -1
        else: trend[i] = trend[i-1]
    st = pd.Series(np.where(trend == 1, final_lowerband, final_upperband), index=df.index)
    return pd.DataFrame({'SUPERT_': st})

def calculate_features(df_in, benchmark_df):
    df = df_in.copy()
    # Basic Indicators
    df['MA60'] = df['Close'].rolling(60).mean()
    df['DC_Upper'] = df['High'].rolling(20).max().shift(1).fillna(method='bfill')
    df['DC_Lower'] = df['Low'].rolling(20).min().shift(1).fillna(method='bfill')
    df['DC_Upper_10'] = df['High'].rolling(10).max().shift(1).fillna(method='bfill')
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range().fillna(method='bfill')
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    try: df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
    except: df['MFI'] = 50.0
    
    ha = calculate_heikin_ashi(df)
    df['HA_Open'] = ha['HA_open']; df['HA_High'] = ha['HA_high']; df['HA_Low'] = ha['HA_low']; df['HA_Close'] = ha['HA_close']
    st1 = calculate_supertrend(df, length=14, multiplier=2.0)
    st2 = calculate_supertrend(df, length=21, multiplier=1.0)
    df['SuperTrend_1'] = st1.iloc[:, 0]; df['SuperTrend_2'] = st2.iloc[:, 0]
    
    base_price = df['DC_Upper'].replace(0, np.nan).fillna(method='bfill')
    for col in ['Close', 'Open', 'High', 'Low', 'DC_Lower', 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'SuperTrend_1', 'SuperTrend_2']:
        df[f'Norm_{col}'] = df[col] / base_price
    df['Norm_RSI'] = df['RSI'] / 100.0; df['Norm_MFI'] = df['MFI'] / 100.0
    df['Norm_ATR_Change'] = (df['ATR'] / df['ATR'].shift(1)).fillna(1.0)
    
    if benchmark_df is not None:
        bench_close = benchmark_df['Close'].reindex(df.index).fillna(method='ffill')
        df['RS_Raw'] = df['Close'] / bench_close
        rs_min = df['RS_Raw'].rolling(250).min(); rs_max = df['RS_Raw'].rolling(250).max()
        denominator = (rs_max - rs_min).replace(0, np.nan).fillna(1.0) + 1e-9
        df['Norm_RS_Ratio'] = ((df['RS_Raw'] - rs_min) / denominator).fillna(0.5)
        for period in [5, 10, 20, 60, 120]: df[f'RS_ROC_{period}'] = df['RS_Raw'].pct_change(period).fillna(0)
    else:
        df['Norm_RS_Ratio'] = 0.0
        for period in [5, 10, 20, 60, 120]: df[f'RS_ROC_{period}'] = 0.0
        
    df['Signal_Buy_Filter'] = (df['Close'] <= df['MA60']) # Modified Filter for Aggressive Buy
    return df.dropna()

# --- 3. Backtester Class ---
class DCABacktester:
    def __init__(self, df, buy_model, sell_model, start_date, end_date):
        self.df = df
        self.buy_model = buy_model
        self.sell_model = sell_model
        
        # Filter Dates
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        self.dates = sorted(self.df[mask].index)
        
        # Portfolio State
        self.cash = 0 # Start with 0, will inject immediately if start of year
        self.shares = 0
        self.yearly_budget = 0
        self.total_invested = 0
        
        # DCA Config
        self.dca_interval = 20
        self.days_since_last_dca = 20 # Trigger immediately
        self.yearly_injection = 600000
        self.dca_amount = 50000 # Approx 600k / 12
        
        self.trade_logs = []
        self.inventory = [] # List of {'shares', 'cost_price', 'date'} for FIFO/Selling logic

    def run(self):
        print(f"執行 DCA 回測 ({self.dates[0].date()} ~ {self.dates[-1].date()})...")
        records = []
        current_year = -1
        
        for current_date in tqdm(self.dates):
            row = self.df.loc[current_date]
            price = row['Close']
            
            # 0. Yearly Injection
            if current_date.year != current_year:
                self.cash += self.yearly_injection
                self.yearly_budget += self.yearly_injection
                current_year = current_date.year
                # print(f"[{current_date.date()}] Year Start! Added {self.yearly_injection}. Budget: {self.yearly_budget}")

            # 1. Sell Logic (High Confidence Only)
            # Check if we have shares to sell
            if self.shares > 0:
                # Calculate avg cost or check individual lots? 
                # Simplified: Check current return based on Avg Cost
                avg_cost = self.total_invested / self.shares if self.shares > 0 else 0
                current_return = price / avg_cost
                
                market_features = row[FEATURE_COLS].values.astype(np.float32)
                sell_state = np.concatenate([market_features, [current_return]])
                obs_tensor = torch.as_tensor(sell_state).unsqueeze(0).to(self.sell_model.device)
                with torch.no_grad():
                    distribution = self.sell_model.policy.get_distribution(obs_tensor)
                    probs = distribution.distribution.probs.cpu().numpy()[0]
                
                if (probs[1] - probs[0]) > 0.85:
                    self._sell(price, "AI_High_Conf", current_date)

            # 2. Buy Logic
            self.days_since_last_dca += 1
            buy_action = False
            buy_reason = ""
            buy_amount = 0
            
            # A. Aggressive Buy Check (Low Low)
            # Filter: Close <= 60MA
            if row['Signal_Buy_Filter'] and self.yearly_budget > 0:
                state = row[FEATURE_COLS].values.astype(np.float32)
                action, _ = self.buy_model.predict(state, deterministic=True)
                if action == 1:
                    buy_amount = self.yearly_budget
                    buy_reason = "AI_Aggressive_Low"
                    buy_action = True
            
            # B. Regular DCA Check
            if not buy_action and self.days_since_last_dca >= self.dca_interval and self.yearly_budget > 0:
                buy_amount = min(self.dca_amount, self.yearly_budget)
                if buy_amount > 0:
                    buy_reason = "DCA_Regular"
                    buy_action = True
            
            # Execute Buy
            if buy_action and buy_amount > 0:
                # Check if we have enough cash (Cash should track budget, but let's be safe)
                actual_invest = min(buy_amount, self.cash)
                if actual_invest > price: # Can buy at least 1 share
                    shares_bought = int(actual_invest / price)
                    cost = shares_bought * price * (1 + 0.001425)
                    
                    self.cash -= cost
                    self.yearly_budget -= actual_invest # Deduct from budget
                    self.shares += shares_bought
                    self.total_invested += cost
                    
                    self.inventory.append({'shares': shares_bought, 'cost_price': price, 'date': current_date})
                    self.trade_logs.append({
                        'Date': current_date, 'Action': 'Buy', 'Price': price, 
                        'Shares': shares_bought, 'Amount': cost, 'Reason': buy_reason
                    })
                    
                    if buy_reason == "DCA_Regular":
                        self.days_since_last_dca = 0
                    elif buy_reason == "AI_Aggressive_Low":
                        # If aggressive buy used up budget, DCA won't trigger anyway
                        pass

            # Record
            market_value = self.shares * price
            total_value = self.cash + market_value
            records.append({
                'Date': current_date, 
                'Total_Value': total_value,
                'Cash': self.cash,
                'Shares': self.shares,
                'Market_Value': market_value,
                'Yearly_Budget': self.yearly_budget
            })
            
        return pd.DataFrame(records).set_index('Date'), self.trade_logs

    def _sell(self, price, reason, date):
        revenue = self.shares * price * (1 - 0.004)
        self.cash += revenue
        
        self.trade_logs.append({
            'Date': date, 'Action': 'Sell', 'Price': price, 
            'Shares': self.shares, 'Amount': revenue, 'Reason': reason
        })
        
        self.shares = 0
        self.total_invested = 0 # Reset cost basis after full sell? Or keep tracking? 
        # Usually full sell means we exit position.
        self.inventory = []

# --- Benchmark Class (Pure DCA) ---
class BenchmarkDCA:
    def __init__(self, df, start_date, end_date):
        self.df = df
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        self.dates = sorted(self.df[mask].index)
        self.cash = 0
        self.shares = 0
        self.yearly_budget = 0
        self.yearly_injection = 600000
        self.dca_amount = 50000
        self.days_since_last_dca = 20
        
    def run(self):
        records = []
        current_year = -1
        for current_date in self.dates:
            row = self.df.loc[current_date]
            price = row['Close']
            
            if current_date.year != current_year:
                self.cash += self.yearly_injection
                self.yearly_budget += self.yearly_injection
                current_year = current_date.year
                
            self.days_since_last_dca += 1
            if self.days_since_last_dca >= 20 and self.yearly_budget > 0:
                invest_amt = min(self.dca_amount, self.yearly_budget)
                if invest_amt > price:
                    shares = int(invest_amt / price)
                    cost = shares * price * (1 + 0.001425)
                    self.cash -= cost
                    self.yearly_budget -= invest_amt
                    self.shares += shares
                    self.days_since_last_dca = 0
            
            total_val = self.cash + (self.shares * price)
            records.append({'Date': current_date, 'Bench_Value': total_val})
            
        return pd.DataFrame(records).set_index('Date')

# --- Main Execution ---
if __name__ == "__main__":
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # Load Models
    print("Loading Models...")
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_split", "best_model.zip")
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_split", "best_model.zip")
    
    if not os.path.exists(best_buy_path) or not os.path.exists(best_sell_path):
        print("❌ 找不到模型檔案！")
        sys.exit(1)
        
    buy_model = PPO.load(best_buy_path, device=device)
    sell_model = PPO.load(best_sell_path, device=device)
    
    # User Input
    default_start = "2018-01-01"
    default_end = datetime.date.today().strftime("%Y-%m-%d")
    print(f"\n請輸入回測日期範圍 (Default: {default_start} ~ {default_end})")
    start_input = input(f"Start Date [{default_start}]: ").strip()
    end_input = input(f"End Date [{default_end}]: ").strip()
    start_date = start_input if start_input else default_start
    end_date = end_input if end_input else default_end
    
    # Fetch Data
    data_dict = fetch_data(DATA_PATH, start_date="2000-01-01")
    df_2330 = data_dict["2330.TW"]
    bench_df = data_dict.get("^TWII")
    
    print("計算特徵...")
    df_processed = calculate_features(df_2330, bench_df)
    
    # Run AI Backtest
    ai_backtester = DCABacktester(df_processed, buy_model, sell_model, start_date, end_date)
    ai_stats, trade_logs = ai_backtester.run()
    
    # Run Benchmark Backtest
    bench_backtester = BenchmarkDCA(df_processed, start_date, end_date)
    bench_stats = bench_backtester.run()
    
    # Results
    if not ai_stats.empty:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        final_ai = ai_stats.iloc[-1]['Total_Value']
        final_bench = bench_stats.iloc[-1]['Bench_Value']
        
        # Calculate Total Capital Injected
        years = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) + 1
        total_capital = years * 600000
        
        ai_roi = (final_ai - total_capital) / total_capital * 100
        bench_roi = (final_bench - total_capital) / total_capital * 100
        
        print(f"\n=== Result ({start_date} ~ {end_date}) ===")
        print(f"Total Capital Injected: {total_capital:,.0f}")
        print(f"AI Final Value:         {final_ai:,.0f} (ROI: {ai_roi:.2f}%)")
        print(f"Benchmark Final Value:  {final_bench:,.0f} (ROI: {bench_roi:.2f}%)")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(ai_stats.index, ai_stats['Total_Value'], label=f'AI DCA (ROI: {ai_roi:.2f}%)', color='red')
        plt.plot(bench_stats.index, bench_stats['Bench_Value'], label=f'Pure DCA (ROI: {bench_roi:.2f}%)', color='gray', linestyle='--')
        
        # Add Markers
        if trade_logs:
             log_df = pd.DataFrame(trade_logs)
             # Map dates to Total_Value for plotting on Equity Curve
             
             # 1. Aggressive Buy Markers (Red Triangle Up)
             agg_buys = log_df[log_df['Reason'] == 'AI_Aggressive_Low']
             if not agg_buys.empty:
                 buy_dates = pd.to_datetime(agg_buys['Date'])
                 # Plot each point
                 for date in buy_dates:
                     if date in ai_stats.index:
                         val = ai_stats.loc[date, 'Total_Value']
                         plt.scatter(date, val, marker='^', color='red', s=100, zorder=5)
                 # Add one dummy for legend
                 plt.scatter([], [], marker='^', color='red', s=100, label='AI Aggressive Buy')

             # 2. Sell Markers (Green Triangle Down)
             sells = log_df[log_df['Action'] == 'Sell']
             if not sells.empty:
                 sell_dates = pd.to_datetime(sells['Date'])
                 for date in sell_dates:
                     if date in ai_stats.index:
                         val = ai_stats.loc[date, 'Total_Value']
                         plt.scatter(date, val, marker='v', color='green', s=100, zorder=5)
                 plt.scatter([], [], marker='v', color='green', s=100, label='AI Sell')

        plt.title(f'2330.TW DCA Strategy Comparison ({start_date} ~ {end_date})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_PATH, f"dca_2330_result_{timestamp}.png"))
        print(f"Plot saved: dca_2330_result_{timestamp}.png")
        
        # Save Logs
        if trade_logs:
            log_df = pd.DataFrame(trade_logs) # Re-create or use existing
            log_df.to_csv(os.path.join(RESULTS_PATH, f"dca_2330_trades_{timestamp}.csv"), index=False)
            print("Trade logs saved.")
            print(log_df.tail(10).to_string(index=False))
