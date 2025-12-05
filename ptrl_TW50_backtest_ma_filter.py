# -*- coding: utf-8 -*-
"""
Pro Trader RL (Backtest Only - MA Filter)
Features:
- Custom Date Range Backtest
- MA-based Signal Filtering:
  - Buy Filter: Close < MA60
  - Sell Filter: Close > MA10 * 1.10
- Loads existing models: ppo_buy_ma, ppo_sell_ma
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
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import glob
import multiprocessing

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

# --- 1. 資料下載 (Start Date: 2000-01-01) ---
def fetch_tw50_data(data_path, start_date="2000-01-01"):
    tickers = [
        "0050.TW", "^TWII",
        "2330.TW", "2317.TW", "2454.TW", "2881.TW", "2382.TW", "2308.TW", "2882.TW", "2412.TW", "2891.TW", "3711.TW",
        "2886.TW", "2884.TW", "2303.TW", "2885.TW", "3231.TW", "3034.TW", "5880.TW", "2892.TW", "3008.TW", "2357.TW",
        "2002.TW", "2890.TW", "1101.TW", "2880.TW", "2883.TW", "2887.TW", "2345.TW", "3045.TW", "5871.TW", "3037.TW",
        "2912.TW", "1216.TW", "6505.TW", "4938.TW", "5876.TW", "1303.TW", "2395.TW", "2379.TW", "1301.TW", "3017.TW",
        "1326.TW", "2603.TW", "1590.TW", "3661.TW", "2327.TW", "4904.TW", "2801.TW", "1605.TW", "1504.TW", "2207.TW"
    ]
    print(f"開始下載 {len(tickers)} 檔標的資料 (Start: {start_date})...")
    data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, threads=True, progress=False)
    clean_data = {}
    if "^TWII" in data.columns.levels[0]:
        market_index = data["^TWII"].copy()
        if isinstance(market_index.columns, pd.MultiIndex):
            market_index.columns = market_index.columns.get_level_values(0)
        clean_data["^TWII"] = market_index

    for t in tqdm(tickers):
        if t == "^TWII": continue
        try:
            df = data[t].copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
            df = df.dropna()
            if len(df) > 250:
                clean_data[t] = df
        except Exception as e:
            print(f"Skipping {t}: {e}")
    return clean_data, clean_data.get("^TWII")

# --- 2. 特徵工程 (Feature Engineering) ---
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

# --- Helper Functions for Indicators ---
def calculate_heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    ha_open = [df['Open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha_close.iloc[i-1]) / 2)
    ha_open = pd.Series(ha_open, index=df.index)
    
    ha_high = pd.concat([df['High'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['Low'], ha_open, ha_close], axis=1).min(axis=1)
    
    return pd.DataFrame({
        'HA_open': ha_open,
        'HA_high': ha_high,
        'HA_low': ha_low,
        'HA_close': ha_close
    })

def calculate_supertrend(df, length=14, multiplier=3.0):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    atr = AverageTrueRange(high, low, close, window=length).average_true_range()
    atr = atr.fillna(method='bfill')
    
    hl2 = (high + low) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()
    
    trend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if basic_upperband.iloc[i] < final_upperband.iloc[i-1] or close.iloc[i-1] > final_upperband.iloc[i-1]:
            final_upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i-1]
            
        if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1] or close.iloc[i-1] < final_lowerband.iloc[i-1]:
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            
        if close.iloc[i] > final_upperband.iloc[i-1]:
            trend[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
    st = pd.Series(np.where(trend == 1, final_lowerband, final_upperband), index=df.index)
    return pd.DataFrame({'SUPERT_': st})

def calculate_features(df_in, benchmark_df):
    df = df_in.copy()
    # Donchian Channel (Still used for features, but not for filtering)
    df['DC_Upper'] = df['High'].rolling(20).max().shift(1)
    df['DC_Lower'] = df['Low'].rolling(20).min().shift(1)
    
    # Fill NA
    df['DC_Upper'] = df['DC_Upper'].fillna(method='bfill')
    df['DC_Lower'] = df['DC_Lower'].fillna(method='bfill')

    # Indicators
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    try:
        df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
    except:
        df['MFI'] = 50.0
    
    ha = calculate_heikin_ashi(df)
    df['HA_Open'] = ha['HA_open']
    df['HA_High'] = ha['HA_high']
    df['HA_Low'] = ha['HA_low']
    df['HA_Close'] = ha['HA_close']

    st1 = calculate_supertrend(df, length=14, multiplier=2.0)
    st2 = calculate_supertrend(df, length=21, multiplier=1.0)
    df['SuperTrend_1'] = st1.iloc[:, 0]
    df['SuperTrend_2'] = st2.iloc[:, 0]

    # Normalization
    base_price = df['DC_Upper'].replace(0, np.nan).fillna(method='bfill')
    price_cols = ['Close', 'Open', 'High', 'Low', 'DC_Lower',
                  'HA_Open', 'HA_High', 'HA_Low', 'HA_Close',
                  'SuperTrend_1', 'SuperTrend_2']
    for col in price_cols:
        df[f'Norm_{col}'] = df[col] / base_price

    df['Norm_RSI'] = df['RSI'] / 100.0
    df['Norm_MFI'] = df['MFI'] / 100.0
    df['Norm_ATR_Change'] = (df['ATR'] / df['ATR'].shift(1)).fillna(1.0)

    if benchmark_df is not None:
        bench_close = benchmark_df['Close'].reindex(df.index).fillna(method='ffill')
        df['RS_Raw'] = df['Close'] / bench_close
        rs_min = df['RS_Raw'].rolling(250).min()
        rs_max = df['RS_Raw'].rolling(250).max()
        denominator = (rs_max - rs_min).replace(0, np.nan).fillna(1.0) + 1e-9
        df['Norm_RS_Ratio'] = (df['RS_Raw'] - rs_min) / denominator
        df['Norm_RS_Ratio'] = df['Norm_RS_Ratio'].fillna(0.5)
        for period in [5, 10, 20, 60, 120]:
            df[f'RS_ROC_{period}'] = df['RS_Raw'].pct_change(period).fillna(0)
    else:
        df['Norm_RS_Ratio'] = 0.0
        for period in [5, 10, 20, 60, 120]:
            df[f'RS_ROC_{period}'] = 0.0

    # --- New Filters ---
    df['MA60'] = df['Close'].rolling(60).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    
    # Buy Filter: Close < MA60
    df['Signal_Buy_Filter'] = (df['Close'] < df['MA60'])
    
    # Sell Filter: Close > MA10 * 1.10
    df['Signal_Sell_Filter'] = (df['Close'] > df['MA10'] * 1.10)
    
    df['Next_20d_Max'] = df['High'].shift(-20).rolling(20).max() / df['Close'] - 1
    
    feature_cols_to_check = [c for c in df.columns if c != 'Next_20d_Max']
    return df.dropna(subset=feature_cols_to_check)

# --- 4. 回測類別 ---
class DetailedBacktesterPaper:
    def __init__(self, data_dict, buy_model, sell_model, initial_capital=1000000):
        self.data_dict = data_dict
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.cash = initial_capital
        self.inventory = {}
        self.max_positions = 10
        self.position_size_pct = 0.10
        sample_ticker = list(data_dict.keys())[0]
        self.dates = sorted(data_dict[sample_ticker].index)
        self.trade_logs = []

    def run(self):
        print(f"執行回測 (MA Filter Logic)...")
        records = []
        for current_date in tqdm(self.dates):
            # 1. 賣出檢查
            for ticker in list(self.inventory.keys()):
                info = self.inventory[ticker]
                df = self.data_dict[ticker]
                if current_date not in df.index: continue
                row = df.loc[current_date]
                price = row['Close']
                info['days_held'] += 1
                current_return = price / info['cost_price']
                
                # Stop Loss
                if current_return < 0.90:
                    self._sell(ticker, price, "StopLoss_Dip", current_date)
                    continue
                if info['days_held'] >= 20 and current_return < 1.10:
                    self._sell(ticker, price, "StopLoss_Sideways", current_date)
                    continue

                # AI Signal (Only if Sell Filter is True)
                if row['Signal_Sell_Filter']:
                    market_features = row[FEATURE_COLS].values.astype(np.float32)
                    sell_state = np.concatenate([market_features, [current_return]])
                    obs_tensor = torch.as_tensor(sell_state).unsqueeze(0).to(self.sell_model.device)
                    with torch.no_grad():
                        distribution = self.sell_model.policy.get_distribution(obs_tensor)
                        probs = distribution.distribution.probs.cpu().numpy()[0]
                    
                    if (probs[1] - probs[0]) > 0.85:
                        self._sell(ticker, price, "AI_Signal", current_date)

            # 2. 買入檢查
            current_equity = 0
            for t, info in self.inventory.items():
                if current_date in self.data_dict[t].index:
                    current_equity += info['shares'] * self.data_dict[t].loc[current_date]['Close']
                else:
                    current_equity += info['shares'] * info['cost_price']
            total_asset = self.cash + current_equity

            if len(self.inventory) < self.max_positions and self.cash > 10000:
                tickers = list(self.data_dict.keys())
                np.random.shuffle(tickers)
                for ticker in tickers:
                    if ticker in self.inventory: continue
                    df = self.data_dict[ticker]
                    if current_date not in df.index: continue
                    row = df.loc[current_date]
                    if row['Signal_Buy_Filter']:
                        state = row[FEATURE_COLS].values.astype(np.float32)
                        action, _ = self.buy_model.predict(state, deterministic=True)
                        if action == 1:
                            target_amt = total_asset * self.position_size_pct
                            invest_amt = min(target_amt, self.cash)
                            if invest_amt > row['Close'] * 1000:
                                shares = int(invest_amt / row['Close'])
                                cost = shares * row['Close'] * (1 + 0.001425)
                                self.cash -= cost
                                self.inventory[ticker] = {'shares': shares, 'cost_price': row['Close'], 'days_held': 0}
                                
                                # Log Buy
                                self.trade_logs.append({
                                    'Date': current_date, 'Ticker': ticker, 'Action': 'Buy',
                                    'Price': row['Close'], 'Reason': 'AI_Signal',
                                    'Shares': shares, 'Cash': self.cash, 'Total_Asset': total_asset
                                })
                                if len(self.inventory) >= self.max_positions: break
            
            # 紀錄
            daily_equity = 0
            held_stocks = []
            for t, info in self.inventory.items():
                if current_date in self.data_dict[t].index:
                    val = info['shares'] * self.data_dict[t].loc[current_date]['Close']
                else:
                    val = info['shares'] * info['cost_price']
                daily_equity += val
                held_stocks.append(t)
            
            records.append({
                'Date': current_date, 'Total_Value': self.cash + daily_equity,
                'Cash': self.cash, 'Invested_Amount': daily_equity, 'Holdings_Count': len(held_stocks)
            })
            
        if not records:
            return pd.DataFrame(columns=['Total_Value', 'Cash', 'Invested_Amount', 'Holdings_Count']), self.trade_logs
        return pd.DataFrame(records).set_index('Date'), self.trade_logs

    def _sell(self, ticker, price, reason, date):
        info = self.inventory[ticker]
        revenue = info['shares'] * price * (1 - 0.004)
        self.cash += revenue
        shares_sold = info['shares']
        del self.inventory[ticker]
        
        self.trade_logs.append({
            'Date': date, 'Ticker': ticker, 'Action': 'Sell',
            'Price': price, 'Reason': reason,
            'Shares': shares_sold, 'Cash': self.cash, 'Total_Asset': np.nan 
        })

    def export_results(self, daily_stats, results_path, bench_data=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Export Trade Log
        if self.trade_logs:
            df_log = pd.DataFrame(self.trade_logs)
            log_path = os.path.join(results_path, f"trade_log_{timestamp}.csv")
            df_log.to_csv(log_path, index=False)
            print(f"Trade log saved to: {log_path}")
        
        # 2. Plot Equity Curve (AI vs Benchmark)
        plt.figure(figsize=(12, 6))
        
        # AI Performance
        ai_roi = (daily_stats['Total_Value'].iloc[-1] / daily_stats['Total_Value'].iloc[0] - 1) * 100
        plt.plot(daily_stats.index, daily_stats['Total_Value'], label=f'AI (ROI: {ai_roi:.2f}%)', color='red')
        
        # Benchmark Performance
        if bench_data is not None:
            # Align benchmark to backtest dates
            bench_plot = bench_data.reindex(daily_stats.index)
            bench_plot = bench_plot.fillna(method='ffill').fillna(method='bfill')
            
            # Normalize benchmark to start at same initial value as AI
            if not bench_plot['Close'].isnull().all():
                initial_val = daily_stats['Total_Value'].iloc[0]
                first_valid_price = bench_plot['Close'].iloc[0]
                if pd.isna(first_valid_price):
                     # Fallback if first index is still NaN (should be covered by bfill, but safety check)
                     first_valid_price = bench_plot['Close'].dropna().iloc[0]
                
                bench_norm = (bench_plot['Close'] / first_valid_price) * initial_val
                
                bench_roi = (bench_plot['Close'].iloc[-1] / first_valid_price - 1) * 100
                plt.plot(daily_stats.index, bench_norm, label=f'^TWII (ROI: {bench_roi:.2f}%)', color='gray', linestyle='--')
            
        plt.title(f"Out-of-Sample Backtest (MA Filter)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (TWD)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_path, f"ma_filter_backtest_result_{timestamp}.png"))
        plt.close()
        
        # 3. Plot Individual Stocks (Top 5 by trade count)
        if self.trade_logs:
            df_log = pd.DataFrame(self.trade_logs)
            top_tickers = df_log['Ticker'].value_counts().head(5).index.tolist()
            
            for ticker in top_tickers:
                if ticker not in self.data_dict: continue
                
                df = self.data_dict[ticker]
                # Filter data to backtest range
                mask = (df.index >= daily_stats.index[0]) & (df.index <= daily_stats.index[-1])
                df_plot = df.loc[mask]
                
                if df_plot.empty: continue
                
                plt.figure(figsize=(14, 7))
                plt.plot(df_plot.index, df_plot['Close'], label='Close Price', color='black', alpha=0.6)
                
                # Plot MA60
                if 'MA60' in df_plot.columns:
                    plt.plot(df_plot.index, df_plot['MA60'], label='MA60', color='blue', linestyle='--', alpha=0.7)
                
                # Plot Buy Points
                buys = df_log[(df_log['Ticker'] == ticker) & (df_log['Action'] == 'Buy')]
                plt.scatter(buys['Date'], buys['Price'], marker='^', color='red', s=100, label='Buy', zorder=5)
                
                # Plot Sell Points
                sells = df_log[(df_log['Ticker'] == ticker) & (df_log['Action'] == 'Sell')]
                plt.scatter(sells['Date'], sells['Price'], marker='v', color='green', s=100, label='Sell', zorder=5)
                
                plt.title(f"Trade History: {ticker}")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)
                
                safe_ticker = ticker.replace('.', '_')
                plt.savefig(os.path.join(results_path, f"plot_{safe_ticker}_{timestamp}.png"))
                plt.close()
                print(f"Saved plot for {ticker}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. 設定環境
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # 2. 下載與處理資料
    raw_data_dict, market_index_df = fetch_tw50_data(DATA_PATH, start_date="2000-01-01")
    
    print("正在計算特徵...")
    processed_data = {}
    bench_df = raw_data_dict.get("^TWII")
    for ticker, df in raw_data_dict.items():
        if ticker != "^TWII":
            try:
                processed_data[ticker] = calculate_features(df, bench_df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
    # 3. User Input for Date Range
    print("\n=== Custom Backtest Configuration ===")
    start_date_str = input("Enter Start Date (YYYY-MM-DD): ").strip()
    end_date_str = input("Enter End Date (YYYY-MM-DD): ").strip()
    
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
    except ValueError:
        print("Invalid date format. Using default range: 2012-01-01 to 2017-12-31")
        start_date = pd.to_datetime("2012-01-01")
        end_date = pd.to_datetime("2017-12-31")

    print(f"Backtesting from {start_date.date()} to {end_date.date()}")

    # 4. Filter Data for Backtest
    backtest_data = {}
    for ticker, df in processed_data.items():
        if ticker == "0050.TW": continue
        
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_backtest = df[mask].copy()
        
        if not df_backtest.empty:
            backtest_data[ticker] = df_backtest
            
    print(f"Backtesting Tickers: {len(backtest_data)}")

    # 5. Load Models
    print("\n=== Loading Models ===")
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_ma", "best_model.zip")
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_ma", "best_model.zip")
    final_buy_path = os.path.join(MODELS_PATH, "ppo_buy_ma_final.zip")
    final_sell_path = os.path.join(MODELS_PATH, "ppo_sell_ma_final.zip")
    
    buy_model = None
    sell_model = None
    
    if os.path.exists(best_buy_path):
        buy_model = PPO.load(best_buy_path, device=device)
        print(f"Loaded Best Buy Model: {best_buy_path}")
    elif os.path.exists(final_buy_path):
        buy_model = PPO.load(final_buy_path, device=device)
        print(f"Loaded Final Buy Model: {final_buy_path}")
    else:
        print("❌ No Buy Model found! Please train first.")
        sys.exit(1)
        
    if os.path.exists(best_sell_path):
        sell_model = PPO.load(best_sell_path, device=device)
        print(f"Loaded Best Sell Model: {best_sell_path}")
    elif os.path.exists(final_sell_path):
        sell_model = PPO.load(final_sell_path, device=device)
        print(f"Loaded Final Sell Model: {final_sell_path}")
    else:
        print("❌ No Sell Model found! Please train first.")
        sys.exit(1)

    # 6. Run Backtest
    print("\n=== Running Backtest ===")
    backtester = DetailedBacktesterPaper(backtest_data, buy_model, sell_model)
    daily_stats, trade_logs = backtester.run()
    
    if not daily_stats.empty:
        # Prepare Benchmark Data
        bench_data = None
        if "^TWII" in raw_data_dict:
            bench_df = raw_data_dict["^TWII"]
            mask = (bench_df.index >= start_date) & (bench_df.index <= end_date)
            bench_data = bench_df.loc[mask].copy()
            bench_data = bench_data.dropna(subset=['Close'])

        # Export Results (CSV, Plots)
        backtester.export_results(daily_stats, RESULTS_PATH, bench_data=bench_data)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_val = daily_stats.iloc[0]['Total_Value']
        final_val = daily_stats.iloc[-1]['Total_Value']
        roi = (final_val - initial_val) / initial_val * 100
        print(f"Backtest ROI: {roi:.2f}%")
        
        # Calculate Benchmark ROI for printing
        if bench_data is not None and not bench_data.empty:
            bench_start = bench_data.iloc[0]['Close']
            bench_end = bench_data.iloc[-1]['Close']
            bench_roi = (bench_end - bench_start) / bench_start * 100
            print(f"Benchmark ROI: {bench_roi:.2f}%")
    else:
        print("No trades or data for the selected period.")
