# -*- coding: utf-8 -*-
"""
Pro Trader RL - Flexible Backtest (Split Model Version)
Features:
- Uses models trained by `ptrl_TW50_split_train.py`
- Flexible Date Range Selection
- Saves detailed trade logs to CSV
- Prints summary table of recent trades
- Benchmark Comparison (^TWII)
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
    df['Signal_Buy_Filter'] = (df['High'] > df['DC_Upper_10'])
    return df.dropna()

# --- 3. Backtester Class ---
class DetailedBacktesterFlexible:
    def __init__(self, data_dict, buy_model, sell_model, start_date, end_date, initial_capital=1000000):
        self.data_dict = data_dict
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.cash = initial_capital
        self.inventory = {}
        self.max_positions = 10
        self.position_size_pct = 0.10
        
        # Filter Dates
        sample_ticker = list(data_dict.keys())[0]
        df = data_dict[sample_ticker]
        mask = (df.index >= start_date) & (df.index <= end_date)
        self.dates = sorted(df[mask].index)
        self.trade_logs = []

    def run(self):
        print(f"執行回測 ({self.dates[0].date()} ~ {self.dates[-1].date()})...")
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
                
                should_sell = False
                reason = ""
                
                # Stop Loss
                if current_return < 0.90:
                    should_sell = True; reason = "StopLoss_Dip"
                elif info['days_held'] >= 20 and current_return < 1.10:
                    should_sell = True; reason = "StopLoss_Sideways"
                else:
                    # AI Signal
                    market_features = row[FEATURE_COLS].values.astype(np.float32)
                    sell_state = np.concatenate([market_features, [current_return]])
                    obs_tensor = torch.as_tensor(sell_state).unsqueeze(0).to(self.sell_model.device)
                    with torch.no_grad():
                        distribution = self.sell_model.policy.get_distribution(obs_tensor)
                        probs = distribution.distribution.probs.cpu().numpy()[0]
                    if (probs[1] - probs[0]) > 0.85:
                        should_sell = True; reason = "AI_Signal"
                
                if should_sell:
                    self._sell(ticker, price, reason, current_date)

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
                                self.trade_logs.append({
                                    'Date': current_date, 'Ticker': ticker, 'Action': 'Buy',
                                    'Price': row['Close'], 'Reason': 'AI_Signal',
                                    'Shares': shares, 'Balance': self.cash
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
            
        return pd.DataFrame(records).set_index('Date'), self.trade_logs

    def _sell(self, ticker, price, reason, date):
        info = self.inventory[ticker]
        revenue = info['shares'] * price * (1 - 0.004)
        self.cash += revenue
        del self.inventory[ticker]
        self.trade_logs.append({
            'Date': date, 'Ticker': ticker, 'Action': 'Sell',
            'Price': price, 'Reason': reason,
            'Shares': info['shares'], 'Balance': self.cash
        })

# --- Main Execution ---
if __name__ == "__main__":
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # Load Models (Split Version)
    print("Loading Models...")
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_split", "best_model.zip")
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_split", "best_model.zip")
    final_buy_path = os.path.join(MODELS_PATH, "ppo_buy_split_final.zip")
    final_sell_path = os.path.join(MODELS_PATH, "ppo_sell_split_final.zip")
    
    buy_model_path = best_buy_path if os.path.exists(best_buy_path) else final_buy_path
    sell_model_path = best_sell_path if os.path.exists(best_sell_path) else final_sell_path
    
    if not os.path.exists(buy_model_path) or not os.path.exists(sell_model_path):
        print("❌ 找不到模型檔案！請先執行 ptrl_TW50_split_train.py 進行訓練。")
        sys.exit(1)
        
    buy_model = PPO.load(buy_model_path, device=device)
    sell_model = PPO.load(sell_model_path, device=device)
    print(f"✅ Loaded Buy Model: {os.path.basename(buy_model_path)}")
    print(f"✅ Loaded Sell Model: {os.path.basename(sell_model_path)}")
    
    # User Input
    default_start = "2023-01-01"
    default_end = datetime.date.today().strftime("%Y-%m-%d")
    
    print(f"\n請輸入回測日期範圍 (Default: {default_start} ~ {default_end})")
    start_input = input(f"Start Date (YYYY-MM-DD) [{default_start}]: ").strip()
    end_input = input(f"End Date (YYYY-MM-DD) [{default_end}]: ").strip()
    
    start_date = start_input if start_input else default_start
    end_date = end_input if end_input else default_end
    
    # Fetch Data
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
    
    # Run Backtest
    backtester = DetailedBacktesterFlexible(processed_data, buy_model, sell_model, start_date, end_date)
    daily_stats, trade_logs = backtester.run()
    
    if not daily_stats.empty:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_val = daily_stats.iloc[0]['Total_Value']
        final_val = daily_stats.iloc[-1]['Total_Value']
        roi = (final_val - initial_val) / initial_val * 100
        print(f"\n=== Result ({start_date} ~ {end_date}) ===")
        print(f"Initial Capital: {initial_val:,.0f}")
        print(f"Final Value:     {final_val:,.0f}")
        print(f"ROI:             {roi:.2f}%")
        print(f"Total Trades:    {len(trade_logs)}")
        
        # Benchmark ROI
        bench_roi = 0.0
        bench_data = None
        if bench_df is not None:
             mask = (bench_df.index >= start_date) & (bench_df.index <= end_date)
             bench_data = bench_df.loc[mask].copy().dropna(subset=['Close'])
             if not bench_data.empty:
                 b_start = bench_data.iloc[0]['Close']
                 b_end = bench_data.iloc[-1]['Close']
                 bench_roi = (b_end - b_start) / b_start * 100
                 print(f"Benchmark (^TWII) ROI: {bench_roi:.2f}%")
                 bench_data['Normalized_Value'] = (bench_data['Close'] / b_start) * initial_val

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(daily_stats.index, daily_stats['Total_Value'], label=f'AI Portfolio (ROI: {roi:.2f}%)', color='red')
        if bench_data is not None:
            plt.plot(bench_data.index, bench_data['Normalized_Value'], label=f'^TWII (ROI: {bench_roi:.2f}%)', color='gray', linestyle='--')
        plt.title(f'Flexible Backtest Equity Curve ({start_date} ~ {end_date})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_PATH, f"flexible_backtest_result_{timestamp}.png"))
        print("Plot saved.")
        
        # Save Trade Logs
        if trade_logs:
            log_df = pd.DataFrame(trade_logs)
            log_path = os.path.join(RESULTS_PATH, f"flexible_backtest_trades_{timestamp}.csv")
            log_df.to_csv(log_path, index=False)
            print(f"Trade logs saved to: {log_path}")
            
            print("\n=== Recent Trades Summary ===")
            print(log_df.tail(10).to_string(index=False))
        else:
            print("No trades executed.")
