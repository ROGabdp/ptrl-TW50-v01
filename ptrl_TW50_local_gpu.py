# -*- coding: utf-8 -*-
"""
Pro Trader RL (Local GPU Version)
Adapted for WSL2/Ubuntu with Nvidia RTX 4070 support.
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
from collections import Counter
import glob

# --- 0. 環境與 GPU 設定 (Environment & GPU Setup) ---
def setup_environment():
    # 檢查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️ CUDA not available. Using CPU.")
        device = "cpu"

    # 設定專案路徑 (使用當前目錄)
    PROJECT_PATH = os.getcwd()
    MODELS_PATH = os.path.join(PROJECT_PATH, 'models')
    RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')
    DATA_PATH = os.path.join(PROJECT_PATH, 'data')

    for path in [MODELS_PATH, RESULTS_PATH, DATA_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"已建立資料夾: {path}")
        else:
            print(f"資料夾已存在: {path}")
            
    return PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device

# --- 1. 資料下載 (Data Acquisition) ---
def fetch_tw50_data(data_path, start_date="2010-01-01"):
    # 台灣市值前 50 大股票代號 (0050成分股) + 0050ETF本身(當基準) + 大盤指數
    tickers = [
        "0050.TW", "^TWII", # 基準與大盤
        "2330.TW", "2317.TW", "2454.TW", "2881.TW", "2382.TW", "2308.TW", "2882.TW", "2412.TW", "2891.TW", "3711.TW",
        "2886.TW", "2884.TW", "2303.TW", "2885.TW", "3231.TW", "3034.TW", "5880.TW", "2892.TW", "3008.TW", "2357.TW",
        "2002.TW", "2890.TW", "1101.TW", "2880.TW", "2883.TW", "2887.TW", "2345.TW", "3045.TW", "5871.TW", "3037.TW",
        "2912.TW", "1216.TW", "6505.TW", "4938.TW", "5876.TW", "1303.TW", "2395.TW", "2379.TW", "1301.TW", "3017.TW",
        "1326.TW", "2603.TW", "1590.TW", "3661.TW", "2327.TW", "4904.TW", "2801.TW", "1605.TW", "1504.TW", "2207.TW"
    ]

    print(f"開始下載 {len(tickers)} 檔標的資料...")
    # 使用 threads 加速下載
    data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, threads=True, progress=False)

    clean_data = {}
    
    # 處理大盤指數
    if "^TWII" in data.columns.levels[0]:
        market_index = data["^TWII"].copy()
        if isinstance(market_index.columns, pd.MultiIndex):
            market_index.columns = market_index.columns.get_level_values(0)
        clean_data["^TWII"] = market_index

    # 處理個股
    print("正在整理數據...")
    for t in tqdm(tickers):
        if t == "^TWII": continue
        try:
            df = data[t].copy()
            # 處理 yfinance 多層索引
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 欄位重新命名確保一致
            df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})

            # 移除空值與資料過少的股票
            df = df.dropna()
            if len(df) > 250: # 至少要有一年的資料
                clean_data[t] = df
        except Exception as e:
            print(f"Skipping {t}: {e}")

    print(f"資料處理完成，共 {len(clean_data)} 檔有效資料。")
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
    
    ha_high = df[['High', 'Open', 'Close']].max(axis=1) # Simplified
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
            
        if trend[i] == 1:
            final_upperband.iloc[i] = np.nan
        else:
            final_lowerband.iloc[i] = np.nan
            
    st = pd.Series(np.where(trend == 1, final_lowerband, final_upperband), index=df.index)
    return pd.DataFrame({'SUPERT_': st})

def calculate_features(df_in, benchmark_df):
    df = df_in.copy()

    # --- 1. 計算原始指標 (Raw Indicators) ---
    # Donchian Channel (20日 - 正規化基準)
    df['DC_Upper'] = df['High'].rolling(20).max().shift(1)
    df['DC_Lower'] = df['Low'].rolling(20).min().shift(1)

    # Donchian Channel (10日 - 買入訊號用)
    df['DC_Upper_10'] = df['High'].rolling(10).max().shift(1)

    # 填補空值
    df['DC_Upper'] = df['DC_Upper'].fillna(method='bfill')
    df['DC_Lower'] = df['DC_Lower'].fillna(method='bfill')
    df['DC_Upper_10'] = df['DC_Upper_10'].fillna(method='bfill')

    # 基礎指標
    # ATR
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range()
    
    # RSI
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # MFI
    try:
        df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
    except:
        df['MFI'] = 50.0

    # Heikin Ashi
    ha = calculate_heikin_ashi(df)
    df['HA_Open'] = ha['HA_open']
    df['HA_High'] = ha['HA_high']
    df['HA_Low'] = ha['HA_low']
    df['HA_Close'] = ha['HA_close']

    # SuperTrend
    st1 = calculate_supertrend(df, length=14, multiplier=2.0)
    st2 = calculate_supertrend(df, length=21, multiplier=1.0)
    df['SuperTrend_1'] = st1.iloc[:, 0]
    df['SuperTrend_2'] = st2.iloc[:, 0]

    # --- 2. 論文正規化邏輯 (Normalization) ---
    # [A] 價格類正規化：除以 DC_Upper
    base_price = df['DC_Upper'].replace(0, np.nan).fillna(method='bfill')

    price_cols = ['Close', 'Open', 'High', 'Low', 'DC_Lower',
                  'HA_Open', 'HA_High', 'HA_Low', 'HA_Close',
                  'SuperTrend_1', 'SuperTrend_2']

    for col in price_cols:
        df[f'Norm_{col}'] = df[col] / base_price

    # [B] 震盪指標正規化：除以 100
    df['Norm_RSI'] = df['RSI'] / 100.0
    df['Norm_MFI'] = df['MFI'] / 100.0

    # [C] 波動率正規化：變動率
    df['Norm_ATR_Change'] = (df['ATR'] / df['ATR'].shift(1)).fillna(1.0)

    # [D] 相對強弱 RS 正規化
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

    # --- 3. Label 與訊號 ---
    df['Signal_Buy_Filter'] = (df['High'] > df['DC_Upper_10'])
    df['Next_20d_Max'] = df['High'].shift(-20).rolling(20).max() / df['Close'] - 1

    return df.dropna()

# --- 3. RL 環境定義 (RL Environments) ---
class BuyEnv(gym.Env):
    def __init__(self, data_dict, is_training=True):
        super().__init__()
        self.samples = []
        
        # 檢查欄位
        sample_df = next(iter(data_dict.values()))
        missing = [c for c in FEATURE_COLS if c not in sample_df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")

        for t, df in data_dict.items():
            signals = df[df['Signal_Buy_Filter'] == True]
            if len(signals) > 0:
                states = signals[FEATURE_COLS].values.astype(np.float32)
                future_rets = signals['Next_20d_Max'].values.astype(np.float32)
                for i in range(len(signals)):
                    self.samples.append((states[i], future_rets[i]))

        self.action_space = spaces.Discrete(2) # 0: Wait, 1: Buy
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(FEATURE_COLS),), dtype=np.float32)
        self.is_training = is_training
        self.idx = 0

    def reset(self, seed=None, options=None):
        if self.is_training:
            self.idx = np.random.randint(0, len(self.samples))
        else:
            self.idx = (self.idx + 1) % len(self.samples)
        return self.samples[self.idx][0], {}

    def step(self, action):
        _, max_ret = self.samples[self.idx]
        reward = 0
        if action == 1: # Buy
            reward = max_ret * 10.0
            if max_ret < -0.05: reward -= 1.0
        else: # Wait
            if max_ret > 0.02: reward = -(max_ret * 10.0)
            elif max_ret < 0: reward = 0.1
            else: reward = 0.0
        return self.samples[self.idx][0], reward, True, False, {}

class SellEnv(gym.Env):
    def __init__(self, data_dict):
        super().__init__()
        self.episodes = []
        for t, df in data_dict.items():
            buy_indices = np.where(df['Signal_Buy_Filter'])[0]
            feature_data = df[FEATURE_COLS].values.astype(np.float32)
            close_prices = df['Close'].values.astype(np.float32)
            for idx in buy_indices:
                if idx + 120 < len(df):
                    episode_features = feature_data[idx : idx+120]
                    episode_prices = close_prices[idx : idx+120]
                    max_price = np.max(episode_prices)
                    self.episodes.append({
                        'features': episode_features,
                        'prices': episode_prices,
                        'max_price': max_price
                    })
        self.action_space = spaces.Discrete(2) # 0: Hold, 1: Sell
        self.obs_dim = len(FEATURE_COLS) + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_episode = self.episodes[np.random.randint(0, len(self.episodes))]
        self.day = 0
        self.buy_price = self.current_episode['prices'][0]
        self.max_potential_price = self.current_episode['max_price']
        return self._get_observation(), {}

    def _get_observation(self):
        market_features = self.current_episode['features'][self.day]
        current_price = self.current_episode['prices'][self.day]
        current_return = current_price / self.buy_price
        time_ratio = self.day / 120.0
        obs = np.concatenate([market_features, [current_return, time_ratio]])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.current_episode['prices'][self.day]
        current_return = current_price / self.buy_price
        reward = 0
        done = False

        if current_return < 0.90: # Stop Loss
            reward = -5.0
            done = True
        elif action == 1: # Sell
            potential_profit = self.max_potential_price - self.buy_price
            actual_profit = current_price - self.buy_price
            if potential_profit > 0:
                efficiency = actual_profit / potential_profit
                reward = efficiency * 2.0
                if efficiency > 0.9: reward += 1.0
            else:
                reward = (current_return - 1.0) * 5.0
                if current_return > 0.98: reward += 1.0
            done = True
        elif self.day >= 119: # Time's up
            potential_profit = self.max_potential_price - self.buy_price
            actual_profit = current_price - self.buy_price
            if potential_profit > 0: reward = (actual_profit / potential_profit) * 2.0
            else: reward = (current_return - 1.0) * 5.0
            done = True
        else: # Hold
            if current_return > 1.02: reward = 0.02
            self.day += 1

        next_obs = self._get_observation() if not done else self._get_observation()
        return next_obs, reward, done, False, {}

# --- 4. 回測類別 (Backtester) ---
class DetailedBacktester:
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
        self.dates = [d for d in self.dates if d >= pd.Timestamp('2021-01-01')]

    def run(self):
        print(f"正在執行回測 ({self.dates[0].date()} ~ {self.dates[-1].date()})...")
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
                
                market_features = row[FEATURE_COLS].values.astype(np.float32)
                ret = price / info['cost_price']
                time_ratio = info['days_held'] / 120.0
                sell_state = np.concatenate([market_features, [ret, time_ratio]])
                
                action, _ = self.sell_model.predict(sell_state, deterministic=True)
                
                if action == 1 or ret < 0.90 or (info['days_held'] > 20 and ret < 1.05):
                    revenue = info['shares'] * price * (1 - 0.004)
                    self.cash += revenue
                    del self.inventory[ticker]

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
                'Date': current_date,
                'Total_Value': self.cash + daily_equity,
                'Cash': self.cash,
                'Invested_Amount': daily_equity,
                'Holdings_Count': len(held_stocks),
                'Holdings_List': held_stocks
            })
            
        return pd.DataFrame(records).set_index('Date')

# --- Main Execution ---
if __name__ == "__main__":
    # 1. 設定環境
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # 2. 下載與處理資料
    raw_data_dict, market_index_df = fetch_tw50_data(DATA_PATH)
    
    print("正在計算特徵...")
    processed_data = {}
    bench_df = raw_data_dict.get("^TWII")
    for ticker, df in raw_data_dict.items():
        if ticker != "^TWII":
            try:
                processed_data[ticker] = calculate_features(df, bench_df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
    # 3. 準備訓練環境
    train_data = {k: v for k, v in processed_data.items() if k != "0050.TW"}
    buy_env = BuyEnv(train_data, is_training=True)
    sell_env = SellEnv(train_data)
    eval_env = BuyEnv(train_data, is_training=True)
    
    # 4. 設定 PPO 參數 (包含 device)
    ppo_params = {
        "learning_rate": 0.0001,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.05,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "verbose": 1,
        "device": device # 使用 GPU
    }
    
    # 5. 訓練 Buy Agent
    print("\n=== 開始訓練 Buy Agent ===")
    TOTAL_TIMESTEPS = 2000000
    
    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=MODELS_PATH, name_prefix="ppo_buy_ckpt")
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(MODELS_PATH, "best_buy_model"),
                                 log_path=RESULTS_PATH, eval_freq=20000, deterministic=True, render=False)
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # 檢查是否有存檔可續練
    latest_ckpt = None
    files = glob.glob(os.path.join(MODELS_PATH, "ppo_buy_ckpt_*_steps.zip"))
    if files:
        latest_ckpt = max(files, key=lambda x: int(x.split('_')[-2]))
        
    if latest_ckpt:
        print(f"發現中斷的存檔：{latest_ckpt}，繼續訓練...")
        buy_model = PPO.load(latest_ckpt, env=buy_env, **ppo_params)
        current_steps = int(latest_ckpt.split('_')[-2])
        remaining_steps = max(0, TOTAL_TIMESTEPS - current_steps)
        if remaining_steps > 0:
            buy_model.learn(total_timesteps=remaining_steps, callback=callbacks, reset_num_timesteps=False)
    else:
        print("開始重新訓練 Buy Agent...")
        buy_model = PPO("MlpPolicy", buy_env, **ppo_params)
        buy_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
        
    buy_model.save(os.path.join(MODELS_PATH, "ppo_buy_tw50"))
    
    # 6. 訓練 Sell Agent
    print("\n=== 開始訓練 Sell Agent ===")
    sell_model = PPO("MlpPolicy", sell_env, **ppo_params)
    sell_model.learn(total_timesteps=TOTAL_TIMESTEPS // 2)
    sell_model.save(os.path.join(MODELS_PATH, "ppo_sell_tw50"))
    
    # 7. 執行回測
    print("\n=== 執行回測 ===")
    # 載入最佳模型
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_model", "best_model.zip")
    if os.path.exists(best_buy_path):
        buy_model = PPO.load(best_buy_path, device=device)
    else:
        buy_model = PPO.load(os.path.join(MODELS_PATH, "ppo_buy_tw50"), device=device)
    
    sell_model = PPO.load(os.path.join(MODELS_PATH, "ppo_sell_tw50"), device=device)
    
    stock_data_only = {k: v for k, v in processed_data.items() if k != "^TWII" and k != "0050.TW"}
    analyzer = DetailedBacktester(stock_data_only, buy_model, sell_model)
    daily_stats = analyzer.run()
    
    # 8. 輸出結果
    print("\n回測完成！")
    print(f"期初資金: {daily_stats.iloc[0]['Total_Value']:.0f}")
    print(f"期末資金: {daily_stats.iloc[-1]['Total_Value']:.0f}")
    
    # 繪圖
    daily_stats['Total_Value'].plot(title='AI Portfolio Value', figsize=(10, 6))
    plt.savefig(os.path.join(RESULTS_PATH, "portfolio_performance.png"))
    print(f"圖表已儲存至: {os.path.join(RESULTS_PATH, 'portfolio_performance.png')}")
