# -*- coding: utf-8 -*-
"""
Pro Trader RL (Flexible Backtest Version)
Allows backtesting on custom date ranges using the best trained models.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
import torch

# Import from main script
# Ensure the current directory is in sys.path
sys.path.append(os.getcwd())
try:
    from ptrl_TW50_paper_version import (
        setup_environment,
        fetch_tw50_data,
        calculate_features,
        DetailedBacktesterPaper,
        FEATURE_COLS
    )
except ImportError:
    print("Error: Could not import from ptrl_TW50_paper_version.py")
    print("Please make sure this script is in the same directory as ptrl_TW50_paper_version.py")
    sys.exit(1)

class FlexibleBacktester(DetailedBacktesterPaper):
    def __init__(self, data_dict, buy_model, sell_model, start_date, end_date, initial_capital=1000000):
        super().__init__(data_dict, buy_model, sell_model, initial_capital)
        
        # Override dates with custom range
        sample_ticker = list(data_dict.keys())[0]
        all_dates = sorted(data_dict[sample_ticker].index)
        
        self.dates = [d for d in all_dates if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]
        
        if not self.dates:
            print(f"⚠️ Warning: No dates found for backtesting between {start_date} and {end_date}!")
        else:
            print(f"Backtesting Range: {self.dates[0].date()} to {self.dates[-1].date()} ({len(self.dates)} trading days)")

if __name__ == "__main__":
    # 1. Setup
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # 2. Data
    print("Loading data...")
    raw_data_dict, market_index_df = fetch_tw50_data(DATA_PATH)
    
    print("Calculating features...")
    processed_data = {}
    bench_df = raw_data_dict.get("^TWII")
    for ticker, df in raw_data_dict.items():
        if ticker != "^TWII":
            try:
                processed_data[ticker] = calculate_features(df, bench_df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                
    stock_data_only = {k: v for k, v in processed_data.items() if k != "^TWII" and k != "0050.TW"}

    # 3. Load Best Models
    print("\n=== Loading Best Models ===")
    
    # Buy Model
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_paper", "best_model.zip")
    final_buy_path = os.path.join(MODELS_PATH, "ppo_buy_paper_final.zip")
    
    if os.path.exists(best_buy_path):
        buy_model = PPO.load(best_buy_path, device=device)
        print(f"✅ Loaded Best Buy Model: {best_buy_path}")
    elif os.path.exists(final_buy_path):
        buy_model = PPO.load(final_buy_path, device=device)
        print(f"⚠️ Loaded Final Buy Model (Best not found): {final_buy_path}")
    else:
        print("❌ Error: No Buy Model found! Please train the agent first.")
        sys.exit(1)
        
    # Sell Model
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_paper", "best_model.zip")
    final_sell_path = os.path.join(MODELS_PATH, "ppo_sell_paper_final.zip")
    
    if os.path.exists(best_sell_path):
        sell_model = PPO.load(best_sell_path, device=device)
        print(f"✅ Loaded Best Sell Model: {best_sell_path}")
    elif os.path.exists(final_sell_path):
        sell_model = PPO.load(final_sell_path, device=device)
        print(f"⚠️ Loaded Final Sell Model (Best not found): {final_sell_path}")
    else:
        print("❌ Error: No Sell Model found! Please train the agent first.")
        sys.exit(1)

    # 4. Interactive Input
    print("\n=== Flexible Backtest Settings ===")
    default_start = "2022-01-01"
    default_end = "2023-12-31"
    
    start_input = input(f"Enter Start Date (YYYY-MM-DD) [Default: {default_start}]: ").strip()
    end_input = input(f"Enter End Date (YYYY-MM-DD) [Default: {default_end}]: ").strip()
    
    start_date = start_input if start_input else default_start
    end_date = end_input if end_input else default_end
    
    # 5. Run Backtest
    print(f"\nStarting backtest from {start_date} to {end_date}...")
    try:
        analyzer = FlexibleBacktester(stock_data_only, buy_model, sell_model, start_date, end_date)
        daily_stats, trade_logs = analyzer.run()
        
        if not daily_stats.empty:
            # 6. Results
            print("\n=== Backtest Results ===")
            initial_val = daily_stats.iloc[0]['Total_Value']
            final_val = daily_stats.iloc[-1]['Total_Value']
            roi = (final_val - initial_val) / initial_val * 100
            
            print(f"Initial Capital: {initial_val:,.0f}")
            print(f"Final Capital:   {final_val:,.0f}")
            print(f"Total Return:    {roi:.2f}%")
            
            # --- Benchmark Comparison (0050.TW) ---
            bench_roi = 0.0
            bench_data = None
            if "0050.TW" in raw_data_dict:
                bench_df = raw_data_dict["0050.TW"]
                # Filter benchmark data to match backtest dates
                mask = (bench_df.index >= pd.Timestamp(start_date)) & (bench_df.index <= pd.Timestamp(end_date))
                bench_data = bench_df.loc[mask].copy()
                
                if not bench_data.empty:
                    # Align with daily_stats index if possible, or just use date range
                    # Normalize benchmark to initial capital
                    bench_start_price = bench_data.iloc[0]['Close']
                    bench_end_price = bench_data.iloc[-1]['Close']
                    bench_roi = (bench_end_price - bench_start_price) / bench_start_price * 100
                    
                    # Create normalized series for plotting (scaled to initial capital)
                    bench_data['Normalized_Value'] = (bench_data['Close'] / bench_start_price) * initial_val
                    
                    print(f"0050 Benchmark:  {bench_roi:.2f}%")
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(daily_stats.index, daily_stats['Total_Value'], label=f'AI Portfolio (ROI: {roi:.2f}%)', color='red', linewidth=2)
            
            if bench_data is not None and not bench_data.empty:
                 plt.plot(bench_data.index, bench_data['Normalized_Value'], label=f'0050 Benchmark (ROI: {bench_roi:.2f}%)', color='gray', linestyle='--')

            plt.title(f'Backtest Result ({start_date} to {end_date})')
            plt.xlabel('Date')
            plt.ylabel('Total Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_filename = f"backtest_{start_date}_{end_date}.png"
            plot_path = os.path.join(RESULTS_PATH, plot_filename)
            plt.savefig(plot_path)
            print(f"\nPlot saved to: {plot_path}")
            plt.show()
            
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
