import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

# Import functions from the main script
# Assuming ptrl_TW50_split_train_ma_filter.py is in the same directory
try:
    from ptrl_TW50_split_train_ma_filter import setup_environment, fetch_tw50_data, calculate_features, FEATURE_COLS
except ImportError:
    # Fallback if import fails (e.g. if running from a different directory)
    sys.path.append(os.getcwd())
    from ptrl_TW50_split_train_ma_filter import setup_environment, fetch_tw50_data, calculate_features, FEATURE_COLS

def check_balance():
    # 1. Setup
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # 2. Fetch Data
    raw_data_dict, market_index_df = fetch_tw50_data(DATA_PATH, start_date="2000-01-01")
    
    print("Calculating features...")
    processed_data = {}
    bench_df = raw_data_dict.get("^TWII")
    for ticker, df in raw_data_dict.items():
        if ticker != "^TWII":
            try:
                processed_data[ticker] = calculate_features(df, bench_df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
    # 3. Split Data (Train Period: 2000-2011 AND 2018-Present)
    print("\n=== Data Splitting ===")
    print("Train Period: 2000-2011 AND 2018-Present")
    
    train_data = {}
    
    for ticker, df in processed_data.items():
        if ticker == "0050.TW": continue
        
        # Create masks
        test_mask = (df.index >= '2012-01-01') & (df.index <= '2017-12-31')
        train_mask = ~test_mask
        
        df_train = df[train_mask].copy()
        
        if not df_train.empty:
            train_data[ticker] = df_train
            
    print(f"Training Tickers: {len(train_data)}")

    # 4. Count Samples
    print("\n=== Counting Samples (Buy Agent) ===")
    total_pos = 0
    total_neg = 0
    total_samples = 0
    
    ticker_stats = []

    for t, df in train_data.items():
        # Ensure Next_20d_Max is valid
        df = df.dropna(subset=['Next_20d_Max'])
        
        # Apply Buy Filter: Close < MA60
        signals = df[df['Signal_Buy_Filter'] == True]
        
        n_pos = 0
        n_neg = 0
        
        if len(signals) > 0:
            future_rets = signals['Next_20d_Max'].values
            
            # Positive: >= 10% profit in 20 days
            n_pos = np.sum(future_rets >= 0.10)
            n_neg = np.sum(future_rets < 0.10)
            
        total_pos += n_pos
        total_neg += n_neg
        total_samples += len(signals)
        
        ticker_stats.append({
            'Ticker': t,
            'Pos': n_pos,
            'Neg': n_neg,
            'Total': len(signals),
            'Pos_Rate': (n_pos / len(signals) * 100) if len(signals) > 0 else 0.0
        })

    # 5. Report
    print(f"\nTotal Samples (After MA Filter): {total_samples}")
    print(f"Total Positive Samples (>= 10% in 20 days): {total_pos}")
    print(f"Total Negative Samples (< 10% in 20 days): {total_neg}")
    
    if total_samples > 0:
        pos_ratio = total_pos / total_samples * 100
        print(f"Overall Positive Ratio: {pos_ratio:.2f}%")
        print(f"Class Imbalance (Neg/Pos): {total_neg / total_pos if total_pos > 0 else 'Inf'}")
    else:
        print("No samples found!")

    print("\n--- Top 10 Tickers by Positive Samples ---")
    ticker_stats.sort(key=lambda x: x['Pos'], reverse=True)
    for stat in ticker_stats[:10]:
        print(f"{stat['Ticker']}: Pos={stat['Pos']}, Neg={stat['Neg']}, Total={stat['Total']}, Rate={stat['Pos_Rate']:.1f}%")

if __name__ == "__main__":
    check_balance()
