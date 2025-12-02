# -*- coding: utf-8 -*-
"""
Pro Trader RL (Backtest Only)
此程式用於載入已訓練好的模型進行回測，不進行訓練。
"""
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
# 從主程式匯入必要函式與類別
from ptrl_TW50_local_gpu import (
    setup_environment,
    fetch_tw50_data,
    calculate_features,
    DetailedBacktester,
    FEATURE_COLS
)

if __name__ == "__main__":
    # 1. 設定環境
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    print(f"Running backtest on device: {device}")

    # 2. 下載與處理資料
    # 注意：這裡會重跑一次資料下載與特徵工程，確保資料最新
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

    # 3. 載入模型
    # 嘗試載入 Best Buy Model
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_model", "best_model.zip")
    final_buy_path = os.path.join(MODELS_PATH, "ppo_buy_tw50.zip")
    
    if os.path.exists(best_buy_path):
        print(f"✅ 載入最佳買入模型: {best_buy_path}")
        buy_model = PPO.load(best_buy_path, device=device)
    elif os.path.exists(final_buy_path):
        print(f"⚠️ 找不到最佳模型，載入最終買入模型: {final_buy_path}")
        buy_model = PPO.load(final_buy_path, device=device)
    else:
        raise FileNotFoundError(f"❌ 找不到任何買入模型！請先執行 ptrl_TW50_local_gpu.py 進行訓練。")

    # 載入 Sell Model
    sell_path = os.path.join(MODELS_PATH, "ppo_sell_tw50.zip")
    if os.path.exists(sell_path):
        print(f"✅ 載入賣出模型: {sell_path}")
        sell_model = PPO.load(sell_path, device=device)
    else:
        raise FileNotFoundError(f"❌ 找不到賣出模型：{sell_path}")

    # 4. 執行回測
    print("\n=== 開始執行回測 ===")
    stock_data_only = {k: v for k, v in processed_data.items() if k != "^TWII" and k != "0050.TW"}
    analyzer = DetailedBacktester(stock_data_only, buy_model, sell_model)
    daily_stats = analyzer.run()

    # 5. 輸出結果
    print("\n=== 回測結果 ===")
    print(f"期初資金: {daily_stats.iloc[0]['Total_Value']:.0f}")
    print(f"期末資金: {daily_stats.iloc[-1]['Total_Value']:.0f}")
    
    roi = (daily_stats.iloc[-1]['Total_Value'] - daily_stats.iloc[0]['Total_Value']) / daily_stats.iloc[0]['Total_Value']
    print(f"總報酬率: {roi*100:.2f}%")

    # 繪圖
    plt.figure(figsize=(12, 6))
    daily_stats['Total_Value'].plot(title=f'AI Portfolio Value (ROI: {roi*100:.2f}%)', grid=True)
    save_path = os.path.join(RESULTS_PATH, "backtest_performance.png")
    plt.savefig(save_path)
    print(f"圖表已儲存至: {save_path}")
    # plt.show() # WSL2 環境通常不顯示視窗，直接存檔即可
