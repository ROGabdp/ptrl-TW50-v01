import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def investigate():
    ticker = "^TWII"
    start_date = "2013-01-01"
    end_date = "2015-01-01"
    
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    try:
        df_adj = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        df_raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    if df_adj.empty:
        print("No data found!")
        return

    print("\n--- Adjusted Data (auto_adjust=True) ---")
    print(df_adj.head())
    print(df_adj.tail())
    
    print("\n--- Raw Data (auto_adjust=False) ---")
    print(df_raw.head())
    print(df_raw.tail())

    # Check around 2014-01-01 for Raw Data
    print("\nRaw Data around 2014-01-01:")
    try:
        around_2014_raw = df_raw.loc["2013-12-01":"2014-02-01"]
        print(around_2014_raw[['Close']])
    except Exception as e:
        print(f"Could not slice data: {e}")

    df = df_adj # Use adjusted for drop check

    # Check for large drops
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df['Pct_Change'] = df['Close'].pct_change()
    large_drops = df[df['Pct_Change'] < -0.10] # Drops > 10%
    
    if not large_drops.empty:
        print("\n⚠️ Large drops detected:")
        print(large_drops[['Close', 'Pct_Change']])
    else:
        print("\n✅ No drops > 10% detected.")
    
    import sys
    sys.stdout.flush()

if __name__ == "__main__":
    investigate()
