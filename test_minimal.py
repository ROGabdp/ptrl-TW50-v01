import sys
print("Python version:", sys.version)
try:
    import yfinance as yf
    print("yfinance imported successfully")
except Exception as e:
    print(f"Error importing yfinance: {e}")

print("Done")
