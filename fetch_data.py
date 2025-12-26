import yfinance as yf
import pandas as pd

ticker = "AAPL"

df = yf.download(ticker, start = "2021-01-01", end = "2025-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

    
df.reset_index(inplace=True)

df.to_csv('raw_data.csv', index=False)

print("Data fetched and saved to raw_data.csv")
print(df.head())

