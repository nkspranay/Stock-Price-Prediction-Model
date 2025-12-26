import pandas as pd

df = pd.read_csv('raw_data.csv')

df["Date"] = pd.to_datetime(df["Date"])



price_columns = ["Close","High","Low","Open","Volume"]
for col in price_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')



df["MA_10"] = df["Close"].rolling(window=10).mean()  # 10-day moving average
df["MA_50"] = df["Close"].rolling(window=50).mean()  # 50-day moving average
df["Return"] = df["Close"].pct_change()     # Daily return calculation, pct_change gives percentage change between the current and a prior element
df["Volatility"] = df["Return"].rolling(window=10).std() #calculates the standard deviation of the return of the past 10 days, this giving the volatility(stability)
df.dropna(inplace= True)  # Drop rows with missing values

df.to_csv('cleaned_data.csv', index=False)  
print("Feature engineering completed and saved to cleaned_data.csv")
