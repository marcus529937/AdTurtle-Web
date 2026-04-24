import yfinance as yf
df = yf.download("2330.TW", period="5d")
print(df)