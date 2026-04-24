from market_data import get_price_history
import pandas as pd

df = get_price_history("2330")
print(df.tail(3).to_string())
print()
print("last_date =", df.index[-1])
print("last_close =", df["Close"].iloc[-1])
print("is_nan =", pd.isna(df["Close"].iloc[-1]))