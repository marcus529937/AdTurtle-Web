"""CLI 工具：python single_stock_runner.py 2330"""
import sys
from market_data       import get_price_history, get_recent_volume
from strategy_adturtle import adturtle_simple


def main():
    if len(sys.argv) < 2:
        print("用法：python single_stock_runner.py <代碼>  (例如 2330)")
        sys.exit(1)

    ticker = sys.argv[1].strip()
    print(f"\n正在抓取 {ticker}.TW 資料…")
    df = get_price_history(ticker)

    if df is None or df.empty:
        print("❌ 無法取得資料")
        sys.exit(1)

    print(f"✅ 取得 {len(df)} 筆 OHLCV")
    print(f"欄位：{df.columns.tolist()}")

    result = adturtle_simple(ticker, df)
    print(f"\n── 策略結果 ──")
    print(f"訊號       : {result.signal}")
    print(f"最新收盤   : {result.last_price}")
    print(f"進場參考價 : {result.entry_price}")
    print(f"突破上軌   : {result.upper_band}")
    print(f"突破下軌   : {result.lower_band}")
    print(f"停損線     : {result.stop_loss}")
    print(f"報酬率     : {result.return_pct}%")
    print(f"未實現損益 : {result.unrealized_pnl}")
    if result.error:
        print(f"錯誤       : {result.error}")

    vol = get_recent_volume(df, n=5)
    print(f"\n近五日成交量：{vol}")


if __name__ == "__main__":
    main()
