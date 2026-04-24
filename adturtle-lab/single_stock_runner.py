import sys
import traceback
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

CURRENT_FILE = Path(__file__).resolve()
LAB_DIR = CURRENT_FILE.parent
PROJECT_ROOT = LAB_DIR.parent
OUTPUT_DIR = LAB_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import market_data
from market_data import get_price_history, get_recent_volume
from strategy_adturtle import adturtle_simple


LINE = "=" * 60
SUBLINE = "-" * 60


def fmt_num(value, digits=2, default="-"):
    if value is None:
        return default
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def fmt_int(value, default="-"):
    if value is None:
        return default
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def normalize_price_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    work_df = df.copy()

    if isinstance(work_df.columns, pd.MultiIndex):
        work_df.columns = work_df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in work_df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")

    work_df = work_df.dropna(subset=required).copy()
    if work_df.empty:
        return None

    work_df.sort_index(inplace=True)
    if not isinstance(work_df.index, pd.DatetimeIndex):
        work_df.index = pd.to_datetime(work_df.index)

    return work_df


def print_header(symbol: str, df: pd.DataFrame, lookback_days: int):
    start_date = str(df.index.min())[:10]
    end_date = str(df.index.max())[:10]

    print(LINE)
    print(f"AdTurtle 單股回測 | {symbol}")
    print(LINE)
    print(f"設定觀察天數  {lookback_days}（lookback_days）")
    print(f"實際資料區間  {start_date} ~ {end_date}")
    print(f"實際資料筆數  {len(df)}")
    print(f"快取路徑      {market_data.CACHE_DIR}")


def print_market_block(df: pd.DataFrame):
    latest = df.iloc[-1]
    recent_vol = get_recent_volume(df, n=min(10, len(df)))

    print("\n[市場資料]")
    print(f"最新收盤      {fmt_num(latest['Close'])}")
    print(f"最新開高低    O:{fmt_num(latest['Open'])}  H:{fmt_num(latest['High'])}  L:{fmt_num(latest['Low'])}")
    print(f"最新成交量    {fmt_int(latest['Volume'])}")
    if recent_vol:
        print(f"最近十日量    {', '.join(fmt_int(v) for v in recent_vol)}")
    else:
        print("最近十日量    -")


def print_strategy_block(result):
    position_text = f"持倉中 ({result.shares_held:,} 股)" if result.in_position else "空手"

    print("\n[策略狀態]")
    print(f"最新訊號      {result.latest_signal}")
    print(f"持倉狀態      {position_text}")
    print(f"最新價格      {fmt_num(result.last_price)}")
    print(f"進場成本      {fmt_num(result.entry_price)}")
    print(f"上軌 / 下軌   {fmt_num(result.upper_band)} / {fmt_num(result.lower_band)}")
    print(f"停損線        {fmt_num(result.stop_loss)}")

    if result.error:
        print(f"錯誤訊息      {result.error}")


def print_performance_block(result):
    print("\n[績效摘要]")
    print(f"總報酬率      {fmt_num(result.return_pct)}%")
    print(f"總損益        {fmt_int(result.total_pnl)}")
    print(f"已實現損益    {fmt_int(result.realized_pnl)}")
    print(f"未實現損益    {fmt_int(result.unrealized_pnl)}")
    print(f"總資產        {fmt_num(result.equity)}")
    print(f"總交易數      {result.total_trades}")
    print(f"已平倉數      {result.closed_trades}")
    print(f"勝率          {fmt_num(result.win_rate)}% ({result.win_trades}/{result.closed_trades})")


def print_recent_signals(result, limit=5):
    print("\n[最近訊號]")

    if not result.signal_rows:
        print("無訊號資料")
        return

    rows = result.signal_rows[-limit:][::-1]
    for row in rows:
        date = row.get("date", "-")
        signal = row.get("signal", "-")
        close = fmt_num(row.get("close"))
        channel = fmt_num(row.get("channel_price"))
        volume = fmt_int(row.get("volume"))
        shares = fmt_int(row.get("shares"))
        pnl = row.get("pnl")

        line = (
            f"{date}  "
            f"{signal:<4}  "
            f"close={close}  "
            f"channel={channel}  "
            f"volume={volume}"
        )

        if shares != "-":
            line += f"  shares={shares}"
        if pnl is not None:
            line += f"  pnl={fmt_int(pnl)}"

        print(line)


def print_recent_trades(result, limit=5):
    print("\n[最近交易]")

    if not result.trade_log:
        print("無交易紀錄")
        return

    trades = result.trade_log[-limit:]
    for trade in trades:
        entry_date = trade.entry_date or "-"
        exit_date = trade.exit_date or "-"
        entry_price = fmt_num(trade.entry_price)
        exit_price = fmt_num(trade.exit_price)
        shares = fmt_int(trade.shares)
        pnl = fmt_int(trade.realized_pnl)
        ret = fmt_num(trade.return_pct)

        print(
            f"{entry_date} -> {exit_date} | "
            f"{trade.status:<6} | "
            f"entry={entry_price}  "
            f"exit={exit_price}  "
            f"shares={shares}  "
            f"pnl={pnl}  "
            f"ret={ret}%"
        )


def get_backtest_window(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if len(df) <= lookback_days:
        return df.copy()

    return df.tail(lookback_days).copy()


def build_equity_curve_from_window(window_df: pd.DataFrame, result) -> pd.DataFrame:
    """
    依「最近 N 個交易日重新開局」語意重建 equity curve：
    - 初始資金：result.initial_capital
    - 僅使用 window_df 內資料
    - BUY / SELL 以當日收盤價成交
    - 每次 BUY 全額買入，SELL 全數賣出
    """
    if window_df is None or window_df.empty:
        return pd.DataFrame()

    capital = float(result.initial_capital)
    shares = 0

    signal_map = {}
    for row in result.signal_rows:
        d = row.get("date")
        if not d:
            continue
        signal_map.setdefault(d, []).append(row)

    equity_records = []

    for ts, r in window_df.iterrows():
        date_str = ts.strftime("%Y-%m-%d")
        close = float(r["Close"])

        for s in signal_map.get(date_str, []):
            sig = s.get("signal")
            if sig == "BUY":
                buy_shares = int(capital // close)
                if buy_shares > 0:
                    capital -= buy_shares * close
                    shares += buy_shares
            elif sig == "SELL":
                if shares > 0:
                    capital += shares * close
                    shares = 0

        equity = capital + shares * close
        equity_records.append(
            {
                "date": date_str,
                "close": round(close, 2),
                "equity": round(equity, 2),
            }
        )

    eq_df = pd.DataFrame(equity_records).set_index("date")
    return eq_df


def save_equity_csv(symbol: str, lookback_days: int, eq_df: pd.DataFrame):
    csv_path = OUTPUT_DIR / f"{symbol}_equity_{lookback_days}d.csv"
    eq_df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"Equity CSV    : {csv_path}")


def save_backtest_figure(
    symbol: str,
    lookback_days: int,
    full_df: pd.DataFrame,
    result,
    upper_period: int = 20,
    lower_period: int = 10,
):
    """
    最近 N 個交易日重新開局版：
    - 僅繪製最後 N 個交易日的 K 線
    - 疊加 Donchian channel
    - 疊加 BUY / SELL 訊號
    - 不畫 equity 子圖
    """
    if full_df is None or full_df.empty:
        print("⚠️ Cannot plot: empty price DataFrame")
        return

    price_view = get_backtest_window(full_df, lookback_days)
    if price_view.empty:
        print("⚠️ Cannot plot: empty backtest window")
        return

    price_view = price_view.copy()
    price_view["upper_band"] = price_view["High"].rolling(upper_period).max()
    price_view["lower_band"] = price_view["Low"].rolling(lower_period).min()

    buy_markers = pd.Series(index=price_view.index, dtype=float)
    sell_markers = pd.Series(index=price_view.index, dtype=float)

    for row in result.signal_rows:
        d = row.get("date")
        sig = row.get("signal")
        c = row.get("close")
        if not d or c is None:
            continue

        ts = pd.to_datetime(d)
        if ts not in price_view.index:
            continue

        if sig == "BUY":
            buy_markers.loc[ts] = float(c)
        elif sig == "SELL":
            sell_markers.loc[ts] = float(c)

    addplots = [
        mpf.make_addplot(
            price_view["upper_band"],
            color="#fb8c00",
            width=1.2,
        ),
        mpf.make_addplot(
            price_view["lower_band"],
            color="#1565c0",
            width=1.2,
        ),
        mpf.make_addplot(
            buy_markers,
            type="scatter",
            marker="^",
            markersize=120,
            color="#2e7d32",
        ),
        mpf.make_addplot(
            sell_markers,
            type="scatter",
            marker="v",
            markersize=120,
            color="#c62828",
        ),
    ]

    png_path = OUTPUT_DIR / f"{symbol}_backtest_{lookback_days}d.png"

    fig, axes = mpf.plot(
        price_view,
        type="candle",
        style="yahoo",
        volume=False,
        addplot=addplots,
        figsize=(14, 7),
        returnfig=True,
        show_nontrading=False,
        datetime_format="%Y-%m-%d",
        title=f"{symbol} price / channel / signals - last {lookback_days} days",
        ylabel="Price",
    )

    ax = axes[0]
    ax.legend(
        ["upper_band", "lower_band", "BUY", "SELL"],
        loc="upper left",
    )
    ax.grid(True, alpha=0.2)

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Backtest PNG  : {png_path}")


def main():
    if len(sys.argv) < 2:
        print("用法：python adturtle-lab/single_stock_runner.py <股票代碼> [lookback_days]")
        print("例如：python adturtle-lab/single_stock_runner.py 0052")
        print("例如：python adturtle-lab/single_stock_runner.py 2330 120")
        sys.exit(1)

    symbol = sys.argv[1].strip()
    lookback_days = 60

    if len(sys.argv) >= 3:
        try:
            lookback_days = int(sys.argv[2])
        except ValueError:
            print("lookback_days 必須是整數")
            sys.exit(1)

    if lookback_days <= 0:
        print("lookback_days 必須 > 0")
        sys.exit(1)

    try:
        df = get_price_history(symbol, period="max", use_cache=True)

        if df is None or df.empty:
            print("❌ 無法取得資料")
            sys.exit(1)

        df = normalize_price_df(df)
        if df is None or df.empty:
            print("❌ 清理後資料為空")
            sys.exit(1)

        result = adturtle_simple(
            symbol=symbol,
            df=df,
            upper_period=20,
            lower_period=10,
            lookback_days=lookback_days,
        )

        print_header(symbol, df, lookback_days)
        print_market_block(df)
        print_strategy_block(result)
        print_performance_block(result)
        print_recent_signals(result, limit=5)
        print_recent_trades(result, limit=5)

        window_df = get_backtest_window(df, lookback_days)
        eq_df = build_equity_curve_from_window(window_df, result)

        save_backtest_figure(
            symbol=symbol,
            lookback_days=lookback_days,
            full_df=df,
            result=result,
            upper_period=20,
            lower_period=10,
        )
        save_equity_csv(symbol, lookback_days, eq_df)

        print(f"\n{SUBLINE}")

    except Exception as e:
        print(LINE)
        print("執行失敗")
        print(LINE)
        print(f"錯誤類型: {type(e).__name__}")
        print(f"錯誤訊息: {e}")
        print("\n--- traceback ---")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()