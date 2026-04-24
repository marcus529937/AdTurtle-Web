"""
AdTurtle 單股測試工具
位置：
    adturtle-lab/single_stock_runner.py

預設：
    抓最近 60 個交易日資料來跑策略
    可透過第二個參數調整天數

用法：
    python adturtle-lab/single_stock_runner.py 0052
    python adturtle-lab/single_stock_runner.py 2330 120
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
LAB_DIR = CURRENT_FILE.parent
PROJECT_ROOT = LAB_DIR.parent

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

    return work_df


def slice_last_n_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    從完整 df 中裁出「最近 days 個交易日」，
    只做顯示與策略運算用。
    """
    if df is None or df.empty:
        return df
    if days <= 0:
        return df
    # 直接用 tail，按交易日數裁切（比算日期區間直覺）
    return df.tail(days)


def print_header(symbol, df_view, total_len, cache_dir, days):
    start_date = str(df_view.index.min())[:10]
    end_date = str(df_view.index.max())[:10]

    print(LINE)
    print(f"AdTurtle 單股測試 | {symbol}")
    print(LINE)
    print(f"觀察天數      {days} (最近 {days} 個交易日)")
    print(f"資料區間      {start_date} ~ {end_date}")
    print(f"資料筆數      {len(df_view)} / 原始 {total_len}")
    print(f"快取路徑      {cache_dir}")


def print_market_block(df_view):
    latest = df_view.iloc[-1]
    recent_vol = get_recent_volume(df_view, n=min(10, len(df_view)))

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

    rows = result.signal_rows[:limit]
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


def main():
    if len(sys.argv) < 2:
        print("用法：python adturtle-lab/single_stock_runner.py <股票代碼> [lookback_days]")
        print("例如：python adturtle-lab/single_stock_runner.py 0052")
        print("例如：python adturtle-lab/single_stock_runner.py 2330 120")
        sys.exit(1)

    symbol = sys.argv[1].strip()
    lookback_days = 60  # 預設 60 天

    if len(sys.argv) >= 3:
        try:
            lookback_days = int(sys.argv[2])
        except ValueError:
            print("lookback_days 必須是整數")
            sys.exit(1)

    try:
        # 抓一段稍長的 period，避免資料不足
        # 讓 60 天、120 天這類 lookback 也有 buffer 可以用
        if lookback_days <= 60:
            period = "6mo"
        elif lookback_days <= 120:
            period = "1y"
        else:
            period = "2y"

        df_full = get_price_history(symbol, period=period, use_cache=True)

        if df_full is None or df_full.empty:
            print("❌ 無法取得資料")
            sys.exit(1)

        df_full = normalize_price_df(df_full)
        if df_full is None or df_full.empty:
            print("❌ 清理後資料為空")
            sys.exit(1)

        # 只拿最近 lookback_days 個交易日給策略與顯示用
        df_view = slice_last_n_days(df_full, lookback_days)

        if df_view is None or df_view.empty:
            print("❌ 無法切出指定天數的資料")
            sys.exit(1)

        result = adturtle_simple(
            symbol=symbol,
            df=df_view,
            upper_period=20,
            lower_period=10,
            lookback_days=lookback_days
        )

        print_header(symbol, df_view, total_len=len(df_full), cache_dir=market_data.CACHE_DIR, days=lookback_days)
        print_market_block(df_view)
        print_strategy_block(result)
        print_performance_block(result)
        print_recent_signals(result, limit=5)
        print_recent_trades(result, limit=5)
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