import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.lines import Line2D

COLOR_BG = "#ffffff"
COLOR_PANEL = "#ffffff"
COLOR_TEXT = "#1a1a2e"
COLOR_MUTED = "#9e9e9e"
COLOR_GRID = "#e0e0e0"
COLOR_UP = "#c62828"
COLOR_DOWN = "#2e7d32"
COLOR_BUY = "#fb8c00"
COLOR_SELL = "#1565c0"

DEFAULT_LOOKBACK_DAYS = 60

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


def fmt_num(value, digits: int = 2, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def fmt_int(value, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def normalize_price_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
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


def print_header(symbol: str, df: pd.DataFrame, lookback_days: int) -> None:
    start_date = str(df.index.min())[:10]
    end_date = str(df.index.max())[:10]

    print(LINE)
    print(f"AdTurtle 單股回測 | {symbol}")
    print(LINE)
    print(f"設定觀察天數   : {lookback_days}（lookback_days）")
    print(f"資料區間       : {start_date} ~ {end_date}")
    print(f"資料筆數       : {len(df)}")


def print_market_block(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    recent_vol = get_recent_volume(df, n=min(10, len(df)))

    print("\n[市場資料]")
    print(
        "最新開高低收   : "
        f"O:{fmt_num(latest['Open'])}  "
        f"H:{fmt_num(latest['High'])}  "
        f"L:{fmt_num(latest['Low'])}  "
        f"C:{fmt_num(latest['Close'])}"
    )
    print(f"最新成交量     : {fmt_int(latest['Volume'])}")
    if recent_vol:
        print(f"最近十日量     : {', '.join(fmt_int(v) for v in recent_vol)}")
    else:
        print("最近十日量     : -")


def print_strategy_block(result) -> None:
    position_text = (
        f"持倉中 ({result.shares_held:,} 股)" if result.in_position else "空手"
    )

    print("\n[策略狀態]")
    print(f"最新訊號       : {result.latest_signal}")
    print(f"持倉狀態       : {position_text}")
    print(f"最新價格       : {fmt_num(result.last_price)}")
    print(f"進場成本       : {fmt_num(result.entry_price)}")
    print(f"上軌 / 下軌    : {fmt_num(result.upper_band)} / {fmt_num(result.lower_band)}")
    print(f"停損線         : {fmt_num(result.stop_loss)}")

    if result.error:
        print(f"錯誤訊息       : {result.error}")


def print_performance_block(result) -> None:
    print("\n[績效摘要]")
    print(f"總報酬率       : {fmt_num(result.return_pct)}%")
    print(f"總損益         : {fmt_int(result.total_pnl)}")
    print(f"已實現損益     : {fmt_int(result.realized_pnl)}")
    print(f"未實現損益     : {fmt_int(result.unrealized_pnl)}")
    print(f"總資產         : {fmt_num(result.equity)}")
    print(f"總交易數       : {result.total_trades}")
    print(
        f"勝率           : {fmt_num(result.win_rate)}% "
        f"({result.win_trades}/{result.closed_trades})"
    )


def _signal_line(row) -> str:
    date = row.get("date", "-")
    signal = row.get("signal", "-")
    close = fmt_num(row.get("close"))
    channel = fmt_num(row.get("channel_price"))
    volume = fmt_int(row.get("volume"))
    shares = fmt_int(row.get("shares"))
    pnl = row.get("pnl")

    line = (
        f"{date}  {signal:<4}  close={close}  channel={channel}  volume={volume}"
    )
    if shares != "-":
        line += f"  shares={shares}"
    if pnl is not None:
        line += f"  pnl={fmt_int(pnl)}"
    return line


def _trade_unrealized_metrics(trade, last_price):
    if getattr(trade, "status", "") != "OPEN":
        return None, None
    if last_price is None or trade.entry_price is None or not trade.shares:
        return None, None
    try:
        unrealized_pnl = (float(last_price) - float(trade.entry_price)) * float(trade.shares)
        unrealized_ret = ((float(last_price) / float(trade.entry_price)) - 1.0) * 100.0
        return unrealized_pnl, unrealized_ret
    except Exception:
        return None, None


def _trade_line(trade, last_price=None) -> str:
    entry_date = trade.entry_date or "-"
    exit_date = trade.exit_date or "-"
    entry_price = fmt_num(trade.entry_price)
    exit_price = fmt_num(trade.exit_price)
    shares = fmt_int(trade.shares)
    pnl = fmt_int(trade.realized_pnl)
    ret = fmt_num(trade.return_pct)

    line = (
        f"{entry_date} -> {exit_date} | "
        f"{trade.status:<6} | "
        f"entry={entry_price}  "
        f"exit={exit_price}  "
        f"shares={shares}"
    )

    if getattr(trade, "status", "") == "OPEN":
        unrealized_pnl, unrealized_ret = _trade_unrealized_metrics(trade, last_price)
        line += (
            f"  unrealized_pnl={fmt_int(unrealized_pnl)}"
            f"  unrealized_ret={fmt_num(unrealized_ret)}%"
        )
    else:
        line += f"  pnl={pnl}  ret={ret}%"

    return line


def save_detail_txt(symbol: str, lookback_days: int, result, last_price=None):
    txt_path = OUTPUT_DIR / f"{symbol}_details_{lookback_days}d.txt"
    lines = []
    lines.append("[最近訊號 - 全部]\n")
    if result.signal_rows:
        for row in result.signal_rows:
            lines.append(_signal_line(row))
    else:
        lines.append("無訊號資料")

    lines.append("\n[最近交易 - 全部]\n")
    if result.trade_log:
        for trade in result.trade_log:
            lines.append(_trade_line(trade, last_price=last_price))
    else:
        lines.append("無交易紀錄")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Detail TXT    : {txt_path}")


def print_recent_signals(result, lookback_days: int, default_lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
    print("\n[最近訊號]")

    if not result.signal_rows:
        print("無訊號資料")
        return

    if lookback_days == default_lookback_days:
        rows = result.signal_rows
    else:
        rows = result.signal_rows[-5:][::-1]

    for row in rows:
        print(_signal_line(row))



def print_recent_trades(result, lookback_days: int, last_price=None, default_lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
    print("\n[最近交易]")

    if not result.trade_log:
        print("無交易紀錄")
        return

    if lookback_days == default_lookback_days:
        trades = result.trade_log
    else:
        trades = result.trade_log[-5:]

    for trade in trades:
        print(_trade_line(trade, last_price=last_price))


def get_backtest_window(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if len(df) <= lookback_days:
        return df.copy()

    return df.tail(lookback_days).copy()


def build_equity_curve_from_window(window_df: pd.DataFrame, result) -> pd.DataFrame:
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


def _build_price_and_signals(
    full_df: pd.DataFrame,
    result,
    lookback_days: int,
    upper_period: int,
    lower_period: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    price_view = get_backtest_window(full_df, lookback_days)
    if price_view.empty:
        raise ValueError("empty backtest window")

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

    return price_view, buy_markers, sell_markers


def _build_mpf_style():
    market_colors = mpf.make_marketcolors(
        up=COLOR_UP,
        down=COLOR_DOWN,
        edge={"up": COLOR_UP, "down": COLOR_DOWN},
        wick={"up": COLOR_UP, "down": COLOR_DOWN},
        volume={"up": COLOR_UP, "down": COLOR_DOWN},
        ohlc={"up": COLOR_UP, "down": COLOR_DOWN},
    )

    return mpf.make_mpf_style(
        marketcolors=market_colors,
        base_mpf_style="default",
        facecolor=COLOR_BG,
        figcolor=COLOR_BG,
        edgecolor=COLOR_GRID,
        gridcolor=COLOR_GRID,
        gridstyle="-",
        y_on_right=False,
        rc={
            "axes.facecolor": COLOR_PANEL,
            "axes.edgecolor": COLOR_GRID,
            "axes.labelcolor": COLOR_TEXT,
            "axes.titlecolor": COLOR_TEXT,
            "figure.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,
            "savefig.edgecolor": COLOR_BG,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "text.color": COLOR_TEXT,
            "font.size": 10,
        },
    )


def save_backtest_figure(
    symbol: str,
    lookback_days: int,
    full_df: pd.DataFrame,
    result,
    upper_period: int = 20,
    lower_period: int = 10,
):
    if full_df is None or full_df.empty:
        print("⚠️ Cannot plot: empty price DataFrame")
        return

    try:
        price_view, buy_markers, sell_markers = _build_price_and_signals(
            full_df=full_df,
            result=result,
            lookback_days=lookback_days,
            upper_period=upper_period,
            lower_period=lower_period,
        )
    except ValueError as e:
        print(f"⚠️ Cannot plot: {e}")
        return

    addplots = [
        mpf.make_addplot(price_view["upper_band"], color=COLOR_BUY, width=1.3),
        mpf.make_addplot(price_view["lower_band"], color=COLOR_SELL, width=1.3),
        mpf.make_addplot(buy_markers, type="scatter", marker="^", markersize=90, color=COLOR_BUY),
        mpf.make_addplot(sell_markers, type="scatter", marker="v", markersize=90, color=COLOR_SELL),
    ]

    mpf_style = _build_mpf_style()
    png_path = OUTPUT_DIR / f"{symbol}_backtest_{lookback_days}d.png"

    fig, axes = mpf.plot(
        price_view,
        type="candle",
        style=mpf_style,
        volume=False,
        addplot=addplots,
        figsize=(14, 7),
        returnfig=True,
        show_nontrading=False,
        datetime_format="%Y-%m-%d",
        title=f"{symbol} - AdTurtle channel & signals (last {lookback_days} days)",
        ylabel="Price",
    )

    ax = axes[0]
    ax.set_facecolor(COLOR_PANEL)
    ax.grid(True, color=COLOR_GRID, alpha=0.6, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)

    legend_handles = [
        Line2D([0], [0], color=COLOR_BUY, lw=1.8, label="upper_band"),
        Line2D([0], [0], color=COLOR_SELL, lw=1.8, label="lower_band"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_BUY, markeredgecolor=COLOR_BUY, markersize=8, linestyle="None", label="BUY"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=COLOR_SELL, markeredgecolor=COLOR_SELL, markersize=8, linestyle="None", label="SELL"),
    ]
    legend = ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=9)
    legend.get_frame().set_facecolor(COLOR_BG)
    legend.get_frame().set_edgecolor(COLOR_GRID)
    legend.get_frame().set_alpha(0.95)

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Backtest PNG  : {png_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python adturtle-lab/single_stock_runner.py <股票代碼> [lookback_days]")
        print("例如：python adturtle-lab/single_stock_runner.py 0052")
        print("例如：python adturtle-lab/single_stock_runner.py 2330 120")
        sys.exit(1)

    symbol = sys.argv[1].strip()
    lookback_days = DEFAULT_LOOKBACK_DAYS

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
        df = get_price_history(symbol, use_cache=True)

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
        print_recent_signals(result, lookback_days=lookback_days)
        print_recent_trades(result, lookback_days=lookback_days, last_price=result.last_price)

        if lookback_days > DEFAULT_LOOKBACK_DAYS:
            save_detail_txt(symbol, lookback_days, result, last_price=result.last_price)

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
