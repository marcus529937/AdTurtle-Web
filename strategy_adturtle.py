from dataclasses import dataclass, field
import pandas as pd


@dataclass
class TradeRecord:
    entry_date: str
    entry_price: float
    shares: int
    exit_date: str | None = None
    exit_price: float | None = None
    realized_pnl: float = 0.0
    return_pct: float = 0.0
    status: str = "OPEN"


@dataclass
class StrategyResult:
    symbol: str
    signal: str = "PENDING"
    latest_signal: str = "HOLD"
    last_price: float | None = None

    entry_price: float | None = None
    stop_loss: float | None = None
    upper_band: float | None = None
    lower_band: float | None = None

    initial_capital: float = 1_000_000.0
    shares_held: int = 0
    in_position: bool = False

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    equity: float = 0.0
    return_pct: float = 0.0

    total_trades: int = 0
    closed_trades: int = 0
    win_trades: int = 0
    win_rate: float = 0.0

    trade_log: list[TradeRecord] = field(default_factory=list)
    signal_rows: list[dict] = field(default_factory=list)
    error: str | None = None


def adturtle_simple(
    symbol: str,
    df: pd.DataFrame,
    upper_period: int = 20,
    lower_period: int = 10,
    lookback_days: int = 60
) -> StrategyResult:
    result = StrategyResult(symbol=symbol)

    try:
        if df is None or df.empty:
            result.signal = "NO_DATA"
            result.latest_signal = "NO_DATA"
            result.error = "DataFrame 為空"
            return result

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Close", "High", "Low", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            result.signal = "ERROR"
            result.latest_signal = "ERROR"
            result.error = f"缺少欄位: {missing}"
            return result

        raw_df = df.copy().dropna(subset=["Close", "High", "Low", "Volume"])

        min_required = max(upper_period, lower_period) + 1
        if len(raw_df) < min_required:
            result.signal = "NO_DATA"
            result.latest_signal = "NO_DATA"
            result.error = f"資料筆數不足（至少需要 {min_required} 筆）"
            return result

        buffer_days = upper_period + 5
        need_rows = max(lookback_days + buffer_days, min_required)
        work_df = raw_df.tail(need_rows).copy()

        work_df["upper_20"] = work_df["High"].rolling(upper_period).max()
        work_df["lower_10"] = work_df["Low"].rolling(lower_period).min()

        rows = work_df.reset_index()
        date_col = rows.columns[0]

        calc_start_idx = max(1, len(rows) - lookback_days)

        cash = result.initial_capital
        shares = 0
        in_position = False
        current_trade = None
        realized_pnl_total = 0.0
        signal_rows = []
        latest_signal = "HOLD"

        for i in range(calc_start_idx, len(rows)):
            today = rows.iloc[i]
            yesterday = rows.iloc[i - 1]

            today_date = str(today[date_col])[:10]
            today_close = float(today["Close"])
            today_volume = int(today["Volume"])

            prev_upper = yesterday["upper_20"]
            prev_lower = yesterday["lower_10"]

            if pd.isna(prev_upper) or pd.isna(prev_lower):
                continue

            if (not in_position) and (today_close > float(prev_upper)):
                buy_price = today_close
                buy_shares = int(cash // buy_price)

                if buy_shares > 0:
                    cost = buy_shares * buy_price
                    cash -= cost
                    shares = buy_shares
                    in_position = True
                    latest_signal = "BUY"

                    current_trade = TradeRecord(
                        entry_date=today_date,
                        entry_price=round(buy_price, 2),
                        shares=buy_shares,
                        status="OPEN"
                    )
                    result.trade_log.append(current_trade)

                    signal_rows.append({
                        "date": today_date,
                        "signal": "BUY",
                        "close": round(today_close, 2),
                        "channel_price": round(float(prev_upper), 2),
                        "volume": today_volume,
                        "shares": buy_shares,
                        "pnl": None,
                    })

            elif in_position and (today_close < float(prev_lower)):
                sell_price = today_close
                sell_shares = shares
                proceeds = sell_shares * sell_price

                trade_realized = (sell_price - current_trade.entry_price) * sell_shares
                realized_pnl_total += trade_realized
                cash += proceeds

                current_trade.exit_date = today_date
                current_trade.exit_price = round(sell_price, 2)
                current_trade.realized_pnl = round(trade_realized, 2)
                current_trade.return_pct = round(
                    ((sell_price - current_trade.entry_price) / current_trade.entry_price) * 100, 2
                )
                current_trade.status = "CLOSED"

                signal_rows.append({
                    "date": today_date,
                    "signal": "SELL",
                    "close": round(today_close, 2),
                    "channel_price": round(float(prev_lower), 2),
                    "volume": today_volume,
                    "shares": sell_shares,
                    "pnl": round(trade_realized, 2),
                })

                shares = 0
                in_position = False
                current_trade = None
                latest_signal = "SELL"

        last_row = work_df.iloc[-1]
        prev_row = work_df.iloc[-2]

        last_close = float(last_row["Close"])
        prev_upper = float(prev_row["upper_20"]) if pd.notna(prev_row["upper_20"]) else None
        prev_lower = float(prev_row["lower_10"]) if pd.notna(prev_row["lower_10"]) else None

        unrealized_pnl = 0.0
        current_entry_price = None

        if in_position and current_trade is not None:
            current_entry_price = current_trade.entry_price
            unrealized_pnl = (last_close - current_entry_price) * shares

        equity = cash + shares * last_close
        total_pnl = realized_pnl_total + unrealized_pnl
        cumulative_return_pct = (total_pnl / result.initial_capital) * 100 if result.initial_capital else 0.0

        closed_trades = [t for t in result.trade_log if t.status == "CLOSED"]
        win_trades = [t for t in closed_trades if t.realized_pnl > 0]
        win_rate = (len(win_trades) / len(closed_trades) * 100) if closed_trades else 0.0

        if latest_signal in ("BUY", "SELL"):
            result.signal = latest_signal
        else:
            result.signal = "HOLD"

        result.latest_signal = latest_signal
        result.last_price = round(last_close, 2)
        result.entry_price = round(current_entry_price, 2) if current_entry_price is not None else None
        result.upper_band = round(prev_upper, 2) if prev_upper is not None else None
        result.lower_band = round(prev_lower, 2) if prev_lower is not None else None
        result.stop_loss = round(prev_lower, 2) if prev_lower is not None else None

        result.shares_held = shares
        result.in_position = in_position

        result.realized_pnl = round(realized_pnl_total, 2)
        result.unrealized_pnl = round(unrealized_pnl, 2)
        result.total_pnl = round(total_pnl, 2)
        result.equity = round(equity, 2)
        result.return_pct = round(cumulative_return_pct, 2)

        result.closed_trades = len(closed_trades)
        result.total_trades = len(result.trade_log)
        result.win_trades = len(win_trades)
        result.win_rate = round(win_rate, 2)

        signal_rows.sort(key=lambda x: x["date"], reverse=True)
        result.signal_rows = signal_rows

    except Exception as e:
        result.signal = "ERROR"
        result.latest_signal = "ERROR"
        result.error = f"{type(e).__name__}: {e}"

    return result