from dataclasses import dataclass
from typing import List


@dataclass
class PortfolioSummary:
    stock_count: int = 0

    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0

    in_position_count: int = 0
    cash_position_count: int = 0

    total_initial_capital: float = 0.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_equity: float = 0.0
    return_pct: float = 0.0

    total_closed_trades: int = 0
    total_win_trades: int = 0
    total_win_rate: float = 0.0


def aggregate_results(results: List) -> PortfolioSummary:
    summary = PortfolioSummary()
    summary.stock_count = len(results)

    for r in results:
        # 訊號統計：用 latest_signal（BUY / SELL / HOLD）
        sig = getattr(r, "latest_signal", getattr(r, "signal", "HOLD"))
        if sig == "BUY":
            summary.buy_count += 1
        elif sig == "SELL":
            summary.sell_count += 1
        elif sig == "HOLD":
            summary.hold_count += 1

        if getattr(r, "in_position", False):
            summary.in_position_count += 1
        else:
            summary.cash_position_count += 1

        summary.total_initial_capital += getattr(r, "initial_capital", 0.0)
        summary.total_realized_pnl += getattr(r, "realized_pnl", 0.0)
        summary.total_unrealized_pnl += getattr(r, "unrealized_pnl", 0.0)
        summary.total_pnl += getattr(r, "total_pnl", 0.0)
        summary.total_equity += getattr(r, "equity", 0.0)

        summary.total_closed_trades += getattr(r, "closed_trades", 0)
        summary.total_win_trades += getattr(r, "win_trades", 0)

    if summary.total_initial_capital > 0:
        summary.return_pct = round(summary.total_pnl / summary.total_initial_capital * 100, 2)
    else:
        summary.return_pct = 0.0

    if summary.total_closed_trades > 0:
        summary.total_win_rate = round(summary.total_win_trades / summary.total_closed_trades * 100, 2)
    else:
        summary.total_win_rate = 0.0

    return summary