import traceback
from flask import Flask, render_template, request, url_for
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

from top50_updater import ensure_top50_data
from market_data import get_price_history
from strategy_adturtle import adturtle_simple
from portfolio import aggregate_results

app = Flask(__name__)

LOOKBACK_DAYS = 60
RECENT_TABLE_DAYS = 30

top50_rows = []
strategy_results = []
portfolio_summary = {}


def normalize_top50_rows(raw_rows):
    if raw_rows is None:
        return []

    if isinstance(raw_rows, list):
        if len(raw_rows) == 0:
            return []
        if all(isinstance(x, dict) for x in raw_rows):
            return raw_rows
        print(f"[init_data] top50 格式異常：list 內不是 dict，內容範例={raw_rows[:3]}")
        return []

    if isinstance(raw_rows, dict):
        if "data" in raw_rows and isinstance(raw_rows["data"], list):
            return raw_rows["data"]
        if "rows" in raw_rows and isinstance(raw_rows["rows"], list):
            return raw_rows["rows"]
        if all(k in raw_rows for k in ["code", "name"]):
            return [raw_rows]

        print(f"[init_data] top50 dict 格式無法辨識，keys={list(raw_rows.keys())}")
        return []

    print(f"[init_data] top50 回傳型別異常: {type(raw_rows)}")
    return []


def normalize_signed_zero(val, digits=2):
    if val is None:
        return None
    try:
        v = round(float(val), digits)
        if abs(v) < (10 ** -digits):
            return 0.0
        return v + 0.0
    except Exception:
        return val


def parse_weight_pct(value):
    if value is None:
        return -1.0
    try:
        s = str(value).strip().replace("%", "").replace(",", "")
        return float(s)
    except Exception:
        return -1.0


def sort_rows(rows, sort_key="rank", direction="asc"):
    reverse = direction == "desc"

    def key_func(row):
        if sort_key == "rank":
            return row.get("rank", 999999)
        elif sort_key == "code":
            return str(row.get("code", ""))
        elif sort_key == "name":
            return str(row.get("name", ""))
        elif sort_key == "weight_pct":
            return parse_weight_pct(row.get("weight_pct"))
        elif sort_key == "last_price":
            return row.get("last_price") if row.get("last_price") is not None else -999999
        elif sort_key == "signal":
            return str(row.get("signal", ""))
        elif sort_key == "return_pct":
            return row.get("return_pct") if row.get("return_pct") is not None else -999999
        elif sort_key == "total_pnl":
            return row.get("total_pnl") if row.get("total_pnl") is not None else -999999
        return row.get("rank", 999999)

    return sorted(rows, key=key_func, reverse=reverse)


def prepare_price_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    work_df = df.copy()

    if isinstance(work_df.columns, pd.MultiIndex):
        work_df.columns = work_df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in work_df.columns]
    if missing:
        print(f"[prepare_price_df] 缺少欄位: {missing}")
        return None

    work_df = work_df.dropna(subset=required).copy()
    if work_df.empty:
        return None

    work_df["upper_20"] = work_df["High"].rolling(20).max()
    work_df["lower_10"] = work_df["Low"].rolling(10).min()
    return work_df


def init_data():
    global top50_rows, strategy_results, portfolio_summary

    raw_rows = ensure_top50_data()
    rows = normalize_top50_rows(raw_rows)
    results = []

    print(f"[init_data] top50 筆數: {len(rows)}")

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            print(f"[init_data] 第 {i} 筆不是 dict: {row}")
            continue

        symbol = row.get("code") or row.get("symbol") or row.get("ticker")
        if not symbol:
            row["signal"] = "ERROR"
            row["error"] = "缺少 code/symbol/ticker 欄位"
            row["last_price"] = None
            row["return_pct"] = None
            row["total_pnl"] = None
            row["equity"] = None
            continue

        row.setdefault("rank", i + 1)
        row.setdefault("name", "")
        row.setdefault("weight_pct", "")
        row.setdefault("signal", "PENDING")
        row.setdefault("last_price", None)
        row.setdefault("return_pct", None)
        row.setdefault("realized_pnl", None)
        row.setdefault("unrealized_pnl", None)
        row.setdefault("total_pnl", None)
        row.setdefault("equity", None)
        row.setdefault("error", None)

        try:
            df = get_price_history(symbol)

            if df is None or df.empty:
                row["signal"] = "NO_DATA"
                row["error"] = "無法從 yfinance 取得資料"
                continue

            result = adturtle_simple(symbol, df, lookback_days=LOOKBACK_DAYS)

            row["signal"] = result.latest_signal
            row["last_price"] = normalize_signed_zero(result.last_price, 2)
            row["return_pct"] = normalize_signed_zero(result.return_pct, 2)
            row["realized_pnl"] = int(normalize_signed_zero(result.realized_pnl, 0)) if result.realized_pnl is not None else 0
            row["unrealized_pnl"] = int(normalize_signed_zero(result.unrealized_pnl, 0)) if result.unrealized_pnl is not None else 0
            row["total_pnl"] = int(normalize_signed_zero(result.total_pnl, 0)) if result.total_pnl is not None else 0
            row["equity"] = normalize_signed_zero(result.equity, 2)
            row["error"] = result.error

            if result.signal not in ("NO_DATA", "ERROR"):
                results.append(result)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[init_data] ERROR for {symbol}: {err_msg}")
            print(traceback.format_exc())
            row["signal"] = "ERROR"
            row["error"] = err_msg
            row["last_price"] = None
            row["return_pct"] = None
            row["realized_pnl"] = None
            row["unrealized_pnl"] = None
            row["total_pnl"] = None
            row["equity"] = None

    top50_rows = rows
    strategy_results = results
    portfolio_summary = aggregate_results(results)
    print(f"[init_data] 完成：{len(rows)} 檔，成功 {len(results)} 檔")


def get_stock_name(symbol: str) -> str:
    for row in top50_rows:
        if str(row.get("code")) == str(symbol):
            return row.get("name") or symbol
    return symbol


def build_stock_chart_and_signals(symbol, stock_name, df, strategy_signal_rows=None, lookback_days=LOOKBACK_DAYS):
    full_df = prepare_price_df(df)
    if full_df is None or full_df.empty:
        return None, []

    chart_df = full_df.tail(lookback_days).copy()
    x_labels = chart_df.index.strftime("%Y-%m-%d")
    chart_dates = set(x_labels)

    strategy_signal_rows = strategy_signal_rows or []
    filtered_signal_rows = [
        row for row in strategy_signal_rows
        if str(row.get("date", "")) in chart_dates
    ]
    filtered_signal_rows.sort(key=lambda x: x["date"], reverse=True)

    buy_dates = {row["date"] for row in filtered_signal_rows if row.get("signal") == "BUY"}
    sell_dates = {row["date"] for row in filtered_signal_rows if row.get("signal") == "SELL"}

    chart_date_series = pd.Series(chart_df.index.strftime("%Y-%m-%d"), index=chart_df.index)
    buy_df = chart_df[chart_date_series.isin(buy_dates)]
    sell_df = chart_df[chart_date_series.isin(sell_dates)]

    start_date = x_labels[0] if len(x_labels) > 0 else "-"
    end_date = x_labels[-1] if len(x_labels) > 0 else "-"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            f"{stock_name} {symbol} - 最近 {lookback_days} 個交易日通道股價與信號（{start_date} ~ {end_date}）",
            "成交量"
        )
    )

    fig.add_trace(
        go.Candlestick(
            x=x_labels,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="K線",
            increasing_line_color="#c62828",
            increasing_fillcolor="#ef5350",
            decreasing_line_color="#2e7d32",
            decreasing_fillcolor="#66bb6a"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=chart_df["upper_20"],
            mode="lines",
            name="上通道(20)",
            line=dict(color="#607d8b", width=1.8, dash="dash")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=chart_df["lower_10"],
            mode="lines",
            name="下通道(10)",
            line=dict(color="#6b8f71", width=1.8, dash="dash")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=buy_df.index.strftime("%Y-%m-%d"),
            y=buy_df["Close"],
            mode="markers",
            name="BUY",
            marker=dict(
                symbol="triangle-up",
                size=13,
                color="#fb8c00",
                line=dict(width=1.2, color="#ef6c00")
            )
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sell_df.index.strftime("%Y-%m-%d"),
            y=sell_df["Close"],
            mode="markers",
            name="SELL",
            marker=dict(
                symbol="triangle-down",
                size=13,
                color="#1565c0",
                line=dict(width=1.2, color="#0d47a1")
            )
        ),
        row=1, col=1
    )

    vol_colors = [
        "#c62828" if c >= o else "#2e7d32"
        for o, c in zip(chart_df["Open"], chart_df["Close"])
    ]

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=chart_df["Volume"],
            name="成交量",
            marker_color=vol_colors
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_white",
        height=780,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(title_text="股價", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    chart_html = to_html(fig, full_html=False, include_plotlyjs="cdn")
    return chart_html, filtered_signal_rows


@app.route("/")
def index():
    sort_key = request.args.get("sort", "rank")
    direction = request.args.get("direction", "asc")

    rows = [dict(r) for r in top50_rows]
    rows = sort_rows(rows, sort_key, direction)

    def next_direction(col):
        if sort_key == col and direction == "asc":
            return "desc"
        return "asc"

    sort_urls = {
        "rank": url_for("index", sort="rank", direction=next_direction("rank")),
        "code": url_for("index", sort="code", direction=next_direction("code")),
        "name": url_for("index", sort="name", direction=next_direction("name")),
        "weight_pct": url_for("index", sort="weight_pct", direction=next_direction("weight_pct")),
        "last_price": url_for("index", sort="last_price", direction=next_direction("last_price")),
        "signal": url_for("index", sort="signal", direction=next_direction("signal")),
        "return_pct": url_for("index", sort="return_pct", direction=next_direction("return_pct")),
        "total_pnl": url_for("index", sort="total_pnl", direction=next_direction("total_pnl")),
    }

    return render_template(
        "index.html",
        rows=rows,
        summary=portfolio_summary,
        sort_key=sort_key,
        direction=direction,
        sort_urls=sort_urls,
        lookback_days=LOOKBACK_DAYS,
    )


@app.route("/stock/<symbol>")
def stock_detail(symbol):
    stock_name = get_stock_name(symbol)
    df = get_price_history(symbol)
    result = None
    recent_30 = []
    chart_html = None
    signal_rows = []

    if df is not None and not df.empty:
        clean_df = prepare_price_df(df)
        if clean_df is not None and not clean_df.empty:
            result = adturtle_simple(symbol, clean_df, lookback_days=LOOKBACK_DAYS)

            recent_30_df = clean_df.tail(RECENT_TABLE_DAYS).copy().reset_index()
            if "Date" in recent_30_df.columns:
                recent_30_df = recent_30_df.sort_values(by="Date", ascending=False)
            else:
                recent_30_df = recent_30_df.sort_index(ascending=False)
            recent_30 = recent_30_df.to_dict("records")

            raw_signal_rows = result.signal_rows if result and result.signal_rows else []
            chart_html, signal_rows = build_stock_chart_and_signals(
                symbol,
                stock_name,
                clean_df,
                strategy_signal_rows=raw_signal_rows,
                lookback_days=LOOKBACK_DAYS,
            )

    return render_template(
        "stock_detail.html",
        symbol=symbol,
        stock_name=stock_name,
        result=result,
        recent_30=recent_30,
        chart_html=chart_html,
        signal_rows=signal_rows,
        lookback_days=LOOKBACK_DAYS,
    )


if __name__ == "__main__":
    init_data()
    app.run(debug=False, host="0.0.0.0", port=5000)