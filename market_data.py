import os
import pickle
import contextlib
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "prices_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TW_TZ = ZoneInfo("Asia/Taipei")
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(13, 30)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.pkl")


def _load_cache(ticker: str):
    path = _cache_path(ticker)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_cache(ticker: str, df: pd.DataFrame):
    try:
        with open(_cache_path(ticker), "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        print(f"[market_data] Cache write error for {ticker}: {e}")


def _cache_mtime(ticker: str):
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path), tz=TW_TZ)


def _now_tw() -> datetime:
    return datetime.now(TW_TZ)


def _is_same_day(dt1: datetime, dt2: datetime) -> bool:
    return dt1.date() == dt2.date()


def _market_phase(now: datetime) -> str:
    t = now.time()
    if t < MARKET_OPEN:
        return "preopen"
    elif MARKET_OPEN <= t <= MARKET_CLOSE:
        return "open"
    return "closed"


def _should_use_cache(symbol: str, use_cache: bool) -> bool:
    if not use_cache:
        return False

    cached = _load_cache(symbol)
    if cached is None or cached.empty:
        return False

    now = _now_tw()
    mtime = _cache_mtime(symbol)
    if mtime is None:
        return False

    phase = _market_phase(now)

    # 開盤前：直接用今天或之前的快取
    if phase == "preopen":
        return True

    # 盤中：如果 10 分鐘內抓過就直接用快取，避免過度重抓
    if phase == "open":
        age_seconds = (now - mtime).total_seconds()
        return age_seconds < 600

    # 收盤後：如果今天收盤後已抓過一次，就不要再抓第二次
    if phase == "closed":
        if _is_same_day(now, mtime) and mtime.time() >= MARKET_CLOSE:
            return True
        return False

    return False


def get_price_history(ticker: str, period: str = "6mo", use_cache: bool = True):
    symbol = f"{ticker}.TW"

    if _should_use_cache(symbol, use_cache):
        cached = _load_cache(symbol)
        if cached is not None and not cached.empty:
            return cached

    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                df = yf.download(
                    symbol,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    actions=False,
                    threads=False,
                )
    except Exception as e:
        print(f"[market_data] Download exception for {symbol}: {e}")
        return None

    if df is None or df.empty:
        print(f"[market_data] No data returned for {symbol}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[market_data] Missing columns {missing} for {symbol}, got: {df.columns.tolist()}")
        return None

    df = df.copy().sort_index()

    if use_cache:
        _save_cache(symbol, df)

    return df


def get_recent_volume(df: pd.DataFrame, n: int = 5) -> list:
    if df is None or df.empty or "Volume" not in df.columns:
        return []
    return [int(v) for v in df["Volume"].iloc[-n:].tolist()]