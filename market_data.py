import os
import pickle
import contextlib
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "prices_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TW_TZ = ZoneInfo("Asia/Taipei")
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(13, 30)

RECENT_PERIODS: List[str] = ["3mo", "6mo", "1y", "2y", "5y"]
FULL_PERIOD = "max"


def _cache_path(cache_key: str) -> str:
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")


def _load_cache(cache_key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_key)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                return obj
        except Exception:
            return None
    return None


def _save_cache(cache_key: str, df: pd.DataFrame) -> None:
    try:
        with open(_cache_path(cache_key), "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        print(f"[market_data] Cache write error for {cache_key}: {e}")


def _cache_mtime(cache_key: str) -> Optional[datetime]:
    path = _cache_path(cache_key)
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


def _normalize_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    work_df = df.copy()

    if isinstance(work_df.columns, pd.MultiIndex):
        work_df.columns = work_df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in work_df.columns]
    if missing:
        return None

    work_df = work_df.dropna(subset=required).copy()
    if work_df.empty:
        return None

    if not isinstance(work_df.index, pd.DatetimeIndex):
        work_df.index = pd.to_datetime(work_df.index)

    if work_df.index.tz is not None:
        work_df.index = work_df.index.tz_localize(None)

    work_df = work_df.sort_index()
    return work_df


def _last_data_date(df: Optional[pd.DataFrame]) -> Optional[datetime]:
    if df is None or df.empty:
        return None
    try:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        ts = idx.max()
        if pd.isna(ts):
            return None
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(TW_TZ).tz_localize(None)
        return datetime.combine(ts.date(), time(0, 0), tzinfo=TW_TZ)
    except Exception:
        return None


def _is_data_fresh_enough(last_dt: Optional[datetime], now: datetime) -> bool:
    if last_dt is None:
        return False

    days_stale = (now.date() - last_dt.date()).days

    if days_stale < 0:
        return True

    phase = _market_phase(now)

    if phase == "preopen":
        return days_stale <= 1

    if phase == "open":
        return days_stale <= 1

    if phase == "closed":
        return days_stale == 0

    return False


def _should_use_cache(cache_key: str, use_cache: bool) -> bool:
    if not use_cache:
        return False

    cached = _load_cache(cache_key)
    if cached is None or cached.empty:
        return False

    now = _now_tw()
    mtime = _cache_mtime(cache_key)
    last_dt = _last_data_date(cached)

    if mtime is None or last_dt is None:
        return False

    if not _is_data_fresh_enough(last_dt, now):
        return False

    phase = _market_phase(now)

    if phase == "preopen":
        return True

    if phase == "open":
        age_seconds = (now - mtime).total_seconds()
        return age_seconds < 600

    if phase == "closed":
        return _is_same_day(now, mtime)

    return False


def _download_yf(symbol: str, period: str) -> Optional[pd.DataFrame]:
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                df = yf.download(
                    symbol,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    group_by="column",
                    multi_level_index=False,
                )
        return _normalize_df(df)
    except Exception as e:
        print(f"[market_data] Download exception for {symbol} period={period}: {e}")
        return None


def _pick_best_recent(symbol: str) -> Optional[pd.DataFrame]:
    best_df = None
    best_last = None

    for period in RECENT_PERIODS:
        df = _download_yf(symbol, period)
        if df is None or df.empty:
            continue

        last_dt = _last_data_date(df)
        if best_df is None:
            best_df = df
            best_last = last_dt
            continue

        if last_dt is not None and (best_last is None or last_dt > best_last):
            best_df = df
            best_last = last_dt

    return best_df


def _merge_full_and_recent(full_df: Optional[pd.DataFrame], recent_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if full_df is None or full_df.empty:
        return recent_df
    if recent_df is None or recent_df.empty:
        return full_df

    full_df = full_df.copy()
    recent_df = recent_df.copy()

    merged = pd.concat([full_df[~full_df.index.isin(recent_df.index)], recent_df])
    merged = merged.sort_index()

    merged = merged[~merged.index.duplicated(keep="last")]
    return _normalize_df(merged)


def get_price_history(ticker: str, period: str = "max", use_cache: bool = True) -> Optional[pd.DataFrame]:
    symbol = f"{ticker}.TW"
    cache_key = symbol

    if _should_use_cache(cache_key, use_cache):
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

    recent_df = _pick_best_recent(symbol)
    full_df = _download_yf(symbol, FULL_PERIOD)

    df = _merge_full_and_recent(full_df, recent_df)

    if df is None or df.empty:
        print(f"[market_data] No data returned for {symbol}")
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            print(f"[market_data] Fallback to cached data for {symbol}")
            return cached
        return None

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[market_data] Missing columns {missing} for {symbol}, got: {df.columns.tolist()}")
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            print(f"[market_data] Fallback to cached data for {symbol}")
            return cached
        return None

    if period != "max":
        try:
            if period.endswith("y"):
                years = int(period[:-1])
                start_dt = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
                df = df[df.index >= start_dt]
            elif period.endswith("mo"):
                months = int(period[:-2])
                start_dt = pd.Timestamp.now().normalize() - pd.DateOffset(months=months)
                df = df[df.index >= start_dt]
            elif period.endswith("d"):
                days = int(period[:-1])
                start_dt = pd.Timestamp.now().normalize() - pd.DateOffset(days=days)
                df = df[df.index >= start_dt]
        except Exception:
            pass

    if use_cache and df is not None and not df.empty:
        _save_cache(cache_key, df)

    return df


def get_recent_volume(df: pd.DataFrame, n: int = 5) -> list:
    if df is None or df.empty or "Volume" not in df.columns:
        return []
    return [int(v) for v in df["Volume"].iloc[-n:].tolist()]