"""
data.py — Binance API layer
Handles: klines, funding rate, open interest, market scanner
"""

import time
import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com"
SPOT_URL = "https://api.binance.com"

# ─────────────────────────────────────────────
# MARKET SCANNER
# ─────────────────────────────────────────────

def get_futures_symbols(min_volume_usdt: float = 10_000_000) -> list[str]:
    """
    Returns list of USDT-margined perpetual futures symbols
    filtered by minimum 24h volume.
    """
    try:
        url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        tickers = resp.json()

        symbols = [
            t["symbol"]
            for t in tickers
            if t["symbol"].endswith("USDT")
            and float(t.get("quoteVolume", 0)) >= min_volume_usdt
        ]
        logger.info(f"Found {len(symbols)} symbols with sufficient volume")
        return symbols

    except Exception as e:
        logger.error(f"get_futures_symbols error: {e}")
        return []


def get_24h_change(symbol: str) -> Optional[float]:
    """Returns 24h price change percent for a symbol."""
    try:
        url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        resp = requests.get(url, params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return float(data["priceChangePercent"])
    except Exception as e:
        logger.warning(f"get_24h_change({symbol}) error: {e}")
        return None


# ─────────────────────────────────────────────
# KLINES (OHLCV)
# ─────────────────────────────────────────────

def get_klines(symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV candlestick data from Binance Futures.

    Args:
        symbol:   e.g. 'BTCUSDT'
        interval: e.g. '4h', '1d', '15m'
        limit:    number of candles (max 1500)

    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume
    """
    try:
        url = f"{BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)

        return df[["open", "high", "low", "close", "volume", "quote_volume"]]

    except Exception as e:
        logger.warning(f"get_klines({symbol}, {interval}) error: {e}")
        return None


def get_historical_klines(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch full historical klines for backtesting using pagination.

    Args:
        start_str: e.g. '2024-01-01'
        end_str:   e.g. '2024-12-31' (None = now)
    """
    try:
        import time as t
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(t.time() * 1000)

        all_candles = []
        url = f"{BASE_URL}/fapi/v1/klines"

        current = start_ts
        while current < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ts,
                "limit": 1500,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            all_candles.extend(batch)
            current = batch[-1][0] + 1
            time.sleep(0.1)  # rate limit respect

        if not all_candles:
            return None

        df = pd.DataFrame(all_candles, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        return df[["open", "high", "low", "close", "volume"]]

    except Exception as e:
        logger.error(f"get_historical_klines({symbol}) error: {e}")
        return None


# ─────────────────────────────────────────────
# FUNDING RATE
# ─────────────────────────────────────────────

def get_funding_rate(symbol: str) -> Optional[float]:
    """
    Returns current funding rate for perpetual futures.
    Positive = longs paying shorts (bearish pressure).
    """
    try:
        url = f"{BASE_URL}/fapi/v1/premiumIndex"
        resp = requests.get(url, params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return float(data["lastFundingRate"])
    except Exception as e:
        logger.warning(f"get_funding_rate({symbol}) error: {e}")
        return None


# ─────────────────────────────────────────────
# OPEN INTEREST
# ─────────────────────────────────────────────

def get_open_interest_history(
    symbol: str,
    period: str = "4h",
    limit: int = 10
) -> Optional[pd.DataFrame]:
    """
    Returns open interest history.
    period: '5m','15m','30m','1h','2h','4h','6h','12h','1d'
    """
    try:
        url = f"{BASE_URL}/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": period, "limit": limit}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"])
        df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"])
        df.set_index("timestamp", inplace=True)
        return df

    except Exception as e:
        logger.warning(f"get_open_interest_history({symbol}) error: {e}")
        return None


def get_open_interest_snapshot(symbol: str) -> Optional[float]:
    """Returns current open interest in USDT."""
    try:
        url = f"{BASE_URL}/fapi/v1/openInterest"
        resp = requests.get(url, params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return float(data["openInterest"])
    except Exception as e:
        logger.warning(f"get_open_interest_snapshot({symbol}) error: {e}")
        return None


def is_oi_diverging(symbol: str) -> bool:
    """
    Detect OI divergence: price rising while OI falling.
    This is a bearish signal (exhaustion of buyers).
    Returns True if divergence detected.
    """
    try:
        oi_df = get_open_interest_history(symbol, period="1h", limit=6)
        klines = get_klines(symbol, "1h", limit=6)

        if oi_df is None or klines is None:
            return False

        price_change = klines["close"].iloc[-1] / klines["close"].iloc[0] - 1
        oi_change = oi_df["sumOpenInterest"].iloc[-1] / oi_df["sumOpenInterest"].iloc[0] - 1

        # Price up, OI down = exhaustion
        diverging = price_change > 0.01 and oi_change < -0.01
        if diverging:
            logger.info(f"{symbol}: OI divergence detected (price +{price_change:.2%}, OI {oi_change:.2%})")
        return diverging

    except Exception as e:
        logger.warning(f"is_oi_diverging({symbol}) error: {e}")
        return False
