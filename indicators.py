"""
indicators.py — Technical indicators
RSI, ATR, resistance levels, volume spike, liquidity sweep
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────

def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    """
    Calculates RSI for the last candle.
    Returns value 0–100.
    """
    if len(closes) < period + 1:
        return 50.0

    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1])


# ─────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculates ATR (Average True Range).
    Requires columns: high, low, close.
    """
    if len(df) < period + 1:
        return float(df["close"].iloc[-1] * 0.02)  # fallback: 2%

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return float(atr.iloc[-1])


# ─────────────────────────────────────────────
# VOLUME SPIKE
# ─────────────────────────────────────────────

def is_volume_spike(df: pd.DataFrame, multiplier: float = 3.0, lookback: int = 20) -> bool:
    """
    Returns True if last candle volume is >= multiplier × avg volume.
    Indicates unusual buying/selling pressure.
    """
    if len(df) < lookback + 1:
        return False

    avg_vol = df["volume"].iloc[-(lookback + 1):-1].mean()
    last_vol = df["volume"].iloc[-1]

    spike = last_vol >= avg_vol * multiplier
    if spike:
        ratio = last_vol / avg_vol
        logger.debug(f"Volume spike detected: {ratio:.1f}x average")
    return spike


def get_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Returns ratio of last volume to average volume."""
    if len(df) < lookback + 1:
        return 1.0
    avg_vol = df["volume"].iloc[-(lookback + 1):-1].mean()
    last_vol = df["volume"].iloc[-1]
    return last_vol / avg_vol if avg_vol > 0 else 1.0


# ─────────────────────────────────────────────
# RESISTANCE LEVELS
# ─────────────────────────────────────────────

def find_resistance_levels(df: pd.DataFrame, lookback: int = 50) -> list[float]:
    """
    Find strong resistance levels using:
    1. Pivot highs (local maxima)
    2. Rolling max of last N candles
    3. Price clusters (repeated highs)

    Returns sorted list of resistance levels.
    """
    highs = df["high"].values[-lookback:]
    levels = set()

    # Method 1: Pivot highs (local maxima)
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            levels.add(round(highs[i], 4))

    # Method 2: Rolling max
    levels.add(round(float(np.max(highs)), 4))
    levels.add(round(float(np.max(highs[-25:])), 4))

    # Method 3: Cluster highs (round numbers / frequent touches)
    bins = np.histogram(highs, bins=15)[1]
    for b in bins:
        count = np.sum((highs >= b * 0.995) & (highs <= b * 1.005))
        if count >= 3:
            levels.add(round(float(b), 4))

    return sorted(levels)


def nearest_resistance(price: float, levels: list[float]) -> Optional[float]:
    """Returns the nearest resistance level above current price."""
    above = [l for l in levels if l >= price * 0.97]
    if not above:
        return None
    return min(above, key=lambda x: abs(x - price))


def price_near_resistance(price: float, resistance: float, tolerance_pct: float = 3.0) -> bool:
    """Returns True if price is within tolerance% of a resistance level."""
    if resistance is None or resistance == 0:
        return False
    pct_diff = abs(price - resistance) / resistance * 100
    return pct_diff <= tolerance_pct


# ─────────────────────────────────────────────
# LIQUIDITY SWEEP DETECTION
# ─────────────────────────────────────────────

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10) -> bool:
    """
    Detect liquidity sweep:
    - Price makes a new high (breaks above recent highs)
    - Then quickly reverses and closes below the breakout level

    This is a classic "fake breakout" / stop hunt pattern before a dump.
    Returns True if sweep detected on last candle.
    """
    if len(df) < lookback + 2:
        return False

    recent = df.iloc[-(lookback + 2):-1]
    last = df.iloc[-1]

    recent_high = recent["high"].max()

    # Last candle spiked above recent high (sweep)
    swept_high = last["high"] > recent_high

    # But closed back below — rejection
    closed_below = last["close"] < recent_high

    # Wick is significant (upper wick > 50% of candle range)
    candle_range = last["high"] - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    large_wick = (upper_wick / candle_range > 0.5) if candle_range > 0 else False

    sweep = swept_high and closed_below and large_wick
    if sweep:
        logger.debug(f"Liquidity sweep detected at {last['high']:.4f}")
    return sweep


# ─────────────────────────────────────────────
# PUMP DETECTION (price % change)
# ─────────────────────────────────────────────

def calc_pump_percent(df_4h: pd.DataFrame, candles_back: int = 6) -> float:
    """
    Calculates price change over the last N candles (default 6 × 4H = 24H).
    Returns percentage change.
    """
    if len(df_4h) < candles_back + 1:
        return 0.0
    price_start = df_4h["close"].iloc[-(candles_back + 1)]
    price_now = df_4h["close"].iloc[-1]
    return (price_now / price_start - 1) * 100
