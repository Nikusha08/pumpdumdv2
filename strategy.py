"""
strategy.py — Signal generation logic
Combines all filters to produce SHORT reversal signals.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from data import (
    get_klines,
    get_funding_rate,
    is_oi_diverging,
    get_open_interest_history,
)
from indicators import (
    calc_rsi,
    calc_atr,
    is_volume_spike,
    get_volume_ratio,
    find_resistance_levels,
    nearest_resistance,
    price_near_resistance,
    detect_liquidity_sweep,
    calc_pump_percent,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

class Config:
    MIN_PUMP_PCT: float        = 25.0
    VOLUME_MULTIPLIER: float   = 3.0
    RSI_OVERBOUGHT: float      = 75.0
    RESISTANCE_TOLERANCE_PCT   = 3.0
    MIN_FUNDING_RATE: float    = 0.0001    # 0.01%
    MIN_SCORE: int             = 3
    ATR_PERIOD: int            = 14
    ATR_SL_MULT: float         = 2.0
    ATR_TP1_MULT: float        = 2.0
    ATR_TP2_MULT: float        = 4.0

    # Validation
    MIN_ATR_PCT: float         = 0.003     # ATR >= 0.3% of price
    MIN_CANDLES_FOR_RSI: int   = 20
    MIN_PRICE: float           = 0.000001


# ─────────────────────────────────────────────
# SIGNAL DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class Signal:
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    rsi: float
    funding_rate: float
    open_interest: float
    pump_percent: float
    resistance_4h: Optional[float]
    resistance_1d: Optional[float]
    volume_ratio: float
    oi_divergence: bool
    liquidity_sweep: bool
    score: int
    df_4h: pd.DataFrame = field(repr=False, default=None)


# ─────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame, symbol: str, min_len: int = 30) -> bool:
    if df is None or len(df) < min_len:
        logger.debug(f"{symbol}: not enough candles ({len(df) if df is not None else 0})")
        return False
    if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
        logger.debug(f"{symbol}: NaN in OHLCV")
        return False
    if (df["close"] <= 0).any():
        logger.debug(f"{symbol}: zero/negative prices")
        return False
    return True


def _validate_levels(entry: float, stop_loss: float, tp1: float, tp2: float, symbol: str) -> bool:
    if any(v <= 0 for v in [entry, stop_loss, tp1, tp2]):
        logger.debug(f"{symbol}: non-positive level")
        return False
    if stop_loss <= entry:
        logger.debug(f"{symbol}: SL <= Entry (invalid short)")
        return False
    if tp1 >= entry or tp2 >= entry:
        logger.debug(f"{symbol}: TP >= Entry (invalid short)")
        return False
    if tp2 >= tp1:
        logger.debug(f"{symbol}: TP2 >= TP1")
        return False
    if (stop_loss - entry) / entry < 0.003:
        logger.debug(f"{symbol}: SL distance < 0.3%")
        return False
    return True


# ─────────────────────────────────────────────
# MAIN SIGNAL GENERATOR
# ─────────────────────────────────────────────

def analyze_symbol(symbol: str, pump_pct: float) -> Optional[Signal]:

    # ── Quick pump check ───────────────────────────
    if pump_pct < Config.MIN_PUMP_PCT:
        return None

    # ── Fetch 4H candles ───────────────────────────
    df_4h = get_klines(symbol, "4h", limit=100)
    if not _validate_dataframe(df_4h, symbol, min_len=Config.MIN_CANDLES_FOR_RSI):
        return None

    current_price = float(df_4h["close"].iloc[-1])
    if current_price < Config.MIN_PRICE:
        return None

    # ── RSI — must be valid number ─────────────────
    rsi = calc_rsi(df_4h["close"])
    if math.isnan(rsi) or math.isinf(rsi):
        logger.debug(f"{symbol}: RSI invalid ({rsi})")
        return None

    # ── ATR — must be meaningful ───────────────────
    atr = calc_atr(df_4h, Config.ATR_PERIOD)
    if math.isnan(atr) or math.isinf(atr) or atr <= 0:
        logger.debug(f"{symbol}: ATR invalid ({atr})")
        return None
    if atr / current_price < Config.MIN_ATR_PCT:
        logger.debug(f"{symbol}: ATR too small ({atr/current_price:.4%})")
        return None

    # ── Trade levels — validate before continuing ──
    entry     = current_price
    stop_loss = entry + Config.ATR_SL_MULT * atr
    tp1       = entry - Config.ATR_TP1_MULT * atr
    tp2       = entry - Config.ATR_TP2_MULT * atr

    if not _validate_levels(entry, stop_loss, tp1, tp2, symbol):
        return None

    # ── Funding rate — must exist (= real futures) ─
    funding = get_funding_rate(symbol)
    if funding is None:
        logger.debug(f"{symbol}: no funding rate — not a perp futures, skip")
        return None

    # ── Open Interest — must be real data ──────────
    oi_df = get_open_interest_history(symbol, period="4h", limit=2)
    if oi_df is None or float(oi_df["sumOpenInterestValue"].iloc[-1]) <= 0:
        logger.debug(f"{symbol}: no OI data")
        return None
    oi_value = float(oi_df["sumOpenInterestValue"].iloc[-1])

    # ── SCORE FILTERS ──────────────────────────────
    score = 0
    reasons = []

    # 1. Pump ✅ (already passed)
    score += 1
    reasons.append(f"pump={pump_pct:.1f}%")

    # 2. Volume spike
    vol_ratio = get_volume_ratio(df_4h)
    if is_volume_spike(df_4h, Config.VOLUME_MULTIPLIER):
        score += 1
        reasons.append(f"vol={vol_ratio:.1f}x")

    # 3. RSI overbought
    if rsi >= Config.RSI_OVERBOUGHT:
        score += 1
        reasons.append(f"RSI={rsi:.1f}")

    # 4. Near resistance (4H or 1D)
    res_4h = nearest_resistance(current_price, find_resistance_levels(df_4h))
    near_4h = price_near_resistance(current_price, res_4h, Config.RESISTANCE_TOLERANCE_PCT)

    df_1d = get_klines(symbol, "1d", limit=60)
    res_1d = None
    near_1d = False
    if df_1d is not None and _validate_dataframe(df_1d, symbol, min_len=10):
        res_1d = nearest_resistance(current_price, find_resistance_levels(df_1d))
        near_1d = price_near_resistance(current_price, res_1d, Config.RESISTANCE_TOLERANCE_PCT)

    if near_4h or near_1d:
        score += 1
        reasons.append(f"res={'4H' if near_4h else '1D'}")

    # 5. Funding rate high & positive
    if funding >= Config.MIN_FUNDING_RATE:
        score += 1
        reasons.append(f"funding={funding:.4%}")

    # 6. OI divergence
    oi_div = is_oi_diverging(symbol)
    if oi_div:
        score += 1
        reasons.append("OI_div")

    # 7. Liquidity sweep
    liq_sweep = detect_liquidity_sweep(df_4h)
    if liq_sweep:
        score += 1
        reasons.append("sweep")

    # ── Minimum score ──────────────────────────────
    if score < Config.MIN_SCORE:
        logger.debug(f"{symbol}: score {score} < {Config.MIN_SCORE} — [{', '.join(reasons)}]")
        return None

    logger.info(f"✅ {symbol}: SIGNAL score={score}/7 [{', '.join(reasons)}]")

    return Signal(
        symbol=symbol,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        rsi=rsi,
        funding_rate=funding,
        open_interest=oi_value,
        pump_percent=pump_pct,
        resistance_4h=res_4h,
        resistance_1d=res_1d,
        volume_ratio=vol_ratio,
        oi_divergence=oi_div,
        liquidity_sweep=liq_sweep,
        score=score,
        df_4h=df_4h,
    )
