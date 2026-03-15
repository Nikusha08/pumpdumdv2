"""
strategy.py — Signal generation logic
Combines all filters to produce SHORT reversal signals.
"""

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
    # Pump filter
    MIN_PUMP_PCT: float = 25.0          # minimum 24H pump to consider

    # Volume
    VOLUME_MULTIPLIER: float = 3.0      # last candle vs avg volume

    # RSI
    RSI_OVERBOUGHT: float = 75.0        # RSI threshold on 4H

    # Resistance
    RESISTANCE_TOLERANCE_PCT: float = 3.0  # price within X% of resistance

    # Funding rate
    MIN_FUNDING_RATE: float = 0.0005    # 0.05% — positive & high

    # Score system — minimum score to emit signal
    MIN_SCORE: int = 3                  # at least 3 filters must trigger

    # Trade levels
    ATR_PERIOD: int = 14
    ATR_SL_MULT: float = 2.0
    ATR_TP1_MULT: float = 2.0
    ATR_TP2_MULT: float = 4.0


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
# MAIN SIGNAL GENERATOR
# ─────────────────────────────────────────────

def analyze_symbol(symbol: str, pump_pct: float) -> Optional[Signal]:
    """
    Full analysis pipeline for one symbol.
    Returns Signal if all conditions are met, else None.
    """
    score = 0
    reasons = []

    # ── 1. PUMP FILTER ─────────────────────────────
    if pump_pct < Config.MIN_PUMP_PCT:
        return None  # Quick exit — main precondition
    score += 1
    reasons.append(f"pump={pump_pct:.1f}%")

    # ── Fetch 4H data ──────────────────────────────
    df_4h = get_klines(symbol, "4h", limit=100)
    if df_4h is None or len(df_4h) < 20:
        return None

    current_price = float(df_4h["close"].iloc[-1])

    # ── 2. VOLUME SPIKE ────────────────────────────
    vol_ratio = get_volume_ratio(df_4h)
    if is_volume_spike(df_4h, Config.VOLUME_MULTIPLIER):
        score += 1
        reasons.append(f"vol_spike={vol_ratio:.1f}x")

    # ── 3. RSI OVERBOUGHT ──────────────────────────
    rsi = calc_rsi(df_4h["close"])
    if rsi >= Config.RSI_OVERBOUGHT:
        score += 1
        reasons.append(f"RSI={rsi:.1f}")

    # ── 4. RESISTANCE LEVELS ───────────────────────
    res_4h = nearest_resistance(current_price, find_resistance_levels(df_4h))
    near_res_4h = price_near_resistance(current_price, res_4h, Config.RESISTANCE_TOLERANCE_PCT)

    df_1d = get_klines(symbol, "1d", limit=60)
    res_1d = None
    near_res_1d = False
    if df_1d is not None and len(df_1d) >= 10:
        res_1d = nearest_resistance(current_price, find_resistance_levels(df_1d))
        near_res_1d = price_near_resistance(current_price, res_1d, Config.RESISTANCE_TOLERANCE_PCT)

    if near_res_4h or near_res_1d:
        score += 1
        reasons.append(f"near_resistance={'4H' if near_res_4h else '1D'}")

    # ── 5. FUNDING RATE ────────────────────────────
    funding = get_funding_rate(symbol) or 0.0
    if funding >= Config.MIN_FUNDING_RATE:
        score += 1
        reasons.append(f"funding={funding:.4%}")

    # ── 6. OI DIVERGENCE ───────────────────────────
    oi_div = is_oi_diverging(symbol)
    if oi_div:
        score += 1
        reasons.append("OI_divergence")

    # ── 7. LIQUIDITY SWEEP ─────────────────────────
    liq_sweep = detect_liquidity_sweep(df_4h)
    if liq_sweep:
        score += 1
        reasons.append("liq_sweep")

    # ── SCORE CHECK ────────────────────────────────
    if score < Config.MIN_SCORE:
        logger.debug(f"{symbol}: score {score}/{Config.MIN_SCORE} — skip ({', '.join(reasons)})")
        return None

    logger.info(f"{symbol}: SIGNAL score={score} [{', '.join(reasons)}]")

    # ── TRADE LEVELS ───────────────────────────────
    atr = calc_atr(df_4h, Config.ATR_PERIOD)
    entry = current_price
    stop_loss = entry + Config.ATR_SL_MULT * atr
    tp1 = entry - Config.ATR_TP1_MULT * atr
    tp2 = entry - Config.ATR_TP2_MULT * atr

    # ── OPEN INTEREST SNAPSHOT ─────────────────────
    oi_df = get_open_interest_history(symbol, period="4h", limit=2)
    oi_value = float(oi_df["sumOpenInterestValue"].iloc[-1]) if oi_df is not None else 0.0

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
