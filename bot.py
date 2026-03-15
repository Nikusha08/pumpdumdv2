"""
bot.py — Main entry point
Telegram bot: scanner, chart generator, signal sender, CSV logger.

Environment variables:
    TG_TOKEN           — Telegram bot token
    TG_CHAT            — Telegram chat ID
    SCAN_INTERVAL_SEC  — scan frequency seconds (default: 300)
    MIN_PUMP_PCT       — override pump threshold (default: 25)
    MIN_SCORE          — override min score (default: 3)
"""

import csv
import io
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests

from data import get_futures_symbols, get_24h_change
from strategy import analyze_symbol, Signal, Config
from indicators import calc_rsi

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ],
)
logger = logging.getLogger("bot")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TG_TOKEN        = os.environ.get("TG_TOKEN", "")
TG_CHAT         = os.environ.get("TG_CHAT", "")
SCAN_INTERVAL   = int(os.environ.get("SCAN_INTERVAL_SEC", "300"))
SIGNALS_CSV     = Path("signals.csv")
SIGNAL_COOLDOWN = 3600  # seconds between same symbol signals

if os.environ.get("MIN_PUMP_PCT"):
    Config.MIN_PUMP_PCT = float(os.environ["MIN_PUMP_PCT"])
if os.environ.get("MIN_SCORE"):
    Config.MIN_SCORE = int(os.environ["MIN_SCORE"])

_signal_cache: dict[str, float] = {}


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def tg_send_message(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT:
        logger.warning("TG_TOKEN/TG_CHAT not set")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"tg_send_message: {e}")
        return False


def tg_send_photo(image_bytes: bytes, caption: str = "") -> bool:
    if not TG_TOKEN or not TG_CHAT:
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        resp = requests.post(
            url,
            data={"chat_id": TG_CHAT, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"tg_send_photo: {e}")
        return False


# ─────────────────────────────────────────────
# CHART GENERATOR
# ─────────────────────────────────────────────

BG      = "#0d0d0d"
GREEN   = "#00e676"
RED     = "#ff1744"
YELLOW  = "#ffd600"
ORANGE  = "#ff9800"
PURPLE  = "#e040fb"
CYAN    = "#00e5ff"
WHITE   = "#ffffff"
GREY    = "#555555"


def _price_fmt(price: float) -> str:
    """Format price nicely regardless of magnitude."""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    else:
        return f"{price:.8f}"


def generate_chart(signal: Signal) -> Optional[bytes]:
    """
    Generates a dark-themed chart with:
    - Candlestick bars (last 60 × 4H candles)
    - Entry / SL / TP1 / TP2 horizontal lines with labels
    - Resistance levels
    - RSI subplot
    - Volume bars
    """
    try:
        df = signal.df_4h.iloc[-60:].copy()
        n  = len(df)

        if n < 10:
            logger.warning(f"generate_chart: too few candles ({n})")
            return None

        # Validate all OHLCV values before drawing
        if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
            logger.warning("generate_chart: NaN in candle data")
            return None

        # ── Layout ─────────────────────────────────
        fig = plt.figure(figsize=(14, 10), facecolor=BG)
        gs  = fig.add_gridspec(3, 1, height_ratios=[3, 0.8, 0.8], hspace=0.08)

        ax_price  = fig.add_subplot(gs[0])
        ax_vol    = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi    = fig.add_subplot(gs[2], sharex=ax_price)

        for ax in [ax_price, ax_vol, ax_rsi]:
            ax.set_facecolor(BG)
            ax.tick_params(colors="#888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a2a")

        x = np.arange(n)

        # ── Candlesticks ───────────────────────────
        for i in range(n):
            row   = df.iloc[i]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = GREEN if c >= o else RED

            # Wick
            ax_price.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)

            # Body
            body_h = abs(c - o)
            body_y = min(o, c)
            if body_h < (h - l) * 0.01:  # doji — thin line
                body_h = (h - l) * 0.015
            rect = mpatches.Rectangle(
                (i - 0.38, body_y), 0.76, body_h,
                linewidth=0, facecolor=color, zorder=3
            )
            ax_price.add_patch(rect)

        # ── Horizontal trade levels ─────────────────
        levels = [
            (signal.stop_loss, RED,    "--", f"SL  {_price_fmt(signal.stop_loss)}"),
            (signal.entry,     WHITE,  "--", f"Entry  {_price_fmt(signal.entry)}"),
            (signal.tp1,       GREEN,  ":",  f"TP1  {_price_fmt(signal.tp1)}"),
            (signal.tp2,       "#00c853", ":", f"TP2  {_price_fmt(signal.tp2)}"),
        ]
        if signal.resistance_4h:
            levels.append((signal.resistance_4h, ORANGE, "-.", f"Res4H  {_price_fmt(signal.resistance_4h)}"))
        if signal.resistance_1d:
            levels.append((signal.resistance_1d, "#ff5722", "-.", f"Res1D  {_price_fmt(signal.resistance_1d)}"))

        for price_level, color, ls, label in levels:
            ax_price.axhline(price_level, color=color, linewidth=1.3,
                             linestyle=ls, alpha=0.9, zorder=4)
            ax_price.text(
                n + 0.3, price_level, label,
                color=color, fontsize=7.5, va="center",
                fontweight="bold"
            )

        # ── Zone shading ────────────────────────────
        ax_price.axhspan(signal.entry, signal.stop_loss, alpha=0.06, color=RED,   zorder=1)
        ax_price.axhspan(signal.tp2,   signal.entry,     alpha=0.04, color=GREEN, zorder=1)

        # ── Price axis limits ───────────────────────
        all_prices = list(df["high"]) + list(df["low"]) + [signal.stop_loss, signal.tp2]
        price_min  = min(all_prices) * 0.995
        price_max  = max(all_prices) * 1.005
        ax_price.set_ylim(price_min, price_max)
        ax_price.set_xlim(-1, n + 8)  # right margin for labels
        ax_price.set_ylabel("Price", color="#888", fontsize=9)
        ax_price.yaxis.set_label_position("left")
        ax_price.yaxis.tick_left()

        # ── Title ───────────────────────────────────
        ax_price.set_title(
            f"SHORT SIGNAL  ·  {signal.symbol}  ·  Score {signal.score}/7  ·  Pump +{signal.pump_percent:.1f}%",
            color=WHITE, fontsize=12, fontweight="bold", pad=10
        )

        # ── Volume bars ─────────────────────────────
        vol_avg = df["volume"].mean()
        for i in range(n):
            row   = df.iloc[i]
            color = GREEN if row["close"] >= row["open"] else RED
            alpha = 0.9 if row["volume"] > vol_avg * 2.5 else 0.5
            ax_vol.bar(i, row["volume"], color=color, alpha=alpha, width=0.76)

        ax_vol.axhline(vol_avg, color=GREY, linewidth=0.8, linestyle="--")
        ax_vol.set_ylabel("Vol", color="#888", fontsize=8)
        ax_vol.set_yticks([])

        # ── RSI line ────────────────────────────────
        rsi_values = []
        for i in range(n):
            window = df["close"].iloc[max(0, i - 14): i + 1]
            v = calc_rsi(window)
            rsi_values.append(v if not (np.isnan(v) or np.isinf(v)) else 50.0)

        ax_rsi.plot(x, rsi_values, color=PURPLE, linewidth=1.5, zorder=3)
        ax_rsi.axhline(75, color=RED,   linewidth=0.8, linestyle="--", alpha=0.6)
        ax_rsi.axhline(70, color=ORANGE, linewidth=0.6, linestyle=":",  alpha=0.5)
        ax_rsi.axhline(30, color=GREEN,  linewidth=0.6, linestyle=":",  alpha=0.5)
        ax_rsi.fill_between(x, rsi_values, 75,
                            where=[v > 75 for v in rsi_values],
                            alpha=0.2, color=RED, zorder=2)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", color="#888", fontsize=8)
        ax_rsi.text(n + 0.3, signal.rsi, f"{signal.rsi:.0f}",
                    color=PURPLE, fontsize=8, va="center", fontweight="bold")

        # ── X axis labels (shared) ──────────────────
        step      = max(1, n // 8)
        tick_pos  = list(range(0, n, step))
        tick_labs = [df.index[i].strftime("%m/%d\n%H:%M") for i in tick_pos]
        ax_rsi.set_xticks(tick_pos)
        ax_rsi.set_xticklabels(tick_labs, color="#888", fontsize=7)
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_vol.get_xticklabels(),   visible=False)

        # ── Render ──────────────────────────────────
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor=BG)
        plt.close()
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.error(f"generate_chart({signal.symbol}): {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────
# MESSAGE FORMATTER
# ─────────────────────────────────────────────

def format_signal_message(signal: Signal) -> str:
    res_4h = _price_fmt(signal.resistance_4h) if signal.resistance_4h else "—"
    res_1d = _price_fmt(signal.resistance_1d) if signal.resistance_1d else "—"

    return (
        f"⚡ <b>PUMP REVERSAL SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 <b>Coin:</b> #{signal.symbol}\n"
        f"📍 <b>Entry:</b> <code>{_price_fmt(signal.entry)}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>{_price_fmt(signal.stop_loss)}</code>\n"
        f"✅ <b>TP1:</b> <code>{_price_fmt(signal.tp1)}</code>\n"
        f"🎯 <b>TP2:</b> <code>{_price_fmt(signal.tp2)}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Pump 24H:</b> +{signal.pump_percent:.1f}%\n"
        f"📊 <b>RSI (4H):</b> {signal.rsi:.1f}\n"
        f"💰 <b>Funding:</b> {signal.funding_rate * 100:.4f}%\n"
        f"📦 <b>Open Interest:</b> ${signal.open_interest:,.0f}\n"
        f"🔀 <b>Volume:</b> {signal.volume_ratio:.1f}x avg\n"
        f"{'🔻' if signal.oi_divergence else '➖'} <b>OI Divergence:</b> {'YES' if signal.oi_divergence else 'NO'}\n"
        f"{'🔫' if signal.liquidity_sweep else '➖'} <b>Liq Sweep:</b> {'YES' if signal.liquidity_sweep else 'NO'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏔 <b>Res 4H:</b> <code>{res_4h}</code>\n"
        f"🏔 <b>Res 1D:</b> <code>{res_1d}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⭐ <b>Score:</b> {signal.score}/7\n"
        f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )


# ─────────────────────────────────────────────
# CSV LOGGER
# ─────────────────────────────────────────────

CSV_HEADERS = [
    "date", "symbol", "entry", "stop_loss", "tp1", "tp2",
    "rsi", "funding", "open_interest", "pump_percent", "score"
]


def log_signal_to_csv(signal: Signal):
    file_exists = SIGNALS_CSV.exists()
    try:
        with open(SIGNALS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                w.writeheader()
            w.writerow({
                "date":          datetime.utcnow().isoformat(),
                "symbol":        signal.symbol,
                "entry":         round(signal.entry, 8),
                "stop_loss":     round(signal.stop_loss, 8),
                "tp1":           round(signal.tp1, 8),
                "tp2":           round(signal.tp2, 8),
                "rsi":           round(signal.rsi, 2),
                "funding":       round(signal.funding_rate, 6),
                "open_interest": round(signal.open_interest, 2),
                "pump_percent":  round(signal.pump_percent, 2),
                "score":         signal.score,
            })
    except Exception as e:
        logger.error(f"log_signal_to_csv: {e}")


# ─────────────────────────────────────────────
# COOLDOWN
# ─────────────────────────────────────────────

def is_on_cooldown(symbol: str) -> bool:
    return (time.time() - _signal_cache.get(symbol, 0)) < SIGNAL_COOLDOWN


def mark_signalled(symbol: str):
    _signal_cache[symbol] = time.time()


# ─────────────────────────────────────────────
# MARKET SCAN
# ─────────────────────────────────────────────

def scan_market():
    logger.info("━━━ Starting scan ━━━")
    t0 = time.time()

    symbols = get_futures_symbols(min_volume_usdt=5_000_000)[:250]
    logger.info(f"Scanning {len(symbols)} symbols")

    # Fast pre-filter: collect pumped coins
    pumped = []
    for sym in symbols:
        pct = get_24h_change(sym)
        if pct is not None and pct >= Config.MIN_PUMP_PCT:
            pumped.append((sym, pct))
        time.sleep(0.04)

    pumped.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Pumped ≥{Config.MIN_PUMP_PCT}%: {len(pumped)}")

    sent = 0
    for symbol, pump_pct in pumped:
        if is_on_cooldown(symbol):
            continue
        try:
            signal = analyze_symbol(symbol, pump_pct)
            if signal is None:
                continue

            logger.info(f"🚨 {symbol} score={signal.score}/7")

            chart = generate_chart(signal)
            msg   = format_signal_message(signal)

            if chart:
                tg_send_photo(chart, caption=msg)
            else:
                tg_send_message(msg)

            log_signal_to_csv(signal)
            mark_signalled(symbol)
            sent += 1
            time.sleep(1)

        except Exception as e:
            logger.error(f"{symbol}: {e}", exc_info=True)
        finally:
            time.sleep(0.15)

    logger.info(f"━━━ Done: {sent} signals in {time.time()-t0:.1f}s ━━━")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not TG_TOKEN:
        logger.error("TG_TOKEN not set")
        return
    if not TG_CHAT:
        logger.error("TG_CHAT not set")
        return

    logger.info(f"Starting — interval={SCAN_INTERVAL}s, pump={Config.MIN_PUMP_PCT}%, score={Config.MIN_SCORE}/7")

    tg_send_message(
        f"🤖 <b>Pump Reversal Bot — STARTED</b>\n"
        f"Scan interval: {SCAN_INTERVAL}s\n"
        f"Min pump: {Config.MIN_PUMP_PCT}%\n"
        f"Min score: {Config.MIN_SCORE}/7\n"
        f"RSI threshold: {Config.RSI_OVERBOUGHT}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Waiting for signals..."
    )

    while True:
        try:
            scan_market()
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Main loop: {e}", exc_info=True)
            time.sleep(30)

        logger.info(f"Next scan in {SCAN_INTERVAL}s")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
