"""
bot.py — Main entry point
Telegram bot: market scanner, chart generator, signal sender, CSV logger.

Environment variables required:
    TG_TOKEN  — Telegram bot token
    TG_CHAT   — Telegram chat ID (int)

Optional:
    SCAN_INTERVAL_SEC — scan frequency in seconds (default: 300)
    MIN_PUMP_PCT      — override minimum pump % (default: 25)
    MIN_SCORE         — override minimum score (default: 3)
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
matplotlib.use("Agg")  # headless backend for server
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests

from data import get_futures_symbols, get_24h_change
from strategy import analyze_symbol, Signal, Config

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

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT = os.environ.get("TG_CHAT", "")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL_SEC", "300"))
SIGNALS_CSV = Path("signals.csv")

# Apply env overrides to strategy config
if os.environ.get("MIN_PUMP_PCT"):
    Config.MIN_PUMP_PCT = float(os.environ["MIN_PUMP_PCT"])
if os.environ.get("MIN_SCORE"):
    Config.MIN_SCORE = int(os.environ["MIN_SCORE"])

# Cooldown: prevent repeated signals for same symbol (seconds)
SIGNAL_COOLDOWN = 3600
_signal_cache: dict[str, float] = {}


# ─────────────────────────────────────────────
# TELEGRAM API
# ─────────────────────────────────────────────

def tg_send_message(text: str) -> bool:
    """Sends a text message to Telegram chat."""
    if not TG_TOKEN or not TG_CHAT:
        logger.warning("TG_TOKEN or TG_CHAT not set — skipping Telegram send")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "HTML",
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"tg_send_message error: {e}")
        return False


def tg_send_photo(image_bytes: bytes, caption: str = "") -> bool:
    """Sends a photo (bytes) to Telegram chat."""
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
        logger.error(f"tg_send_photo error: {e}")
        return False


# ─────────────────────────────────────────────
# CHART GENERATOR
# ─────────────────────────────────────────────

def generate_chart(signal: Signal) -> Optional[bytes]:
    """
    Generates a dark-themed price chart with:
    - Candlesticks (last 60 × 4H candles)
    - Entry, SL, TP1, TP2 horizontal lines
    - Resistance level
    - RSI subplot
    """
    try:
        df = signal.df_4h.iloc[-60:].copy()

        fig, (ax_price, ax_rsi) = plt.subplots(
            2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]},
            facecolor="#0d0d0d"
        )

        for ax in [ax_price, ax_rsi]:
            ax.set_facecolor("#0d0d0d")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a2a")

        # ── Candlesticks ────────────────────────────
        x = range(len(df))
        x_labels = df.index

        for i, (_, row) in enumerate(df.iterrows()):
            color = "#00e676" if row["close"] >= row["open"] else "#ff1744"
            ax_price.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
            height = abs(row["close"] - row["open"])
            bottom = min(row["open"], row["close"])
            rect = plt.Rectangle((i - 0.35, bottom), 0.7, height, color=color)
            ax_price.add_patch(rect)

        n = len(df)

        # ── Entry ──────────────────────────────────
        ax_price.axhline(signal.entry, color="#ffffff", linewidth=1.5, linestyle="--", label=f"Entry {signal.entry:.4f}")

        # ── Stop Loss ──────────────────────────────
        ax_price.axhline(signal.stop_loss, color="#ff1744", linewidth=1.5, linestyle="--", label=f"SL {signal.stop_loss:.4f}")

        # ── TP1 ────────────────────────────────────
        ax_price.axhline(signal.tp1, color="#69f0ae", linewidth=1.2, linestyle=":", label=f"TP1 {signal.tp1:.4f}")

        # ── TP2 ────────────────────────────────────
        ax_price.axhline(signal.tp2, color="#00e676", linewidth=1.5, linestyle=":", label=f"TP2 {signal.tp2:.4f}")

        # ── Resistance ─────────────────────────────
        if signal.resistance_4h:
            ax_price.axhline(
                signal.resistance_4h, color="#ff9800",
                linewidth=1.2, linestyle="-.", alpha=0.8,
                label=f"Res 4H {signal.resistance_4h:.4f}"
            )
        if signal.resistance_1d:
            ax_price.axhline(
                signal.resistance_1d, color="#ff5722",
                linewidth=1.2, linestyle="-.", alpha=0.8,
                label=f"Res 1D {signal.resistance_1d:.4f}"
            )

        # ── TP zone shading ────────────────────────
        ax_price.axhspan(signal.tp1, signal.tp2, alpha=0.05, color="#00e676")
        ax_price.axhspan(signal.entry, signal.stop_loss, alpha=0.05, color="#ff1744")

        # ── Labels ─────────────────────────────────
        step = max(1, n // 8)
        ax_price.set_xticks(list(x)[::step])
        ax_price.set_xticklabels(
            [x_labels[i].strftime("%m/%d %H:%M") for i in range(0, n, step)],
            rotation=25, fontsize=7, color="#888"
        )
        ax_price.tick_params(axis="y", colors="#888")
        ax_price.set_xlim(-1, n + 1)
        ax_price.set_title(
            f"⚡ SHORT SIGNAL — {signal.symbol}  |  Score: {signal.score}/7",
            color="white", fontsize=13, fontweight="bold", pad=12
        )
        ax_price.set_ylabel("Price (USDT)", color="#888")

        legend = ax_price.legend(
            facecolor="#1a1a1a", labelcolor="white",
            fontsize=8, loc="upper left", framealpha=0.8
        )

        # ── RSI subplot ────────────────────────────
        from indicators import calc_rsi
        rsi_values = [
            calc_rsi(df["close"].iloc[max(0, i - 14): i + 1])
            for i in range(len(df))
        ]
        ax_rsi.plot(x, rsi_values, color="#e040fb", linewidth=1.5)
        ax_rsi.axhline(70, color="#ff1744", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_rsi.axhline(30, color="#00e676", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_rsi.axhline(75, color="#ff9800", linewidth=0.8, linestyle=":", alpha=0.5)
        ax_rsi.fill_between(x, rsi_values, 70, where=[r > 70 for r in rsi_values], alpha=0.2, color="#ff1744")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_xlim(-1, n + 1)
        ax_rsi.set_ylabel("RSI", color="#888", fontsize=9)
        ax_rsi.tick_params(colors="#888", labelsize=7)
        ax_rsi.set_facecolor("#0d0d0d")
        ax_rsi.set_xticks([])

        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.error(f"generate_chart({signal.symbol}) error: {e}")
        return None


# ─────────────────────────────────────────────
# MESSAGE FORMATTER
# ─────────────────────────────────────────────

def format_signal_message(signal: Signal) -> str:
    """Formats the Telegram signal message."""
    oi_div_icon = "🔻" if signal.oi_divergence else "➖"
    sweep_icon = "🔫" if signal.liquidity_sweep else "➖"

    res_4h_str = f"{signal.resistance_4h:.4f}" if signal.resistance_4h else "—"
    res_1d_str = f"{signal.resistance_1d:.4f}" if signal.resistance_1d else "—"

    funding_pct = signal.funding_rate * 100

    return (
        f"⚡ <b>PUMP REVERSAL SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 <b>Coin:</b> #{signal.symbol}\n"
        f"📍 <b>Entry:</b> <code>{signal.entry:.4f}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>{signal.stop_loss:.4f}</code>\n"
        f"✅ <b>TP1:</b> <code>{signal.tp1:.4f}</code>\n"
        f"🎯 <b>TP2:</b> <code>{signal.tp2:.4f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Pump 24H:</b> +{signal.pump_percent:.1f}%\n"
        f"📊 <b>RSI (4H):</b> {signal.rsi:.1f}\n"
        f"💰 <b>Funding Rate:</b> {funding_pct:.4f}%\n"
        f"📦 <b>Open Interest:</b> ${signal.open_interest:,.0f}\n"
        f"🔀 <b>Volume Ratio:</b> {signal.volume_ratio:.1f}x\n"
        f"{oi_div_icon} <b>OI Divergence:</b> {'YES' if signal.oi_divergence else 'NO'}\n"
        f"{sweep_icon} <b>Liquidity Sweep:</b> {'YES' if signal.liquidity_sweep else 'NO'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏔 <b>Resistance 4H:</b> <code>{res_4h_str}</code>\n"
        f"🏔 <b>Resistance 1D:</b> <code>{res_1d_str}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⭐ <b>Signal Score:</b> {signal.score}/7\n"
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
    """Appends signal data to signals.csv for future backtesting."""
    file_exists = SIGNALS_CSV.exists()
    try:
        with open(SIGNALS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "date": datetime.utcnow().isoformat(),
                "symbol": signal.symbol,
                "entry": round(signal.entry, 6),
                "stop_loss": round(signal.stop_loss, 6),
                "tp1": round(signal.tp1, 6),
                "tp2": round(signal.tp2, 6),
                "rsi": round(signal.rsi, 2),
                "funding": round(signal.funding_rate, 6),
                "open_interest": round(signal.open_interest, 2),
                "pump_percent": round(signal.pump_percent, 2),
                "score": signal.score,
            })
    except Exception as e:
        logger.error(f"log_signal_to_csv error: {e}")


# ─────────────────────────────────────────────
# COOLDOWN CHECK
# ─────────────────────────────────────────────

def is_on_cooldown(symbol: str) -> bool:
    """Returns True if this symbol was recently signalled."""
    last_time = _signal_cache.get(symbol, 0)
    return (time.time() - last_time) < SIGNAL_COOLDOWN


def mark_signalled(symbol: str):
    _signal_cache[symbol] = time.time()


# ─────────────────────────────────────────────
# MARKET SCAN
# ─────────────────────────────────────────────

def scan_market():
    """
    Full market scan cycle:
    1. Get all symbols with min 24h volume
    2. Filter by 24h pump %
    3. Run full strategy analysis on pumped coins
    4. Send signals to Telegram
    """
    logger.info("━━━ Starting market scan ━━━")
    start_time = time.time()

    # Get candidate symbols (limit to top 250 by volume)
    symbols = get_futures_symbols(min_volume_usdt=5_000_000)[:250]
    logger.info(f"Scanning {len(symbols)} symbols...")

    pumped_symbols = []
    for symbol in symbols:
        pct = get_24h_change(symbol)
        if pct is not None and pct >= Config.MIN_PUMP_PCT:
            pumped_symbols.append((symbol, pct))
        time.sleep(0.05)

    logger.info(f"Pumped candidates (≥{Config.MIN_PUMP_PCT}%): {len(pumped_symbols)}")

    # Sort by pump intensity (strongest first)
    pumped_symbols.sort(key=lambda x: x[1], reverse=True)

    signals_sent = 0
    for symbol, pump_pct in pumped_symbols:
        if is_on_cooldown(symbol):
            logger.debug(f"{symbol}: on cooldown, skip")
            continue

        try:
            signal = analyze_symbol(symbol, pump_pct)
            if signal is None:
                continue

            logger.info(f"🚨 SIGNAL: {symbol} (score={signal.score})")

            # Generate chart
            chart_bytes = generate_chart(signal)

            # Format message
            msg = format_signal_message(signal)

            # Send to Telegram
            if chart_bytes:
                tg_send_photo(chart_bytes, caption=msg)
            else:
                tg_send_message(msg)

            # Log to CSV
            log_signal_to_csv(signal)

            # Mark cooldown
            mark_signalled(symbol)
            signals_sent += 1

            time.sleep(1)  # Telegram rate limit buffer

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
        finally:
            time.sleep(0.2)

    elapsed = time.time() - start_time
    logger.info(f"━━━ Scan complete: {signals_sent} signals in {elapsed:.1f}s ━━━")


# ─────────────────────────────────────────────
# STARTUP MESSAGE
# ─────────────────────────────────────────────

def send_startup_message():
    msg = (
        "🤖 <b>Pump Reversal Bot — STARTED</b>\n"
        f"Scan interval: {SCAN_INTERVAL}s\n"
        f"Min pump: {Config.MIN_PUMP_PCT}%\n"
        f"Min score: {Config.MIN_SCORE}/7\n"
        f"RSI threshold: {Config.RSI_OVERBOUGHT}\n"
        f"Symbols: up to 250\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Waiting for signals..."
    )
    tg_send_message(msg)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    logger.info("=" * 50)
    logger.info(" Pump Reversal Bot — Starting")
    logger.info(f" Scan interval: {SCAN_INTERVAL}s")
    logger.info(f" Min pump: {Config.MIN_PUMP_PCT}%")
    logger.info(f" Min score: {Config.MIN_SCORE}/7")
    logger.info("=" * 50)

    if not TG_TOKEN:
        logger.error("TG_TOKEN is not set. Set it in environment variables.")
        return

    if not TG_CHAT:
        logger.error("TG_CHAT is not set. Set it in environment variables.")
        return

    send_startup_message()

    while True:
        try:
            scan_market()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(30)  # brief pause before retry

        logger.info(f"Next scan in {SCAN_INTERVAL}s...")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
