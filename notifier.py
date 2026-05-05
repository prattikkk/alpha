"""
utils/notifier.py — Telegram trade alerts (optional)
"""
from __future__ import annotations
import os
import requests
from core.signal import Signal, Direction
from utils.logger import get_logger

log = get_logger("Notifier")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


def _send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        log.debug(f"Telegram send failed: {e}")


def notify_signal(signal: Signal):
    emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
    text = (
        f"{emoji} <b>AlphaBot Signal</b>\n"
        f"<b>{signal.symbol}</b> — {signal.direction.value}\n"
        f"Strategy: {signal.strategy}\n"
        f"Confidence: {signal.confidence:.0%}\n"
        f"Entry:  {signal.entry_price:.4f}\n"
        f"SL:     {signal.stop_loss:.4f}\n"
        f"TP1:    {signal.take_profit_1:.4f}\n"
        f"TP2:    {signal.take_profit_2:.4f}\n"
        f"RR:     {signal.risk_reward:.1f}x\n"
        f"Reason: {signal.reason}"
    )
    _send(text)


def notify_trade_open(symbol: str, direction: str, entry: float, qty: float, notional: float):
    emoji = "🟢" if direction == "LONG" else "🔴"
    _send(
        f"{emoji} <b>Trade Opened</b>\n"
        f"{symbol} {direction}\n"
        f"Entry: {entry:.4f} | Qty: {qty} | ${notional:.2f}"
    )


def notify_trade_close(symbol: str, pnl: float, reason: str, balance: float):
    emoji = "✅" if pnl > 0 else "❌"
    _send(
        f"{emoji} <b>Trade Closed</b> — {reason}\n"
        f"{symbol} | PnL: <b>${pnl:+.2f}</b>\n"
        f"Balance: ${balance:.2f}"
    )


def notify_stats(stats: dict):
    _send(
        f"📊 <b>AlphaBot Stats</b>\n"
        f"Trades: {stats.get('trades', 0)} | WR: {stats.get('win_rate', 0):.0%}\n"
        f"PnL: ${stats.get('total_pnl', 0):+.2f} | PF: {stats.get('profit_factor', 0):.2f}\n"
        f"Balance: ${stats.get('balance', 0):.2f} ({stats.get('return_pct', 0):+.1f}%)"
    )
