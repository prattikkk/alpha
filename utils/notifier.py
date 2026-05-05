"""
utils/notifier.py — Telegram trade alerts (optional)
"""
from __future__ import annotations

import os
import time

import requests
from config import CONFIG
from core.resilience import TokenBucketLimiter, retry_delay_seconds
from core.signal import Signal, Direction
from utils.logger import get_logger

log = get_logger("Notifier")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
_NOTIFY_LIMITER = TokenBucketLimiter(CONFIG.api.notify_rate_limit_per_minute)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        return

    endpoint = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for attempt in range(CONFIG.api.retry_attempts + 1):
        _NOTIFY_LIMITER.acquire()
        try:
            resp = requests.post(
                endpoint,
                json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
        except Exception as e:
            if attempt < CONFIG.api.retry_attempts:
                time.sleep(
                    retry_delay_seconds(
                        attempt,
                        CONFIG.api.backoff_base_seconds,
                        CONFIG.api.backoff_cap_seconds,
                    )
                )
                continue
            log.debug(f"Telegram send failed: {e}")
            return

        if resp.status_code == 200:
            return

        if resp.status_code in _RETRYABLE_STATUS and attempt < CONFIG.api.retry_attempts:
            time.sleep(
                retry_delay_seconds(
                    attempt,
                    CONFIG.api.backoff_base_seconds,
                    CONFIG.api.backoff_cap_seconds,
                )
            )
            continue

        body = resp.text[:180].replace("\n", " ") if resp.text else ""
        log.debug("Telegram send rejected: %s %s", resp.status_code, body)
        return


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


def notify_event(title: str, message: str):
    _send(f"ℹ️ <b>{title}</b>\n{message}")
