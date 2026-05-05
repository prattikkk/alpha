"""Factory helpers for exchange-specific adapters."""

from __future__ import annotations

from config import CONFIG
from core.data_fetcher import DataFetcher
from core.executor import TestnetExecutor


def create_data_fetcher(exchange: str | None = None):
    target = (exchange or CONFIG.exchange.name or "binance").strip().lower()
    if target == "binance":
        return DataFetcher()
    raise ValueError(f"Unsupported exchange adapter: {target}")


def create_executor(exchange: str | None = None):
    target = (exchange or CONFIG.exchange.name or "binance").strip().lower()
    if target == "binance":
        return TestnetExecutor()
    raise ValueError(f"Unsupported exchange adapter: {target}")
