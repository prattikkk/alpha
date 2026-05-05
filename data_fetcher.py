"""
core/data_fetcher.py
Fetches OHLCV candle data from Binance MAINNET (read-only).
No orders are placed here — this is pure market data.
"""
from __future__ import annotations
import time
import requests
import pandas as pd
import numpy as np
from typing import Optional
from utils.logger import get_logger
from config import CONFIG

log = get_logger("DataFetcher")

MAINNET_BASE = "https://fapi.binance.com"

INTERVAL_MAP = {
    "1m":  60,    "3m":  180,   "5m":  300,
    "15m": 900,   "30m": 1800,  "1h":  3600,
    "2h":  7200,  "4h":  14400, "6h":  21600,
    "1d":  86400,
}


class DataFetcher:
    """
    Pulls OHLCV data from Binance Futures MAINNET.
    Uses the public REST endpoint — no API key required for candles.
    Falls back to spot endpoint if futures fails.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 30  # seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 200,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with columns:
        open_time, open, high, low, close, volume,
        close_time, quote_volume, trades, taker_buy_base, taker_buy_quote
        """
        cache_key = f"{symbol}_{interval}"
        if use_cache and cache_key in self._cache:
            ts, df = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return df.copy()

        df = self._fetch_futures_candles(symbol, interval, limit)
        if df is None:
            log.warning(f"Futures candles failed for {symbol}, trying spot…")
            df = self._fetch_spot_candles(symbol, interval, limit)

        if df is not None and not df.empty:
            df = self._add_derived_columns(df)
            self._cache[cache_key] = (time.time(), df)
            log.debug(f"Fetched {len(df)} candles [{symbol} {interval}]")

        return df

    def get_multi_tf(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Fetch primary + higher timeframes for confluence."""
        cfg = CONFIG.strategy
        result = {}
        for tf in [cfg.primary_tf, cfg.htf_1, cfg.htf_2]:
            df = self.get_ohlcv(symbol, tf, limit=CONFIG.trading.candle_limit)
            if df is not None:
                result[tf] = df
        return result

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Fast ticker price from mainnet."""
        try:
            r = self.session.get(
                f"{MAINNET_BASE}/fapi/v1/ticker/price",
                params={"symbol": symbol},
                timeout=5,
            )
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception as e:
            log.error(f"Price fetch failed [{symbol}]: {e}")
            return None

    def get_exchange_info(self, symbol: str) -> dict:
        """Get tick size, lot size, min notional for a symbol."""
        try:
            r = self.session.get(f"{MAINNET_BASE}/fapi/v1/exchangeInfo", timeout=10)
            r.raise_for_status()
            for s in r.json().get("symbols", []):
                if s["symbol"] == symbol:
                    info = {"symbol": symbol}
                    for f in s.get("filters", []):
                        ft = f["filterType"]
                        if ft == "PRICE_FILTER":
                            info["tick_size"] = float(f["tickSize"])
                        elif ft == "LOT_SIZE":
                            info["step_size"] = float(f["stepSize"])
                            info["min_qty"] = float(f["minQty"])
                        elif ft == "MIN_NOTIONAL":
                            info["min_notional"] = float(f.get("notional", 5))
                    return info
        except Exception as e:
            log.error(f"Exchange info failed [{symbol}]: {e}")
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_futures_candles(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        try:
            r = self.session.get(
                f"{MAINNET_BASE}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10,
            )
            r.raise_for_status()
            return self._parse_candles(r.json())
        except Exception as e:
            log.debug(f"Futures candle error [{symbol}]: {e}")
            return None

    def _fetch_spot_candles(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        try:
            r = self.session.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10,
            )
            r.raise_for_status()
            return self._parse_candles(r.json())
        except Exception as e:
            log.debug(f"Spot candle error [{symbol}]: {e}")
            return None

    @staticmethod
    def _parse_candles(raw: list) -> pd.DataFrame:
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "_ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df.drop(columns=["_ignore"], inplace=True)

        for c in ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[c] = df[c].astype(float)
        df["trades"] = df["trades"].astype(int)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df

    @staticmethod
    def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add taker buy ratio — useful for order flow analysis."""
        df = df.copy()
        df["taker_ratio"] = np.where(
            df["volume"] > 0,
            df["taker_buy_base"] / df["volume"],
            0.5,
        )
        df["body"] = abs(df["close"] - df["open"])
        df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]
        return df
