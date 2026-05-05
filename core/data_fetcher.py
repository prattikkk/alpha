"""
core/data_fetcher.py
Fetches OHLCV candle data from Binance MAINNET (read-only).
No orders are placed here — this is pure market data.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
import requests

from config import CONFIG
from core.resilience import CircuitBreaker, TokenBucketLimiter, retry_delay_seconds
from utils.logger import get_logger

log = get_logger("DataFetcher")

MAINNET_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"

RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}

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
        self._volume_cache: dict[str, tuple[float, float]] = {}
        self._volume_ttl = 60  # seconds
        self._funding_cache: dict[str, tuple[float, float]] = {}
        self._funding_ttl = 60  # seconds
        self._exchange_info_cache: dict[str, tuple[float, dict]] = {}
        self._exchange_info_ttl = 300  # seconds

        api_cfg = CONFIG.api
        self._rate_limiter = TokenBucketLimiter(api_cfg.rate_limit_per_minute)
        self._circuit_breaker = CircuitBreaker(
            api_cfg.circuit_failures,
            api_cfg.circuit_cooldown_seconds,
        )
        self._retry_attempts = api_cfg.retry_attempts
        self._backoff_base = api_cfg.backoff_base_seconds
        self._backoff_cap = api_cfg.backoff_cap_seconds
        self._max_concurrent = max(1, int(api_cfg.max_concurrent_requests))

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
        return self.get_multi_tf_bulk([symbol]).get(symbol, {})

    def get_multi_tf_bulk(self, symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
        """Parallel fetch primary + HTFs for many symbols using asyncio + aiohttp."""
        cfg = CONFIG.strategy
        tfs = [cfg.primary_tf, cfg.htf_1, cfg.htf_2]
        result: dict[str, dict[str, pd.DataFrame]] = {symbol: {} for symbol in symbols}

        now = time.time()
        missing: list[tuple[str, str]] = []
        for symbol in symbols:
            for tf in tfs:
                cache_key = f"{symbol}_{tf}"
                cached = self._cache.get(cache_key)
                if cached and now - cached[0] < self._cache_ttl:
                    result[symbol][tf] = cached[1].copy()
                    continue
                missing.append((symbol, tf))

        if missing:
            fetched = self._fetch_missing_multi_tf(missing, CONFIG.trading.candle_limit)
            for (symbol, tf), df in fetched.items():
                if df is not None:
                    result[symbol][tf] = df

        return result

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Fast ticker price from mainnet."""
        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/ticker/price",
            params={"symbol": symbol},
            timeout=5,
            endpoint_key="market:ticker_price",
        )
        if payload is None:
            return None
        try:
            return float(payload["price"])
        except Exception as e:
            log.error(f"Price parse failed [{symbol}]: {e}")
            return None

    def get_24h_quote_volume(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """24h quote-volume in USDT terms (futures first, then spot fallback)."""
        if use_cache and symbol in self._volume_cache:
            ts, volume = self._volume_cache[symbol]
            if time.time() - ts < self._volume_ttl:
                return volume

        futures_volume = self._fetch_quote_volume(
            url=f"{MAINNET_BASE}/fapi/v1/ticker/24hr",
            symbol=symbol,
        )
        if futures_volume is not None:
            self._volume_cache[symbol] = (time.time(), futures_volume)
            return futures_volume

        spot_volume = self._fetch_quote_volume(
            url="https://api.binance.com/api/v3/ticker/24hr",
            symbol=symbol,
        )
        if spot_volume is not None:
            self._volume_cache[symbol] = (time.time(), spot_volume)
            return spot_volume

        return None

    def get_exchange_info(self, symbol: str) -> dict:
        """Get tick size, lot size, min notional for a symbol."""
        cached = self._exchange_info_cache.get(symbol)
        if cached and time.time() - cached[0] < self._exchange_info_ttl:
            return dict(cached[1])

        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/exchangeInfo",
            params={},
            timeout=10,
            endpoint_key="market:exchange_info",
        )
        if payload is None:
            return {}

        try:
            for s in payload.get("symbols", []):
                if s.get("symbol") != symbol:
                    continue

                info = {"symbol": symbol}
                for f in s.get("filters", []):
                    ft = f.get("filterType")
                    if ft == "PRICE_FILTER":
                        info["tick_size"] = float(f["tickSize"])
                    elif ft == "LOT_SIZE":
                        info["step_size"] = float(f["stepSize"])
                        info["min_qty"] = float(f["minQty"])
                    elif ft == "MIN_NOTIONAL":
                        info["min_notional"] = float(f.get("notional", 5))

                self._exchange_info_cache[symbol] = (time.time(), info)
                return info
        except Exception as e:
            log.error(f"Exchange info parse failed [{symbol}]: {e}")

        return {}

    def get_funding_rate(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """Fetch latest funding rate for the symbol."""
        if use_cache and symbol in self._funding_cache:
            ts, funding = self._funding_cache[symbol]
            if time.time() - ts < self._funding_ttl:
                return funding

        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=5,
            endpoint_key="market:funding_rate",
        )
        if payload is None:
            return None

        try:
            funding = float(payload.get("lastFundingRate", 0.0))
            self._funding_cache[symbol] = (time.time(), funding)
            return funding
        except Exception as e:
            log.debug("Funding rate parse error [%s]: %s", symbol, e)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_futures_candles(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
            endpoint_key="market:futures_klines",
        )
        if payload is None:
            return None
        try:
            return self._parse_candles(payload)
        except Exception as e:
            log.debug(f"Futures candle parse error [{symbol} {interval}]: {e}")
            return None

    def _fetch_spot_candles(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        payload = self._request_json_sync(
            url=f"{SPOT_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
            endpoint_key="market:spot_klines",
        )
        if payload is None:
            return None
        try:
            return self._parse_candles(payload)
        except Exception as e:
            log.debug(f"Spot candle parse error [{symbol} {interval}]: {e}")
            return None

    def _fetch_quote_volume(self, url: str, symbol: str) -> Optional[float]:
        payload = self._request_json_sync(
            url=url,
            params={"symbol": symbol},
            timeout=5,
            endpoint_key="market:24hr_ticker",
        )
        if payload is None:
            return None
        try:
            return float(payload.get("quoteVolume", 0.0))
        except Exception as e:
            log.debug(f"24h volume parse error [{symbol}] from {url}: {e}")
            return None

    def _fetch_missing_multi_tf(
        self,
        missing: list[tuple[str, str]],
        limit: int,
    ) -> dict[tuple[str, str], Optional[pd.DataFrame]]:
        try:
            asyncio.get_running_loop()
            return self._fetch_missing_multi_tf_sync(missing, limit)
        except RuntimeError:
            return asyncio.run(self._fetch_missing_multi_tf_async(missing, limit))

    def _fetch_missing_multi_tf_sync(
        self,
        missing: list[tuple[str, str]],
        limit: int,
    ) -> dict[tuple[str, str], Optional[pd.DataFrame]]:
        out: dict[tuple[str, str], Optional[pd.DataFrame]] = {}
        for symbol, tf in missing:
            out[(symbol, tf)] = self.get_ohlcv(symbol, tf, limit=limit, use_cache=True)
        return out

    async def _fetch_missing_multi_tf_async(
        self,
        missing: list[tuple[str, str]],
        limit: int,
    ) -> dict[tuple[str, str], Optional[pd.DataFrame]]:
        out: dict[tuple[str, str], Optional[pd.DataFrame]] = {}
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=self._max_concurrent)
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Content-Type": "application/json"},
        ) as session:
            tasks = [
                self._fetch_one_tf_async(session, semaphore, symbol, tf, limit)
                for symbol, tf in missing
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                log.debug(f"Async multi-tf fetch task failed: {result}")
                continue
            if result is None:
                continue
            key, df = result
            out[key] = df

        return out

    async def _fetch_one_tf_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        symbol: str,
        interval: str,
        limit: int,
    ) -> tuple[tuple[str, str], Optional[pd.DataFrame]]:
        cache_key = f"{symbol}_{interval}"
        async with semaphore:
            payload = await self._request_json_async(
                session=session,
                url=f"{MAINNET_BASE}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10,
                endpoint_key="market:futures_klines",
            )

            if payload is None:
                payload = await self._request_json_async(
                    session=session,
                    url=f"{SPOT_BASE}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=10,
                    endpoint_key="market:spot_klines",
                )

        if payload is None:
            return (symbol, interval), None

        try:
            df = self._parse_candles(payload)
        except Exception as e:
            log.debug(f"Async candle parse error [{symbol} {interval}]: {e}")
            return (symbol, interval), None

        if df.empty:
            return (symbol, interval), None

        df = self._add_derived_columns(df)
        self._cache[cache_key] = (time.time(), df)
        return (symbol, interval), df.copy()

    def _request_json_sync(
        self,
        url: str,
        params: dict,
        timeout: int,
        endpoint_key: str,
    ) -> Optional[dict | list]:
        if not self._circuit_breaker.allow(endpoint_key):
            log.warning("Circuit open for %s; skipping request", endpoint_key)
            return None

        for attempt in range(self._retry_attempts + 1):
            self._rate_limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except Exception as e:
                self._circuit_breaker.record_failure(endpoint_key)
                if attempt < self._retry_attempts:
                    time.sleep(retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap))
                    continue
                log.debug("Request failed [%s]: %s", endpoint_key, e)
                return None

            if resp.status_code in (200, 201):
                self._circuit_breaker.record_success(endpoint_key)
                try:
                    return resp.json()
                except Exception as e:
                    log.debug("JSON decode failed [%s]: %s", endpoint_key, e)
                    return None

            self._circuit_breaker.record_failure(endpoint_key)
            if resp.status_code in RETRYABLE_STATUS and attempt < self._retry_attempts:
                time.sleep(retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap))
                continue

            text = resp.text[:180].replace("\n", " ") if resp.text else ""
            log.debug("Request rejected [%s] %s: %s", endpoint_key, resp.status_code, text)
            return None

        return None

    async def _request_json_async(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict,
        timeout: int,
        endpoint_key: str,
    ) -> Optional[dict | list]:
        if not self._circuit_breaker.allow(endpoint_key):
            log.warning("Circuit open for %s; skipping async request", endpoint_key)
            return None

        for attempt in range(self._retry_attempts + 1):
            await self._rate_limiter.acquire_async()
            try:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    if resp.status in (200, 201):
                        self._circuit_breaker.record_success(endpoint_key)
                        try:
                            return await resp.json(content_type=None)
                        except Exception as e:
                            log.debug("Async JSON decode failed [%s]: %s", endpoint_key, e)
                            return None

                    self._circuit_breaker.record_failure(endpoint_key)
                    if resp.status in RETRYABLE_STATUS and attempt < self._retry_attempts:
                        await asyncio.sleep(
                            retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap)
                        )
                        continue

                    body = (await resp.text())[:180].replace("\n", " ")
                    log.debug("Async request rejected [%s] %s: %s", endpoint_key, resp.status, body)
                    return None
            except Exception as e:
                self._circuit_breaker.record_failure(endpoint_key)
                if attempt < self._retry_attempts:
                    await asyncio.sleep(
                        retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap)
                    )
                    continue
                log.debug("Async request failed [%s]: %s", endpoint_key, e)
                return None

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
