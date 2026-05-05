"""Market regime classification utilities."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from core.indicators import adx, atr


class MarketRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    RANGING = "RANGING"
    TRENDING = "TRENDING"
    HIGH_VOL = "HIGH_VOL"


def detect_market_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    trend_adx_threshold: float = 23.0,
    vol_window: int = 80,
    high_vol_quantile: float = 0.8,
) -> MarketRegime:
    """Classify regime from closed candles using ADX and ATR-normalized volatility."""
    min_len = max(adx_period + 5, vol_window + 5)
    if df is None or len(df) < min_len:
        return MarketRegime.UNKNOWN

    adx_line, _, _ = adx(df, adx_period)
    atr_line = atr(df, 14)

    signal_idx = -2
    close = df["close"].replace(0, pd.NA)
    atr_norm = (atr_line / close).dropna()
    if atr_norm.empty:
        return MarketRegime.UNKNOWN

    try:
        curr_adx = float(adx_line.iloc[signal_idx])
        curr_atr_norm = float(atr_norm.iloc[signal_idx])
    except Exception:
        return MarketRegime.UNKNOWN

    if pd.isna(curr_adx) or pd.isna(curr_atr_norm):
        return MarketRegime.UNKNOWN

    vol_threshold = float(atr_norm.rolling(vol_window).quantile(high_vol_quantile).iloc[signal_idx])
    if pd.isna(vol_threshold):
        vol_threshold = float(atr_norm.quantile(high_vol_quantile))

    if curr_atr_norm >= vol_threshold:
        return MarketRegime.HIGH_VOL

    if curr_adx >= trend_adx_threshold:
        return MarketRegime.TRENDING

    return MarketRegime.RANGING
