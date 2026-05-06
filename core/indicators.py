"""
core/indicators.py — Pure-numpy/pandas technical indicators
No external TA library dependency for core indicators.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing: alpha = 1 / period.
    return tr.ewm(com=period - 1, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (supertrend_line, direction)
    direction: +1 = bullish (price above), -1 = bearish
    """
    if df.empty:
        empty_float = pd.Series(index=df.index, dtype=float)
        empty_int = pd.Series(index=df.index, dtype=int)
        return empty_float, empty_int

    _atr = atr(df, period).to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * _atr
    basic_lower = hl2 - multiplier * _atr

    n = len(df)
    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)
    st_line = np.full(n, np.nan, dtype=float)
    direction = np.full(n, 1, dtype=int)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    st_line[0] = final_lower[0]

    for i in range(1, n):
        prev_ub = final_upper[i - 1]
        prev_lb = final_lower[i - 1]
        prev_close = close[i - 1]

        curr_ub = basic_upper[i]
        curr_lb = basic_lower[i]

        if np.isnan(prev_ub):
            final_upper[i] = curr_ub
        elif np.isnan(curr_ub):
            final_upper[i] = prev_ub
        elif curr_ub < prev_ub or prev_close > prev_ub:
            final_upper[i] = curr_ub
        else:
            final_upper[i] = prev_ub

        if np.isnan(prev_lb):
            final_lower[i] = curr_lb
        elif np.isnan(curr_lb):
            final_lower[i] = prev_lb
        elif curr_lb > prev_lb or prev_close < prev_lb:
            final_lower[i] = curr_lb
        else:
            final_lower[i] = prev_lb

        prev_dir = direction[i - 1]
        if prev_dir == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1

        st_line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return (
        pd.Series(st_line, index=df.index, dtype=float),
        pd.Series(direction, index=df.index, dtype=int),
    )


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)"""
    h = df["high"]
    l = df["low"]
    c = df["close"]
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    up_move = h - prev_h
    down_move = prev_l - l

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    pos_dm = pd.Series(pos_dm, index=df.index)
    neg_dm = pd.Series(neg_dm, index=df.index)

    _atr = atr(df, period)
    # Wilder smoothing for directional movement.
    smooth_pos = pos_dm.ewm(com=period - 1, adjust=False).mean()
    smooth_neg = neg_dm.ewm(com=period - 1, adjust=False).mean()

    di_pos = 100 * smooth_pos / _atr.replace(0, np.nan)
    di_neg = 100 * smooth_neg / _atr.replace(0, np.nan)

    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    adx_line = dx.ewm(com=period - 1, adjust=False).mean()

    return adx_line, di_pos, di_neg


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest_low = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def volume_profile(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Relative volume vs MA."""
    vol_ma = df["volume"].rolling(period).mean()
    return df["volume"] / vol_ma.replace(0, np.nan)


def pivot_points(df: pd.DataFrame) -> dict[str, float]:
    """Classic pivot points from last completed candle."""
    last = df.iloc[-2]
    pp = (last["high"] + last["low"] + last["close"]) / 3
    r1 = 2 * pp - last["low"]
    s1 = 2 * pp - last["high"]
    r2 = pp + (last["high"] - last["low"])
    s2 = pp - (last["high"] - last["low"])
    r3 = last["high"] + 2 * (pp - last["low"])
    s3 = last["low"] - 2 * (last["high"] - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3,
            "S1": s1, "S2": s2, "S3": s3}
