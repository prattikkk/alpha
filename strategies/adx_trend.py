"""
strategies/adx_trend.py
ADX-confirmed EMA crossover trend strategy.

Best suited for 4h+ timeframes where strong trends can persist.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIG
from core.indicators import adx, atr, ema, rsi, volume_profile
from core.signal import Direction, Signal
from utils.logger import get_logger

log = get_logger("ADX_Trend")
cfg = CONFIG.strategy


class ADXTrendStrategy:
    """ADX-confirmed EMA crossover trend following strategy."""

    name = "adx_trend"

    DEFAULT_PARAMS = {
        "fast_ema": 5,
        "slow_ema": 13,
        "adx_period": 14,
        "adx_min": 22.0,
        "adx_rising_bars": 0,
        "vol_min": 0.7,
        "rsi_low": 25.0,
        "rsi_high": 75.0,
        "body_atr_min": 0.2,
        "sl_mult": 2.0,
        "tp1_mult": 2.0,
        "tp2_mult": 5.0,
        "htf_required": False,
    }

    def __init__(self, params: dict | None = None):
        self.last_skip_reason: str = ""
        config_defaults = {
            "fast_ema": int(getattr(cfg, "adx_trend_fast_ema", self.DEFAULT_PARAMS["fast_ema"])),
            "slow_ema": int(getattr(cfg, "adx_trend_slow_ema", self.DEFAULT_PARAMS["slow_ema"])),
            "adx_period": int(getattr(cfg, "adx_trend_adx_period", self.DEFAULT_PARAMS["adx_period"])),
            "adx_min": float(getattr(cfg, "adx_trend_adx_min", self.DEFAULT_PARAMS["adx_min"])),
            "adx_rising_bars": int(
                getattr(cfg, "adx_trend_adx_rising_bars", self.DEFAULT_PARAMS["adx_rising_bars"])
            ),
            "vol_min": float(getattr(cfg, "adx_trend_vol_min", self.DEFAULT_PARAMS["vol_min"])),
            "rsi_low": float(getattr(cfg, "adx_trend_rsi_low", self.DEFAULT_PARAMS["rsi_low"])),
            "rsi_high": float(getattr(cfg, "adx_trend_rsi_high", self.DEFAULT_PARAMS["rsi_high"])),
            "body_atr_min": float(
                getattr(cfg, "adx_trend_body_atr_min", self.DEFAULT_PARAMS["body_atr_min"])
            ),
            "sl_mult": float(getattr(cfg, "adx_trend_sl_mult", self.DEFAULT_PARAMS["sl_mult"])),
            "tp1_mult": float(getattr(cfg, "adx_trend_tp1_mult", self.DEFAULT_PARAMS["tp1_mult"])),
            "tp2_mult": float(getattr(cfg, "adx_trend_tp2_mult", self.DEFAULT_PARAMS["tp2_mult"])),
            "htf_required": bool(
                getattr(cfg, "adx_trend_htf_required", self.DEFAULT_PARAMS["htf_required"])
            ),
        }

        p = {**self.DEFAULT_PARAMS, **config_defaults}
        if params:
            p.update(params)
        self.params = p

    def _reject(self, reason: str) -> None:
        self.last_skip_reason = reason
        return None

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        htf_df2: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        self.last_skip_reason = ""
        if df is None or len(df) < 80:
            return self._reject("insufficient_primary_data")

        try:
            return self._compute(symbol, df, htf_df, htf_df2)
        except Exception as e:
            log.error(f"[{symbol}] ADX Trend error: {e}", exc_info=True)
            return self._reject("runtime_error")

    def _compute(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        htf_df2: Optional[pd.DataFrame],
    ) -> Optional[Signal]:
        p = self.params
        signal_idx = -2
        prev_idx = -3

        fast_ema = ema(df["close"], p["fast_ema"])
        slow_ema = ema(df["close"], p["slow_ema"])
        adx_line, di_pos, di_neg = adx(df, p["adx_period"])
        rsi_vals = rsi(df["close"], 14)
        atr_vals = atr(df, 14)
        rel_volume = volume_profile(df, 20)

        price = float(df["close"].iloc[signal_idx])
        open_price = float(df["open"].iloc[signal_idx])
        curr_atr = float(atr_vals.iloc[signal_idx])
        curr_adx = float(adx_line.iloc[signal_idx])
        curr_rsi = float(rsi_vals.iloc[signal_idx])
        curr_di_p = float(di_pos.iloc[signal_idx])
        curr_di_n = float(di_neg.iloc[signal_idx])
        curr_vol = float(rel_volume.iloc[signal_idx]) if not np.isnan(rel_volume.iloc[signal_idx]) else np.nan

        prev_fast = float(fast_ema.iloc[prev_idx])
        prev_slow = float(slow_ema.iloc[prev_idx])
        curr_fast = float(fast_ema.iloc[signal_idx])
        curr_slow = float(slow_ema.iloc[signal_idx])

        if not np.isfinite(curr_atr) or curr_atr <= 0:
            return self._reject("invalid_atr")
        if not np.isfinite(curr_adx) or not np.isfinite(curr_rsi):
            return self._reject("invalid_adx_or_rsi")

        if curr_adx < p["adx_min"]:
            return self._reject(f"adx_below_min({curr_adx:.1f}<{p['adx_min']:.1f})")

        rising_bars = max(0, int(p["adx_rising_bars"]))
        if rising_bars > 0:
            past_adx = adx_line.iloc[signal_idx - rising_bars]
            if np.isnan(past_adx) or curr_adx <= float(past_adx):
                return self._reject("adx_not_rising")

        if np.isnan(curr_vol):
            return self._reject("volume_unavailable")
        if curr_vol < p["vol_min"]:
            return self._reject(f"volume_below_min({curr_vol:.2f}<{p['vol_min']:.2f})")

        if curr_rsi > p["rsi_high"] or curr_rsi < p["rsi_low"]:
            return self._reject(
                f"rsi_out_of_range({curr_rsi:.1f} not in [{p['rsi_low']:.1f},{p['rsi_high']:.1f}])"
            )

        body = abs(price - open_price)
        if body < curr_atr * p["body_atr_min"]:
            return self._reject("body_too_small")

        bull_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bear_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        direction = Direction.FLAT
        confidence = 0.0

        if bull_cross and curr_di_p > curr_di_n:
            htf_aligned = self._htf_aligned(htf_df, htf_df2, bullish=True)
            if p["htf_required"] and not htf_aligned:
                return self._reject("htf_misaligned")

            direction = Direction.LONG
            confidence = 0.42

            if curr_adx > 35:
                confidence += 0.18
            elif curr_adx > 28:
                confidence += 0.12
            elif curr_adx > 22:
                confidence += 0.06

            if curr_vol > 1.5:
                confidence += 0.10
            elif curr_vol > 1.2:
                confidence += 0.05

            if 40 < curr_rsi < 65:
                confidence += 0.10

            if htf_aligned:
                confidence += 0.15

        elif bear_cross and curr_di_n > curr_di_p:
            htf_aligned = self._htf_aligned(htf_df, htf_df2, bullish=False)
            if p["htf_required"] and not htf_aligned:
                return self._reject("htf_misaligned")

            direction = Direction.SHORT
            confidence = 0.42

            if curr_adx > 35:
                confidence += 0.18
            elif curr_adx > 28:
                confidence += 0.12
            elif curr_adx > 22:
                confidence += 0.06

            if curr_vol > 1.5:
                confidence += 0.10
            elif curr_vol > 1.2:
                confidence += 0.05

            if 35 < curr_rsi < 60:
                confidence += 0.10

            if htf_aligned:
                confidence += 0.15

        if direction == Direction.FLAT:
            return self._reject("no_fresh_cross")

        confidence = max(0.0, min(confidence, 1.0))

        sl_dist = p["sl_mult"] * curr_atr
        tp1_dist = p["tp1_mult"] * curr_atr
        tp2_dist = p["tp2_mult"] * curr_atr

        if direction == Direction.LONG:
            stop_loss = price - sl_dist
            take_profit_1 = price + tp1_dist
            take_profit_2 = price + tp2_dist
        else:
            stop_loss = price + sl_dist
            take_profit_1 = price - tp1_dist
            take_profit_2 = price - tp2_dist

        reason = (
            f"ADX={curr_adx:.1f}(rising) | EMA({p['fast_ema']}/{p['slow_ema']}) cross | "
            f"DI+={curr_di_p:.1f} DI-={curr_di_n:.1f} | Vol={curr_vol:.2f}x | "
            f"RSI={curr_rsi:.1f} | body_atr={body / curr_atr:.2f}"
        )

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            strategy=self.name,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            atr=curr_atr,
            reason=reason,
            htf_bias=self._htf_bias(htf_df, htf_df2),
        )

    @staticmethod
    def _htf_aligned(htf_df, htf_df2, bullish: bool) -> bool:
        for frame in [htf_df, htf_df2]:
            if frame is None or len(frame) < 55:
                continue
            e9 = ema(frame["close"], 9).iloc[-2]
            e21 = ema(frame["close"], 21).iloc[-2]
            e50 = ema(frame["close"], 50).iloc[-2]

            if bullish and e9 > e21 > e50:
                return True
            if (not bullish) and e9 < e21 < e50:
                return True

        return False

    @staticmethod
    def _htf_bias(htf_df, htf_df2) -> Direction:
        scores = []
        for frame in [htf_df, htf_df2]:
            if frame is None or len(frame) < 55:
                continue
            e9 = ema(frame["close"], 9).iloc[-2]
            e21 = ema(frame["close"], 21).iloc[-2]
            e50 = ema(frame["close"], 50).iloc[-2]
            if e9 > e21 > e50:
                scores.append(1)
            elif e9 < e21 < e50:
                scores.append(-1)

        if not scores:
            return Direction.FLAT

        avg = sum(scores) / len(scores)
        if avg > 0.3:
            return Direction.LONG
        if avg < -0.3:
            return Direction.SHORT
        return Direction.FLAT
