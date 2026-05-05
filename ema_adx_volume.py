"""
strategies/ema_adx_volume.py
Trend + momentum + volume confluence strategy.

Entry logic:
  LONG  — Fast EMA > Slow EMA > Trend EMA (stacked bullish)
           + ADX > threshold (trending, not ranging)
           + +DI > -DI
           + Volume spike (relative volume > 1.5×)
           + Price pulled back to Slow EMA (not chasing)

  SHORT — Inverse of above

This strategy excels in clearly trending markets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from core.signal import Signal, Direction
from core.indicators import ema, adx, atr, volume_profile, rsi
from config import CONFIG
from utils.logger import get_logger

log = get_logger("EMA_ADX")
cfg = CONFIG.strategy
risk = CONFIG.risk


class EMAAdxVolumeStrategy:
    name = "ema_adx_volume"

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        htf_df2: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        if df is None or len(df) < max(cfg.ema_trend + 5, 60):
            return None
        try:
            return self._compute(symbol, df, htf_df, htf_df2)
        except Exception as e:
            log.error(f"[{symbol}] EMA+ADX error: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------ #

    def _compute(self, symbol, df, htf_df, htf_df2):
        # EMAs
        e_fast  = ema(df["close"], cfg.ema_fast)
        e_slow  = ema(df["close"], cfg.ema_slow)
        e_trend = ema(df["close"], cfg.ema_trend)

        # ADX / DI
        adx_line, di_pos, di_neg = adx(df, cfg.adx_period)

        # Volume
        vol_ratio = volume_profile(df, cfg.volume_ma_period)

        # ATR
        _atr = atr(df, 14)

        # RSI filter
        rsi_vals = rsi(df["close"], cfg.rsi_period)

        # Latest values
        curr = df["close"].iloc[-1]
        prev = df["close"].iloc[-2]
        e_f  = e_fast.iloc[-1]
        e_s  = e_slow.iloc[-1]
        e_t  = e_trend.iloc[-1]
        adx_val = adx_line.iloc[-1]
        dip  = di_pos.iloc[-1]
        din  = di_neg.iloc[-1]
        vol  = vol_ratio.iloc[-1]
        curr_atr = _atr.iloc[-1]
        curr_rsi = rsi_vals.iloc[-1]

        if np.isnan(curr_atr) or curr_atr == 0:
            return None
        if np.isnan(adx_val):
            return None

        # Trend must be strong
        trending = adx_val > cfg.adx_threshold

        # --- LONG conditions ---
        ema_bull = e_f > e_s > e_t                  # stacked bullish
        di_bull  = dip > din                         # directional bias
        vol_spike = vol >= cfg.volume_spike_ratio    # volume confirmation
        # Pullback to slow EMA (entry near value, not chase)
        pullback_bull = abs(curr - e_s) / curr < 0.012  # within 1.2% of slow EMA
        rsi_bull = 40 < curr_rsi < 70               # not overbought

        # --- SHORT conditions ---
        ema_bear = e_f < e_s < e_t
        di_bear  = din > dip
        pullback_bear = abs(curr - e_s) / curr < 0.012
        rsi_bear = 30 < curr_rsi < 60

        # ---- Determine direction ----
        direction = Direction.FLAT
        confidence = 0.0

        if trending and ema_bull and di_bull:
            direction = Direction.LONG
            confidence += 0.30   # EMA stack
            confidence += 0.20   # ADX
            if vol_spike:
                confidence += 0.20
            if pullback_bull:
                confidence += 0.15
            if rsi_bull:
                confidence += 0.10
            # HTF bonus
            htf_b = self._htf_score(htf_df, htf_df2)
            if htf_b > 0:
                confidence += 0.10

        elif trending and ema_bear and di_bear:
            direction = Direction.SHORT
            confidence += 0.30
            confidence += 0.20
            if vol_spike:
                confidence += 0.20
            if pullback_bear:
                confidence += 0.15
            if rsi_bear:
                confidence += 0.10
            htf_b = self._htf_score(htf_df, htf_df2)
            if htf_b < 0:
                confidence += 0.10

        if direction == Direction.FLAT:
            return None

        confidence = min(confidence, 1.0)

        # --- SL / TP ---
        sl_dist  = risk.atr_sl_multiplier  * curr_atr
        tp1_dist = risk.atr_tp1_multiplier * curr_atr
        tp2_dist = risk.atr_tp2_multiplier * curr_atr

        if direction == Direction.LONG:
            sl  = curr - sl_dist
            tp1 = curr + tp1_dist
            tp2 = curr + tp2_dist
        else:
            sl  = curr + sl_dist
            tp1 = curr - tp1_dist
            tp2 = curr - tp2_dist

        reason = (
            f"EMA {'↑' if ema_bull else '↓'} | ADX={adx_val:.1f} | "
            f"DI+={dip:.1f} DI-={din:.1f} | Vol={vol:.1f}x | RSI={curr_rsi:.1f}"
        )

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            strategy=self.name,
            entry_price=curr,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            atr=curr_atr,
            reason=reason,
        )

    @staticmethod
    def _htf_score(htf_df, htf_df2) -> float:
        score = 0.0
        for df in [htf_df, htf_df2]:
            if df is None or len(df) < 55:
                continue
            e9  = ema(df["close"], 9).iloc[-1]
            e21 = ema(df["close"], 21).iloc[-1]
            e50 = ema(df["close"], 50).iloc[-1]
            if e9 > e21 > e50:
                score += 1
            elif e9 < e21 < e50:
                score -= 1
        return score
