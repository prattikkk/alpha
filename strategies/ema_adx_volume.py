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
from core.indicators import adx, atr, ema, pivot_points, rsi, volume_profile
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
        # Evaluate on closed bars only (-2 signal bar).
        signal_idx = -2

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
        curr = df["close"].iloc[signal_idx]
        e_f  = e_fast.iloc[signal_idx]
        e_s  = e_slow.iloc[signal_idx]
        e_t  = e_trend.iloc[signal_idx]
        adx_val = adx_line.iloc[signal_idx]
        dip  = di_pos.iloc[signal_idx]
        din  = di_neg.iloc[signal_idx]
        vol  = vol_ratio.iloc[signal_idx]
        curr_atr = _atr.iloc[signal_idx]
        curr_rsi = rsi_vals.iloc[signal_idx]
        pivots = pivot_points(df.iloc[:-1]) if bool(cfg.use_support_resistance) else None

        if np.isnan(curr_atr) or curr_atr == 0:
            return None
        if np.isnan(adx_val):
            return None

        prev_adx = adx_line.iloc[signal_idx - 1]
        adx_rising = not np.isnan(prev_adx) and adx_val > prev_adx
        trending_strong = adx_val >= cfg.adx_threshold
        trending_building = adx_val >= (cfg.adx_threshold * 0.80) and adx_rising

        # --- LONG conditions ---
        ema_bull = e_f > e_s > e_t                  # stacked bullish
        di_bull  = dip > din                         # directional bias
        vol_spike = vol >= cfg.volume_spike_ratio    # volume confirmation
        # Wider pullback tolerance avoids over-filtering during strong trends.
        pullback_pct = abs(curr - e_s) / curr
        pullback_bull = pullback_pct <= 0.025
        pullback_bull_loose = pullback_pct <= 0.040
        rsi_bull = 40 <= curr_rsi <= 68             # not overbought

        # --- SHORT conditions ---
        ema_bear = e_f < e_s < e_t
        di_bear  = din > dip
        pullback_bear = pullback_pct <= 0.025
        pullback_bear_loose = pullback_pct <= 0.040
        rsi_bear = 32 <= curr_rsi <= 60

        # ---- Determine direction ----
        direction = Direction.FLAT
        confidence = 0.0
        trend_state = "none"
        htf_b = self._htf_score(htf_df, htf_df2)

        if ema_bull and di_bull:
            if trending_strong:
                trend_state = "strong"
            elif trending_building:
                trend_state = "emerging"
            else:
                return None

            direction = Direction.LONG
            confidence += 0.34   # base conviction for EMA+DI alignment
            confidence += 0.18 if trend_state == "strong" else 0.10
            if vol_spike:
                confidence += 0.14
            elif vol >= 1.0:
                confidence += 0.06
            if pullback_bull:
                confidence += 0.12
            elif pullback_bull_loose:
                confidence += 0.05
            if rsi_bull:
                confidence += 0.08
            if htf_b > 0:
                confidence += 0.10
            elif htf_b == 0:
                confidence += 0.03
            if pivots:
                if curr >= pivots["PP"]:
                    confidence += 0.04
                if curr >= pivots["R1"]:
                    confidence += 0.02

        elif ema_bear and di_bear:
            if trending_strong:
                trend_state = "strong"
            elif trending_building:
                trend_state = "emerging"
            else:
                return None

            direction = Direction.SHORT
            confidence += 0.34
            confidence += 0.18 if trend_state == "strong" else 0.10
            if vol_spike:
                confidence += 0.14
            elif vol >= 1.0:
                confidence += 0.06
            if pullback_bear:
                confidence += 0.12
            elif pullback_bear_loose:
                confidence += 0.05
            if rsi_bear:
                confidence += 0.08
            if htf_b < 0:
                confidence += 0.10
            elif htf_b == 0:
                confidence += 0.03
            if pivots:
                if curr <= pivots["PP"]:
                    confidence += 0.04
                if curr <= pivots["S1"]:
                    confidence += 0.02

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
            f"EMA {'↑' if ema_bull else '↓'} | ADX={adx_val:.1f} ({trend_state}) | "
            f"DI+={dip:.1f} DI-={din:.1f} | Vol={vol:.1f}x | RSI={curr_rsi:.1f} | "
            f"pullback={pullback_pct*100:.2f}% | "
            f"SR={'on' if pivots else 'off'} | closed_bar=true"
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
            e9  = ema(df["close"], 9).iloc[-2]
            e21 = ema(df["close"], 21).iloc[-2]
            e50 = ema(df["close"], 50).iloc[-2]
            if e9 > e21 > e50:
                score += 1
            elif e9 < e21 < e50:
                score -= 1
        return score
