"""
strategies/supertrend_rsi.py
Primary trend-following strategy.

Entry logic:
  LONG  — SuperTrend bullish + RSI crosses above oversold from below
           + HTF (1h/4h) bias is bullish
  SHORT — SuperTrend bearish + RSI crosses below overbought from above
           + HTF bias is bearish

Exit:
  SL = 1.5× ATR below/above entry
  TP1 = 2.5× ATR (scale out 50%)
  TP2 = 4.0× ATR (full exit)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from core.signal import Signal, Direction
from core.indicators import atr, ema, pivot_points, rsi, supertrend
from config import CONFIG
from utils.logger import get_logger

log = get_logger("ST_RSI")
cfg = CONFIG.strategy
risk = CONFIG.risk


class SuperTrendRSIStrategy:
    name = "supertrend_rsi"

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        htf_df2: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        if df is None or len(df) < 50:
            return None

        try:
            return self._compute(symbol, df, htf_df, htf_df2)
        except Exception as e:
            log.error(f"[{symbol}] SuperTrend+RSI error: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------ #

    def _compute(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        htf_df2: Optional[pd.DataFrame],
    ) -> Optional[Signal]:
        # Evaluate on closed bars only (-2 signal bar, -3 previous bar).
        signal_idx = -2
        prev_idx = -3

        # --- Primary TF indicators ---
        st_line, st_dir = supertrend(df, cfg.supertrend_period, cfg.supertrend_multiplier)
        rsi_vals = rsi(df["close"], cfg.rsi_period)
        _atr = atr(df, 14)

        curr_dir = st_dir.iloc[signal_idx]
        prev_dir = st_dir.iloc[prev_idx]
        curr_rsi = rsi_vals.iloc[signal_idx]
        prev_rsi = rsi_vals.iloc[prev_idx]
        curr_price = df["close"].iloc[signal_idx]
        curr_atr = _atr.iloc[signal_idx]
        pivots = pivot_points(df.iloc[:-1]) if bool(cfg.use_support_resistance) else None

        if curr_atr == 0 or np.isnan(curr_atr):
            return None

        # --- HTF bias ---
        htf_bias = self._htf_bias(htf_df, htf_df2)

        # --- Signal conditions ---
        direction = Direction.FLAT
        entry_mode = ""

        # RSI confirmation
        rsi_bull = prev_rsi < cfg.rsi_oversold and curr_rsi >= cfg.rsi_oversold
        rsi_bear = prev_rsi > cfg.rsi_overbought and curr_rsi <= cfg.rsi_overbought
        rsi_mid_bull = 40 <= curr_rsi <= 65       # momentum zone
        rsi_mid_bear = 35 <= curr_rsi <= 60

        # HTF alignment bonus
        htf_bull = htf_bias == Direction.LONG
        htf_bear = htf_bias == Direction.SHORT

        long_flip = curr_dir == 1 and prev_dir == -1 and (rsi_bull or rsi_mid_bull)
        short_flip = curr_dir == -1 and prev_dir == 1 and (rsi_bear or rsi_mid_bear)
        long_cont = curr_dir == 1 and prev_dir == 1 and htf_bull and rsi_mid_bull
        short_cont = curr_dir == -1 and prev_dir == -1 and htf_bear and rsi_mid_bear

        # ----- LONG -----
        if long_flip or long_cont:
            direction = Direction.LONG
            confidence = 0.0
            if long_flip:
                confidence += 0.32   # fresh flip
                entry_mode = "flip"
            else:
                confidence += 0.22   # continuation with HTF confluence
                entry_mode = "continuation"

            if rsi_bull:
                confidence += 0.18
            elif curr_rsi <= 50:
                confidence += 0.12
            elif curr_rsi <= 60:
                confidence += 0.10
            else:
                confidence += 0.05

            if htf_bull:
                confidence += 0.20
            elif htf_bias == Direction.FLAT:
                confidence += 0.06

            if long_cont:
                confidence += 0.06

            # Taker buy ratio
            if "taker_ratio" in df.columns and df["taker_ratio"].iloc[signal_idx] > 0.55:
                confidence += 0.10
            if pivots:
                if curr_price >= pivots["PP"]:
                    confidence += 0.05
                if curr_price >= pivots["R1"]:
                    confidence += 0.03
            confidence = min(confidence, 1.0)

        # ----- SHORT -----
        elif short_flip or short_cont:
            direction = Direction.SHORT
            confidence = 0.0
            if short_flip:
                confidence += 0.32
                entry_mode = "flip"
            else:
                confidence += 0.22
                entry_mode = "continuation"

            if rsi_bear:
                confidence += 0.18
            elif curr_rsi >= 50:
                confidence += 0.12
            elif curr_rsi >= 40:
                confidence += 0.10
            else:
                confidence += 0.05

            if htf_bear:
                confidence += 0.20
            elif htf_bias == Direction.FLAT:
                confidence += 0.06

            if short_cont:
                confidence += 0.06

            if "taker_ratio" in df.columns and df["taker_ratio"].iloc[signal_idx] < 0.45:
                confidence += 0.10
            if pivots:
                if curr_price <= pivots["PP"]:
                    confidence += 0.05
                if curr_price <= pivots["S1"]:
                    confidence += 0.03
            confidence = min(confidence, 1.0)

        else:
            return None

        # --- SL / TP levels ---
        sl_dist = risk.atr_sl_multiplier * curr_atr
        tp1_dist = risk.atr_tp1_multiplier * curr_atr
        tp2_dist = risk.atr_tp2_multiplier * curr_atr

        if direction == Direction.LONG:
            sl  = curr_price - sl_dist
            tp1 = curr_price + tp1_dist
            tp2 = curr_price + tp2_dist
        else:
            sl  = curr_price + sl_dist
            tp1 = curr_price - tp1_dist
            tp2 = curr_price - tp2_dist

        reason = (
            f"ST={'↑' if curr_dir==1 else '↓'} flip={'yes' if curr_dir!=prev_dir else 'no'} | "
            f"mode={entry_mode or 'n/a'} | "
            f"RSI={curr_rsi:.1f} | HTF={htf_bias.value} | "
            f"SR={'on' if pivots else 'off'} | closed_bar=true"
        )

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            strategy=self.name,
            entry_price=curr_price,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            atr=curr_atr,
            reason=reason,
            htf_bias=htf_bias,
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _htf_bias(
        htf_df: Optional[pd.DataFrame],
        htf_df2: Optional[pd.DataFrame],
    ) -> Direction:
        """Determine higher-timeframe trend bias via EMA alignment."""
        scores = []
        for df in [htf_df, htf_df2]:
            if df is None or len(df) < 55:
                continue
            e9  = ema(df["close"], 9).iloc[-2]
            e21 = ema(df["close"], 21).iloc[-2]
            e50 = ema(df["close"], 50).iloc[-2]
            if e9 > e21 > e50:
                scores.append(1)
            elif e9 < e21 < e50:
                scores.append(-1)
            else:
                scores.append(0)

        if not scores:
            return Direction.FLAT
        avg = sum(scores) / len(scores)
        if avg > 0.3:
            return Direction.LONG
        elif avg < -0.3:
            return Direction.SHORT
        return Direction.FLAT
