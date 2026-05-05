"""
strategies/ensemble.py
Ensemble strategy — aggregates signals from all strategies.

A trade is only taken when ≥ MIN_AGREEMENT strategies agree on direction.
The final confidence is the weighted average of agreeing signals.
"""
from __future__ import annotations
from typing import Optional

import pandas as pd

from core.regime import MarketRegime, detect_market_regime
from core.signal import Signal, Direction
from strategies.supertrend_rsi import SuperTrendRSIStrategy
from strategies.ema_adx_volume import EMAAdxVolumeStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from config import CONFIG
from utils.logger import get_logger

log = get_logger("Ensemble")

# Weights per strategy (must sum to 1.0)
WEIGHTS = {
    SuperTrendRSIStrategy.name:     0.45,
    EMAAdxVolumeStrategy.name:      0.35,
    BreakoutMomentumStrategy.name:  0.20,
}

REGIME_ALLOWLIST = {
    MarketRegime.TRENDING: {
        SuperTrendRSIStrategy.name,
        EMAAdxVolumeStrategy.name,
    },
    MarketRegime.RANGING: {
        BreakoutMomentumStrategy.name,
    },
    MarketRegime.HIGH_VOL: {
        BreakoutMomentumStrategy.name,
    },
    MarketRegime.UNKNOWN: set(WEIGHTS.keys()),
}


class EnsembleStrategy:
    name = "ensemble"

    def __init__(self):
        self._strategies = [
            SuperTrendRSIStrategy(),
            EMAAdxVolumeStrategy(),
            BreakoutMomentumStrategy(),
        ]

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        htf_df2: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        regime = detect_market_regime(
            df,
            adx_period=CONFIG.strategy.adx_period,
            trend_adx_threshold=CONFIG.strategy.regime_adx_trending,
            vol_window=CONFIG.strategy.regime_vol_window,
            high_vol_quantile=CONFIG.strategy.regime_high_vol_quantile,
        )
        allowed = REGIME_ALLOWLIST.get(regime, set(WEIGHTS.keys()))
        runnable = [s for s in self._strategies if s.name in allowed]
        if not runnable:
            return None

        signals: list[Signal] = []

        for strat in runnable:
            sig = strat.generate(symbol, df, htf_df, htf_df2)
            if sig is not None and sig.direction != Direction.FLAT:
                signals.append(sig)
                log.debug(f"  [{symbol}] {strat.name}: {sig.direction.value} conf={sig.confidence:.0%}")

        if not signals:
            return None

        # Count votes by direction
        long_sigs  = [s for s in signals if s.direction == Direction.LONG]
        short_sigs = [s for s in signals if s.direction == Direction.SHORT]

        min_agree = min(CONFIG.strategy.ensemble_min_signals, len(runnable))
        winning_sigs = None
        direction = Direction.FLAT

        if len(long_sigs) >= min_agree:
            winning_sigs = long_sigs
            direction = Direction.LONG
        elif len(short_sigs) >= min_agree:
            winning_sigs = short_sigs
            direction = Direction.SHORT

        if not winning_sigs:
            log.debug(f"[{symbol}] No ensemble agreement (L={len(long_sigs)} S={len(short_sigs)})")
            return None

        # Weighted confidence
        total_weight = sum(WEIGHTS.get(s.strategy, 0.33) for s in winning_sigs)
        weighted_conf = sum(
            s.confidence * WEIGHTS.get(s.strategy, 0.33)
            for s in winning_sigs
        ) / max(total_weight, 1e-9)

        disagreement_count = len(signals) - len(winning_sigs)
        disagreement_penalty = 1.0
        if disagreement_count > 0:
            disagreement_penalty = max(0.65, 1.0 - 0.12 * disagreement_count)
            weighted_conf *= disagreement_penalty

        # Use the highest-weighted signal's price levels as the reference
        ref = max(winning_sigs, key=lambda s: WEIGHTS.get(s.strategy, 0.33))

        reasons = " | ".join(
            f"{s.strategy}({s.confidence:.0%})" for s in winning_sigs
        )

        if disagreement_count > 0:
            reasons = (
                f"{reasons} | dissent={disagreement_count} "
                f"(penalty x{disagreement_penalty:.2f})"
            )

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=round(weighted_conf, 3),
            strategy=self.name,
            entry_price=ref.entry_price,
            stop_loss=ref.stop_loss,
            take_profit_1=ref.take_profit_1,
            take_profit_2=ref.take_profit_2,
            atr=ref.atr,
            reason=(
                f"Regime={regime.value} | Ensemble "
                f"[{len(winning_sigs)}/{len(runnable)}]: {reasons}"
            ),
            htf_bias=ref.htf_bias,
        )
