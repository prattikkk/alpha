"""
core/signal.py — Unified signal dataclass shared by all strategies
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"


@dataclass
class Signal:
    symbol: str
    direction: Direction
    confidence: float          # 0.0 – 1.0
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    atr: float
    reason: str = ""
    htf_bias: Direction = Direction.FLAT
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def risk_reward(self) -> float:
        if self.direction == Direction.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit_2 - self.entry_price
        elif self.direction == Direction.SHORT:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit_2
        else:
            return 0.0
        return reward / risk if risk > 0 else 0.0

    @property
    def is_valid(self) -> bool:
        from config import CONFIG
        return (
            self.direction != Direction.FLAT
            and self.confidence >= CONFIG.strategy.min_confidence
            and self.risk_reward >= CONFIG.risk.min_rr_ratio
            and self.stop_loss > 0
            and self.take_profit_1 > 0
            and self.take_profit_2 > 0
        )

    def __repr__(self) -> str:
        return (
            f"Signal({self.symbol} {self.direction.value} | "
            f"conf={self.confidence:.0%} | RR={self.risk_reward:.1f} | "
            f"SL={self.stop_loss:.4f} | TP2={self.take_profit_2:.4f})"
        )
