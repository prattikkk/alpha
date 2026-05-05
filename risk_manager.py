"""
core/risk_manager.py
Kelly-fraction position sizing with hard portfolio risk limits.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from core.signal import Signal, Direction
from config import CONFIG
from utils.logger import get_logger

log = get_logger("RiskMgr")
risk_cfg = CONFIG.risk


@dataclass
class PositionSize:
    symbol: str
    direction: Direction
    entry_price: float
    quantity: float          # in base asset (BTC, ETH…)
    notional_usdt: float     # entry value in USDT
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_usdt: float         # max $ loss if SL hit
    leverage: int


class RiskManager:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def size_position(
        self,
        signal: Signal,
        exchange_info: dict,
    ) -> Optional[PositionSize]:
        """
        Calculate position size using fixed fractional risk:
            risk_amount = capital × risk_per_trade
            quantity = risk_amount / (entry - stop_loss) [adjusted for leverage]
        """
        capital = self.portfolio.available_capital()
        if capital <= 0:
            log.warning("No available capital")
            return None

        # Hard portfolio risk check
        if not self._portfolio_risk_ok():
            log.warning("Max portfolio risk reached — no new positions")
            return None

        # Open positions limit
        if len(self.portfolio.open_positions) >= risk_cfg.max_open_positions:
            log.warning("Max open positions reached")
            return None

        entry = signal.entry_price
        sl    = signal.stop_loss

        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            log.error(f"Invalid SL for {signal.symbol}")
            return None

        risk_amount = capital * risk_cfg.max_risk_per_trade

        # Raw quantity
        qty = (risk_amount * risk_cfg.leverage) / risk_per_unit

        # Adjust to exchange lot size
        step = exchange_info.get("step_size", 0.001)
        min_qty = exchange_info.get("min_qty", 0.001)
        qty = self._round_step(qty, step)

        if qty < min_qty:
            log.warning(f"[{signal.symbol}] qty={qty} below min_qty={min_qty} — skip")
            return None

        notional = qty * entry

        # Ensure required margin does not exceed available capital.
        max_notional_by_margin = capital * risk_cfg.leverage * 0.95
        if notional > max_notional_by_margin:
            capped_qty = self._round_step(max_notional_by_margin / entry, step)
            if capped_qty < min_qty:
                log.warning(f"[{signal.symbol}] capped qty below min_qty after margin cap — skip")
                return None
            log.info(
                f"[{signal.symbol}] capping size for margin safety: qty {qty} -> {capped_qty}"
            )
            qty = capped_qty
            notional = qty * entry

        min_notional = exchange_info.get("min_notional", 5.0)
        if notional < min_notional:
            log.warning(f"[{signal.symbol}] notional=${notional:.2f} < min=${min_notional}")
            return None

        actual_risk = qty * risk_per_unit / risk_cfg.leverage
        log.info(
            f"[{signal.symbol}] {signal.direction.value} | "
            f"qty={qty} | notional=${notional:.2f} | risk=${actual_risk:.2f}"
        )

        return PositionSize(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry,
            quantity=qty,
            notional_usdt=notional,
            stop_loss=sl,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            risk_usdt=actual_risk,
            leverage=risk_cfg.leverage,
        )

    # ------------------------------------------------------------------ #

    def _portfolio_risk_ok(self) -> bool:
        total_risk = sum(
            p.get("risk_usdt", 0) for p in self.portfolio.open_positions.values()
        )
        cap = self.portfolio.total_capital
        max_risk = cap * risk_cfg.max_portfolio_risk
        return total_risk < max_risk

    @staticmethod
    def _round_step(qty: float, step: float) -> float:
        if step <= 0:
            return round(qty, 3)
        precision = max(0, -int(math.floor(math.log10(step))))
        factor = 10 ** precision
        return math.floor(qty * factor) / factor
