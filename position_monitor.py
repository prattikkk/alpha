"""
core/position_monitor.py
Monitors open positions and manages exits (TP1/TP2/SL/trailing stop).
Runs every 30 seconds independently of the signal scanner.
"""
from __future__ import annotations
from core.portfolio import Portfolio
from core.executor import TestnetExecutor
from core.data_fetcher import DataFetcher
from core.signal import Direction
from config import CONFIG
from utils.logger import get_logger

log = get_logger("PosMon")
risk_cfg = CONFIG.risk


class PositionMonitor:
    def __init__(
        self,
        portfolio: Portfolio,
        executor: TestnetExecutor,
        fetcher: DataFetcher,
    ):
        self.portfolio = portfolio
        self.executor  = executor
        self.fetcher   = fetcher

    def check_all(self):
        """Called periodically to check all open positions."""
        positions = dict(self.portfolio.open_positions)   # copy to avoid mutation
        for symbol, pos in positions.items():
            try:
                self._check_position(symbol, pos)
            except Exception as e:
                log.error(f"Error monitoring {symbol}: {e}", exc_info=True)

    # ------------------------------------------------------------------ #

    def _check_position(self, symbol: str, pos: dict):
        price = self.fetcher.get_current_price(symbol)
        if price is None:
            return

        direction = pos["direction"]
        entry     = pos["entry_price"]
        sl        = pos["stop_loss"]
        tp1       = pos["take_profit_1"]
        tp2       = pos["take_profit_2"]
        tp1_hit   = pos.get("tp1_hit", False)
        order_ids = pos.get("order_ids", {})

        # --- SL check ---
        if direction == Direction.LONG.value:
            if price <= sl:
                self._close_sl(symbol, price, order_ids)
                return
            if not tp1_hit and price >= tp1:
                self._close_tp1(symbol, price, order_ids)
            elif tp1_hit and price >= tp2:
                self._close_tp2(symbol, price, order_ids)
            # Trailing stop: once TP1 hit, move SL to breakeven
            if tp1_hit:
                breakeven_sl = entry + 0.2 * (tp1 - entry)
                if price < breakeven_sl and price > sl:
                    pos["stop_loss"] = breakeven_sl
                    log.debug(f"[{symbol}] Trailing SL → {breakeven_sl:.4f}")

        else:  # SHORT
            if price >= sl:
                self._close_sl(symbol, price, order_ids)
                return
            if not tp1_hit and price <= tp1:
                self._close_tp1(symbol, price, order_ids)
            elif tp1_hit and price <= tp2:
                self._close_tp2(symbol, price, order_ids)
            if tp1_hit:
                breakeven_sl = entry - 0.2 * (entry - tp1)
                if price > breakeven_sl and price < sl:
                    pos["stop_loss"] = breakeven_sl
                    log.debug(f"[{symbol}] Trailing SL → {breakeven_sl:.4f}")

    def _close_sl(self, symbol: str, price: float, order_ids: dict):
        log.warning(f"⛔ SL triggered [{symbol}] @ {price:.4f}")
        # Cancel remaining TP orders on testnet
        for key in ["tp1", "tp2"]:
            oid = order_ids.get(key)
            if oid:
                self.executor.cancel_order(symbol, oid)
        self.portfolio.close_position(symbol, price, reason="SL_HIT")

    def _close_tp1(self, symbol: str, price: float, order_ids: dict):
        pnl = self.portfolio.update_tp1(symbol, price)
        if pnl is not None:
            log.info(f"🎯 TP1 [{symbol}] @ {price:.4f} | pnl=${pnl:+.2f}")

    def _close_tp2(self, symbol: str, price: float, order_ids: dict):
        log.info(f"🏁 TP2 hit [{symbol}] @ {price:.4f}")
        # Cancel SL on testnet
        sl_id = order_ids.get("sl")
        if sl_id:
            self.executor.cancel_order(symbol, sl_id)
        self.portfolio.close_position(symbol, price, reason="TP2_HIT")
