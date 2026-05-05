"""
core/position_monitor.py
Monitors open positions and manages exits (TP1/TP2/SL/trailing stop).
Runs every 30 seconds independently of the signal scanner.
"""
from __future__ import annotations

from datetime import datetime, timezone

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
        self._trailing_pct = max(0.0, float(risk_cfg.trailing_stop_pct))
        self._max_hold_hours = max(0.0, float(CONFIG.trading.max_hold_hours))

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

        order_ids = pos.setdefault("order_ids", {})

        if self._should_force_time_exit(pos):
            self._close_time_exit(symbol, pos, price)
            return

        direction = pos["direction"]
        sl        = pos["stop_loss"]
        tp1       = pos["take_profit_1"]
        tp2       = pos["take_profit_2"]
        tp1_hit   = pos.get("tp1_hit", False)

        # --- SL check ---
        if direction == Direction.LONG.value:
            if price <= sl:
                self._close_sl(symbol, price, order_ids)
                return

            if not tp1_hit and price >= tp1:
                self._close_tp1(symbol, price, pos)
                if symbol not in self.portfolio.open_positions:
                    return
                pos = self.portfolio.open_positions[symbol]
                tp1_hit = True
            elif tp1_hit and price >= tp2:
                self._close_tp2(symbol, price, order_ids)
                return

            # Trailing stop: once TP1 hit, move SL to breakeven
            if tp1_hit:
                self._update_exchange_trailing_stop(symbol, pos, price)

        else:  # SHORT
            if price >= sl:
                self._close_sl(symbol, price, order_ids)
                return

            if not tp1_hit and price <= tp1:
                self._close_tp1(symbol, price, pos)
                if symbol not in self.portfolio.open_positions:
                    return
                pos = self.portfolio.open_positions[symbol]
                tp1_hit = True
            elif tp1_hit and price <= tp2:
                self._close_tp2(symbol, price, order_ids)
                return

            if tp1_hit:
                self._update_exchange_trailing_stop(symbol, pos, price)

    def _close_sl(self, symbol: str, price: float, order_ids: dict):
        log.warning(f"⛔ SL triggered [{symbol}] @ {price:.4f}")
        # Cancel remaining TP orders on testnet
        for key in ["tp1", "tp2"]:
            oid = order_ids.get(key)
            if oid:
                self.executor.cancel_order(symbol, oid)
        self.portfolio.close_position(symbol, price, reason="SL_HIT")

    def _close_tp1(self, symbol: str, price: float, pos: dict):
        pnl = self.portfolio.update_tp1(symbol, price)
        if pnl is None:
            return

        live_pos = self.portfolio.open_positions.get(symbol)
        if not live_pos:
            return

        order_ids = live_pos.setdefault("order_ids", {})
        new_sl = self._breakeven_stop(live_pos)

        sl_id = order_ids.get("sl")
        if sl_id:
            self.executor.cancel_order(symbol, sl_id)

        new_sl_id = self.executor.place_stop_loss(
            symbol=symbol,
            direction=live_pos["direction"],
            quantity=float(live_pos.get("quantity", 0.0)),
            stop_price=float(new_sl),
        )
        if not new_sl_id:
            log.error("[%s] failed to re-arm stop after TP1; flattening position", symbol)
            self._emergency_flatten(symbol, live_pos, price, reason="PROTECTION_REARM_FAIL")
            return

        live_pos["stop_loss"] = float(new_sl)
        order_ids["sl"] = new_sl_id
        if live_pos["direction"] == Direction.LONG.value:
            live_pos["highest_price"] = float(price)
        else:
            live_pos["lowest_price"] = float(price)
        self.portfolio._save()
        log.info(f"🎯 TP1 [{symbol}] @ {price:.4f} | pnl=${pnl:+.2f} | SL re-armed @ {new_sl:.4f}")

    def _close_tp2(self, symbol: str, price: float, order_ids: dict):
        log.info(f"🏁 TP2 hit [{symbol}] @ {price:.4f}")
        # Cancel SL on testnet
        sl_id = order_ids.get("sl")
        if sl_id:
            self.executor.cancel_order(symbol, sl_id)
        self.portfolio.close_position(symbol, price, reason="TP2_HIT")

    def _update_exchange_trailing_stop(self, symbol: str, pos: dict, price: float):
        if self._trailing_pct <= 0:
            return

        qty = float(pos.get("quantity", 0.0))
        if qty <= 0:
            return

        direction = pos["direction"]
        current_sl = float(pos["stop_loss"])
        entry = float(pos["entry_price"])

        if direction == Direction.LONG.value:
            highest = max(float(pos.get("highest_price", price)), float(price))
            pos["highest_price"] = highest
            candidate = max(current_sl, highest * (1 - self._trailing_pct), entry)
            improved = candidate > current_sl * 1.0005
        else:
            lowest = min(float(pos.get("lowest_price", price)), float(price))
            pos["lowest_price"] = lowest
            candidate = min(current_sl, lowest * (1 + self._trailing_pct), entry)
            improved = candidate < current_sl * 0.9995

        if not improved:
            return

        order_ids = pos.setdefault("order_ids", {})
        old_sl = order_ids.get("sl")
        if old_sl:
            self.executor.cancel_order(symbol, old_sl)

        new_sl_id = self.executor.place_stop_loss(
            symbol=symbol,
            direction=direction,
            quantity=qty,
            stop_price=float(candidate),
        )
        if not new_sl_id:
            log.error("[%s] trailing update failed; flattening position for safety", symbol)
            self._emergency_flatten(symbol, pos, price, reason="TRAIL_GUARD_FAIL")
            return

        pos["stop_loss"] = float(candidate)
        order_ids["sl"] = new_sl_id
        self.portfolio._save()
        log.info("[%s] trailing SL updated to %.4f", symbol, candidate)

    def _emergency_flatten(self, symbol: str, pos: dict, price: float, reason: str):
        order_ids = pos.get("order_ids", {})
        for key in ["sl", "tp1", "tp2"]:
            oid = order_ids.get(key)
            if oid:
                self.executor.cancel_order(symbol, oid)

        self.executor.close_position_market(
            symbol=symbol,
            direction=pos.get("direction", Direction.LONG.value),
            quantity=float(pos.get("quantity", 0.0)),
        )
        self.portfolio.close_position(symbol, price, reason=reason)

    def _close_time_exit(self, symbol: str, pos: dict, price: float):
        log.warning("[%s] max hold time reached; forcing market exit", symbol)
        self._emergency_flatten(symbol, pos, price, reason="TIME_EXIT")

    def _should_force_time_exit(self, pos: dict) -> bool:
        if self._max_hold_hours <= 0:
            return False

        opened_at = self._parse_open_time(pos)
        if opened_at is None:
            return False

        age_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600.0
        return age_hours >= self._max_hold_hours

    @staticmethod
    def _parse_open_time(pos: dict) -> datetime | None:
        raw = pos.get("open_time")
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _breakeven_stop(pos: dict) -> float:
        entry = float(pos["entry_price"])
        tp1 = float(pos["take_profit_1"])
        direction = pos["direction"]
        if direction == Direction.LONG.value:
            return entry + 0.2 * (tp1 - entry)
        return entry - 0.2 * (entry - tp1)
