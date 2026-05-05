"""
main.py
Full paper-trading entrypoint for signal scan + position management.

Examples:
    # Single scan cycle (safe smoke test)
    python main.py --once --dry-run

    # Single cycle with real TESTNET order placement
    python main.py --once --live

    # Continuous loop every 5 minutes
    python main.py --analysis-interval 300
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Iterable

from config import CONFIG
from core.data_fetcher import DataFetcher
from core.executor import TestnetExecutor
from core.portfolio import Portfolio
from core.position_monitor import PositionMonitor
from core.risk_manager import RiskManager
from strategies.breakout_momentum import BreakoutMomentumStrategy
from strategies.ema_adx_volume import EMAAdxVolumeStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.supertrend_rsi import SuperTrendRSIStrategy
from utils.logger import get_logger
from utils.notifier import notify_signal, notify_stats, notify_trade_open

log = get_logger("Main")


STRATEGY_MAP = {
    "ensemble": EnsembleStrategy,
    "supertrend_rsi": SuperTrendRSIStrategy,
    "ema_adx_volume": EMAAdxVolumeStrategy,
    "breakout_momentum": BreakoutMomentumStrategy,
}


def _parse_symbols(raw: str) -> list[str]:
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    # Keep deterministic order while removing duplicates.
    return list(dict.fromkeys(symbols))


def _resolve_dry_run(cli_dry_run: bool, cli_live: bool) -> bool:
    if cli_live:
        return False
    if cli_dry_run:
        return True
    return os.getenv("DRY_RUN", "true").lower() == "true"


def _has_testnet_credentials() -> bool:
    key = os.getenv("BINANCE_TESTNET_API_KEY") or os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_TESTNET_SECRET") or os.getenv("BINANCE_API_SECRET")
    return bool(key and secret)


class TradingBot:
    def __init__(self, symbols: Iterable[str], strategy_name: str, dry_run: bool):
        if strategy_name not in STRATEGY_MAP:
            raise ValueError(f"Unsupported strategy: {strategy_name}")

        self.symbols = list(symbols)
        self.strategy_name = strategy_name
        self.strategy = STRATEGY_MAP[strategy_name]()
        self.dry_run = dry_run

        self.fetcher = DataFetcher()
        self.executor = TestnetExecutor()
        self.portfolio = Portfolio()
        self.risk = RiskManager(self.portfolio)
        self.monitor = PositionMonitor(self.portfolio, self.executor, self.fetcher)

        self._sync_positions_from_exchange()

        log.info(
            "Bot initialized | strategy=%s | dry_run=%s | symbols=%s",
            self.strategy_name,
            self.dry_run,
            ",".join(self.symbols),
        )

    def _sync_positions_from_exchange(self) -> None:
        """Reconcile local portfolio state with exchange positions at startup."""
        exchange_positions = self.executor.get_open_positions(self.symbols)

        local_symbols = set(self.portfolio.open_positions.keys())
        exchange_symbols = set(exchange_positions.keys())

        changed = False

        # Local-only positions are stale after interruption or manual intervention.
        for symbol in sorted(local_symbols - exchange_symbols):
            exit_price = self.fetcher.get_current_price(symbol)
            if exit_price is None:
                exit_price = float(self.portfolio.open_positions[symbol].get("entry_price", 0))
            self.portfolio.close_position(symbol, float(exit_price), reason="SYNC_CLOSED_ON_EXCHANGE")
            changed = True
            log.warning("Startup sync: closed stale local position %s", symbol)

        # Exchange-only positions are imported so monitoring can continue.
        for symbol in sorted(exchange_symbols - local_symbols):
            if self._import_exchange_position(exchange_positions[symbol]):
                changed = True

        # Shared positions are aligned on quantity/direction/entry when drift exists.
        for symbol in sorted(local_symbols & exchange_symbols):
            if self._align_local_position(symbol, exchange_positions[symbol]):
                changed = True

        if changed:
            self.portfolio._save()

        log.info(
            "Startup sync complete | local_open=%s | exchange_open=%s",
            len(self.portfolio.open_positions),
            len(exchange_positions),
        )

    def _import_exchange_position(self, pos: dict) -> bool:
        symbol = pos["symbol"]
        quantity = abs(float(pos.get("quantity", 0)))
        if quantity <= 0:
            return False

        direction = "LONG" if float(pos.get("quantity", 0)) > 0 else "SHORT"

        entry_price = float(pos.get("entry_price", 0) or 0)
        mark_price = float(pos.get("mark_price", 0) or 0)
        if entry_price <= 0:
            entry_price = mark_price
        if entry_price <= 0:
            live_price = self.fetcher.get_current_price(symbol)
            if live_price is not None:
                entry_price = float(live_price)
        if entry_price <= 0:
            log.warning("Startup sync: could not infer entry price for %s; skipping import", symbol)
            return False

        sl_pct = float(os.getenv("STOP_LOSS_PCT", "0.03"))
        tp_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.06"))

        if direction == "LONG":
            stop_loss = entry_price * (1 - sl_pct)
            take_profit_1 = entry_price * (1 + tp_pct * 0.5)
            take_profit_2 = entry_price * (1 + tp_pct)
        else:
            stop_loss = entry_price * (1 + sl_pct)
            take_profit_1 = entry_price * (1 - tp_pct * 0.5)
            take_profit_2 = entry_price * (1 - tp_pct)

        leverage = max(1, int(float(pos.get("leverage", 1) or 1)))
        notional = abs(float(pos.get("notional", 0) or 0))
        if notional <= 0:
            notional = quantity * entry_price

        risk_usdt = abs(entry_price - stop_loss) * quantity / leverage

        self.portfolio.open_positions[symbol] = {
            "id": f"{symbol}_sync_{int(time.time())}",
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "quantity": quantity,
            "notional": notional,
            "risk_usdt": risk_usdt,
            "strategy": "startup_sync",
            "confidence": 1.0,
            "open_time": datetime.utcnow().isoformat(),
            "close_time": None,
            "exit_price": None,
            "pnl": 0.0,
            "status": "SYNCED_OPEN",
            "tp1_hit": False,
            "leverage": leverage,
            "order_ids": {
                "entry": "SYNC_IMPORT",
                "sl": None,
                "tp1": None,
                "tp2": None,
            },
        }
        log.warning(
            "Startup sync: imported exchange position %s %s qty=%s",
            symbol,
            direction,
            quantity,
        )
        return True

    def _align_local_position(self, symbol: str, remote: dict) -> bool:
        local = self.portfolio.open_positions.get(symbol)
        if not local:
            return False

        changed = False

        remote_qty = abs(float(remote.get("quantity", 0) or 0))
        local_qty = abs(float(local.get("quantity", 0) or 0))
        if remote_qty > 0 and abs(local_qty - remote_qty) / remote_qty > 0.001:
            local["quantity"] = remote_qty
            changed = True

        remote_dir = "LONG" if float(remote.get("quantity", 0) or 0) > 0 else "SHORT"
        if local.get("direction") != remote_dir:
            local["direction"] = remote_dir
            changed = True

        remote_entry = float(remote.get("entry_price", 0) or 0)
        if remote_entry <= 0:
            remote_entry = float(remote.get("mark_price", 0) or 0)
        local_entry = float(local.get("entry_price", 0) or 0)
        if remote_entry > 0 and (local_entry <= 0 or abs(local_entry - remote_entry) / remote_entry > 0.005):
            local["entry_price"] = remote_entry
            changed = True

        if changed:
            log.warning("Startup sync: aligned local position %s with exchange state", symbol)
        return changed

    def run_cycle(self) -> None:
        # First manage any existing open positions.
        self.monitor.check_all()

        primary_tf = CONFIG.strategy.primary_tf
        htf_1 = CONFIG.strategy.htf_1
        htf_2 = CONFIG.strategy.htf_2

        for symbol in self.symbols:
            if symbol in self.portfolio.open_positions:
                log.debug("[%s] position already open, skipping new entry", symbol)
                continue

            multi_tf = self.fetcher.get_multi_tf(symbol)
            df = multi_tf.get(primary_tf)
            if df is None or len(df) < 120:
                log.warning("[%s] not enough %s data to evaluate", symbol, primary_tf)
                continue

            signal = self.strategy.generate(
                symbol=symbol,
                df=df,
                htf_df=multi_tf.get(htf_1),
                htf_df2=multi_tf.get(htf_2),
            )

            if signal is None:
                continue

            if not signal.is_valid:
                log.info(
                    "[%s] signal rejected | conf=%.2f | rr=%.2f | reason=%s",
                    symbol,
                    signal.confidence,
                    signal.risk_reward,
                    signal.reason,
                )
                continue

            log.info(
                "[%s] valid signal %s | conf=%.2f | rr=%.2f",
                symbol,
                signal.direction.value,
                signal.confidence,
                signal.risk_reward,
            )
            notify_signal(signal)

            exchange_info = self.fetcher.get_exchange_info(symbol)
            if not exchange_info:
                log.warning("[%s] exchange info unavailable, skipping", symbol)
                continue

            position = self.risk.size_position(signal, exchange_info)
            if position is None:
                continue

            if self.dry_run:
                order_ids = {
                    "entry": "DRY_RUN",
                    "sl": "DRY_RUN",
                    "tp1": "DRY_RUN",
                    "tp2": "DRY_RUN",
                }
                log.info("[DRY_RUN] [%s] order placement skipped", symbol)
            else:
                order_ids = self.executor.open_position(position)
                if not order_ids or not order_ids.get("entry"):
                    log.warning("[%s] order placement failed, skipping portfolio open", symbol)
                    continue

            self.portfolio.open_position(position, signal, order_ids)
            notify_trade_open(
                symbol=symbol,
                direction=signal.direction.value,
                entry=position.entry_price,
                qty=position.quantity,
                notional=position.notional_usdt,
            )

        stats = self.portfolio.stats()
        log.info(
            "Cycle complete | balance=$%.2f | open=%s | trades=%s",
            stats.get("balance", 0.0),
            stats.get("open", 0),
            stats.get("trades", 0),
        )
        notify_stats(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaBot paper-trading entrypoint")
    parser.add_argument(
        "--symbols",
        default=os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT"),
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--strategy",
        default=os.getenv("ACTIVE_STRATEGY", "ensemble"),
        choices=list(STRATEGY_MAP.keys()),
        help="Strategy to run",
    )
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=int(os.getenv("ANALYSIS_INTERVAL", "300")),
        help="Seconds between cycles when running continuously",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles to run (0 = infinite)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and size trades without submitting testnet orders",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Submit real TESTNET orders",
    )
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise ValueError("No symbols provided")

    dry_run = _resolve_dry_run(args.dry_run, args.live)
    if not dry_run and not _has_testnet_credentials():
        log.warning("No Binance credentials found; switching to dry-run mode")
        dry_run = True

    cycle_target = 1 if args.once else args.cycles

    bot = TradingBot(symbols=symbols, strategy_name=args.strategy, dry_run=dry_run)

    cycle = 0
    try:
        while True:
            cycle += 1
            log.info("Starting cycle %s", cycle)
            bot.run_cycle()

            if cycle_target > 0 and cycle >= cycle_target:
                break

            sleep_for = max(5, args.analysis_interval)
            log.info("Sleeping %ss before next cycle", sleep_for)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        log.info("Interrupted by user, shutting down")

    final_stats = bot.portfolio.stats()
    log.info("Final stats: %s", final_stats)


if __name__ == "__main__":
    main()
