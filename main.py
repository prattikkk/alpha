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
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import signal as os_signal
import time
from datetime import datetime
from typing import Iterable

from config import CONFIG
from core.ai_sentiment import AISentimentEngine
from core.control_plane import drain_commands, get_control_state
from core.exchange_factory import create_data_fetcher, create_executor
from core.portfolio import Portfolio
from core.position_monitor import PositionMonitor
from core.risk_manager import RiskManager
from core.signal import Direction
from strategies.adx_trend import ADXTrendStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from strategies.ema_adx_volume import EMAAdxVolumeStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.supertrend_rsi import SuperTrendRSIStrategy
from utils.logger import get_logger
from utils.notifier import notify_event, notify_signal, notify_stats, notify_trade_open

log = get_logger("Main")


STRATEGY_MAP = {
    "adx_trend": ADXTrendStrategy,
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
        self.strategy_cls = STRATEGY_MAP[strategy_name]
        self.strategy = self.strategy_cls()
        self.dry_run = dry_run
        self.exchange = CONFIG.exchange.name
        self._symbol_order = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self._runtime_paused = False
        self._runtime_overrides: dict[str, float | int] = {}

        self.fetcher = create_data_fetcher(self.exchange)
        self.executor = create_executor(self.exchange)
        self.portfolio = Portfolio()
        self.risk = RiskManager(self.portfolio)
        self.monitor = PositionMonitor(self.portfolio, self.executor, self.fetcher)
        self.ai_sentiment = AISentimentEngine()

        self.fetcher.start_price_stream(self.symbols)
        self._refresh_runtime_state()

        self._sync_positions_from_exchange()
        self._sync_balance_from_exchange()

        log.info(
            "Bot initialized | exchange=%s | strategy=%s | dry_run=%s | symbols=%s",
            self.exchange,
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

    def _sync_balance_from_exchange(self) -> None:
        if self.dry_run:
            return

        quote_asset = str(CONFIG.binance.quote_asset or "USDT").upper()
        balance_info = self.executor.get_quote_asset_balance(quote_asset)
        if not balance_info:
            log.warning("Startup balance sync: exchange balance unavailable")
            return

        wallet_balance = float(balance_info.get("wallet_balance", 0.0))
        available_balance = float(balance_info.get("available_balance", wallet_balance))
        sync_balance = available_balance if available_balance > 0 else wallet_balance
        if sync_balance <= 0:
            log.warning(
                "Startup balance sync: non-positive balance for %s (wallet=%.4f available=%.4f)",
                quote_asset,
                wallet_balance,
                available_balance,
            )
            return

        reset_total = len(self.portfolio.closed_trades) == 0
        self.portfolio.sync_exchange_balance(sync_balance, reset_total_capital=reset_total)
        log.info(
            "Startup balance sync complete | asset=%s wallet=%.2f available=%.2f",
            quote_asset,
            wallet_balance,
            available_balance,
        )

    def _refresh_runtime_state(self) -> None:
        state = get_control_state()
        self._runtime_paused = bool(state.get("paused", False))

        overrides = state.get("overrides", {})
        parsed: dict[str, float | int] = {}
        if isinstance(overrides, dict):
            min_conf = overrides.get("min_confidence")
            if min_conf is not None:
                try:
                    parsed["min_confidence"] = max(0.0, min(1.0, float(min_conf)))
                except Exception:
                    pass

            corr_threshold = overrides.get("correlation_threshold")
            if corr_threshold is not None:
                try:
                    parsed["correlation_threshold"] = max(0.0, min(1.0, float(corr_threshold)))
                except Exception:
                    pass

            max_corr = overrides.get("max_correlated_positions")
            if max_corr is not None:
                try:
                    parsed["max_correlated_positions"] = max(1, int(max_corr))
                except Exception:
                    pass

        self._runtime_overrides = parsed

    def _effective_min_confidence(self) -> float:
        override = self._runtime_overrides.get("min_confidence")
        if override is None:
            return float(CONFIG.strategy.min_confidence)
        return float(override)

    def _effective_correlation_threshold(self) -> float:
        override = self._runtime_overrides.get("correlation_threshold")
        if override is None:
            return float(CONFIG.risk.correlation_threshold)
        return float(override)

    def _effective_max_correlated_positions(self) -> int:
        override = self._runtime_overrides.get("max_correlated_positions")
        if override is None:
            return max(1, int(CONFIG.risk.max_correlated_positions))
        return max(1, int(override))

    def _process_runtime_commands(self) -> None:
        commands = drain_commands()
        if not commands:
            return

        for command in commands:
            action = str(command.get("action", "")).strip().lower()
            payload = command.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}

            if action == "close_symbol":
                symbol = str(payload.get("symbol", "")).strip().upper()
                if symbol:
                    self._force_close_symbol(symbol, reason="DASHBOARD_CLOSE")
                continue

            if action == "close_all":
                symbols = sorted(self.portfolio.open_positions.keys())
                for symbol in symbols:
                    self._force_close_symbol(symbol, reason="DASHBOARD_CLOSE_ALL")
                continue

            log.warning("Ignoring unsupported runtime command: %s", action)

    def _force_close_symbol(self, symbol: str, reason: str) -> None:
        pos = self.portfolio.open_positions.get(symbol)
        if not pos:
            return

        self._cancel_protective_orders(symbol, pos)

        qty = float(pos.get("quantity", 0.0))
        if qty > 0 and not self.dry_run:
            self.executor.close_position_market(symbol, pos.get("direction", "LONG"), qty)

        exit_price = self.fetcher.get_current_price(symbol)
        if exit_price is None:
            exit_price = float(pos.get("entry_price", 0.0))
        self.portfolio.close_position(symbol, float(exit_price), reason=reason)
        notify_event("Dashboard Action", f"{reason}: closed {symbol}")

    def _cancel_protective_orders(self, symbol: str, pos: dict) -> None:
        order_ids = pos.get("order_ids", {})
        for key in ["sl", "tp1", "tp2", "trail"]:
            oid = order_ids.get(key)
            if oid and oid != "DRY_RUN":
                self.executor.cancel_order(symbol, oid)

    def run_cycle(self) -> None:
        # First manage any existing open positions.
        self.monitor.check_all()
        self._process_runtime_commands()
        self._refresh_runtime_state()

        primary_tf = CONFIG.strategy.primary_tf
        htf_1 = CONFIG.strategy.htf_1
        htf_2 = CONFIG.strategy.htf_2
        min_volume_24h = CONFIG.trading.min_volume_24h

        cycle_stats = self.portfolio.stats()
        daily_loss_limit_pct = CONFIG.risk.max_daily_loss_pct * 100
        halt_new_entries = cycle_stats.get("return_pct", 0.0) <= -daily_loss_limit_pct
        if halt_new_entries:
            log.critical(
                "Daily loss guard active | return=%.2f%% <= -%.2f%% | pausing new entries",
                cycle_stats.get("return_pct", 0.0),
                daily_loss_limit_pct,
            )

        if self._runtime_paused:
            halt_new_entries = True
            log.warning("Runtime pause is active; skipping new entries this cycle")

        eligible_symbols: list[str] = []
        cycle_rejections: dict[str, str] = {}
        opened_symbols: set[str] = set()

        for symbol in self.symbols:
            if halt_new_entries:
                self._record_cycle_rejection(cycle_rejections, symbol, "global_halt_new_entries")
                continue

            if symbol in self.portfolio.open_positions:
                log.debug("[%s] position already open, skipping new entry", symbol)
                self._record_cycle_rejection(cycle_rejections, symbol, "already_open_position")
                continue

            if min_volume_24h > 0:
                quote_volume = self.fetcher.get_24h_quote_volume(symbol)
                if quote_volume is None:
                    log.warning("[%s] 24h quote volume unavailable, skipping", symbol)
                    self._record_cycle_rejection(cycle_rejections, symbol, "volume_unavailable")
                    continue
                if quote_volume < min_volume_24h:
                    log.info(
                        "[%s] skipped: 24h volume %.0f < min %.0f",
                        symbol,
                        quote_volume,
                        min_volume_24h,
                    )
                    self._record_cycle_rejection(
                        cycle_rejections,
                        symbol,
                        f"volume_below_min({quote_volume:.0f}<{min_volume_24h:.0f})",
                    )
                    continue

            eligible_symbols.append(symbol)

        multi_tf_by_symbol = self.fetcher.get_multi_tf_bulk(eligible_symbols) if eligible_symbols else {}
        min_confidence = self._effective_min_confidence()

        signal_candidates: list[dict] = []
        if eligible_symbols:
            max_workers = min(
                len(eligible_symbols),
                max(1, int(CONFIG.api.max_concurrent_requests)),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        self._generate_signal_candidate,
                        symbol,
                        multi_tf_by_symbol.get(symbol, {}),
                        primary_tf,
                        htf_1,
                        htf_2,
                        min_confidence,
                    ): symbol
                    for symbol in eligible_symbols
                }

                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        candidate = future.result()
                    except Exception as e:
                        log.error("[%s] signal generation failed: %s", symbol, e)
                        self._record_cycle_rejection(cycle_rejections, symbol, "signal_generation_error")
                        continue
                    if candidate is None:
                        self._record_cycle_rejection(cycle_rejections, symbol, "signal_generation_empty")
                        continue

                    rejection_reason = candidate.get("rejection_reason")
                    if rejection_reason:
                        self._record_cycle_rejection(cycle_rejections, symbol, str(rejection_reason))
                        continue

                    signal_candidates.append(candidate)

        signal_candidates.sort(key=lambda item: self._symbol_order.get(item["symbol"], 10**9))

        corr_matrix: dict[tuple[str, str], float] = {}
        if bool(CONFIG.risk.correlation_management_enabled) and signal_candidates:
            corr_symbols = [c["symbol"] for c in signal_candidates]
            corr_symbols.extend(self.portfolio.open_positions.keys())
            corr_matrix = self.fetcher.get_close_correlation_matrix(
                corr_symbols,
                primary_tf,
                int(CONFIG.risk.correlation_lookback),
            )

        for candidate in signal_candidates:
            symbol = candidate["symbol"]
            signal = candidate["signal"]
            df = candidate["df"]

            funding_threshold = CONFIG.trading.max_unfavorable_funding_rate
            if funding_threshold > 0:
                funding_rate = self.fetcher.get_funding_rate(symbol)
                if funding_rate is not None:
                    if signal.direction == Direction.LONG and funding_rate > funding_threshold:
                        log.info(
                            "[%s] skipped: funding %.5f too high for LONG (threshold %.5f)",
                            symbol,
                            funding_rate,
                            funding_threshold,
                        )
                        self._record_cycle_rejection(
                            cycle_rejections,
                            symbol,
                            f"funding_unfavorable_long({funding_rate:.5f}>{funding_threshold:.5f})",
                        )
                        continue
                    if signal.direction == Direction.SHORT and funding_rate < -funding_threshold:
                        log.info(
                            "[%s] skipped: funding %.5f too low for SHORT (threshold %.5f)",
                            symbol,
                            funding_rate,
                            funding_threshold,
                        )
                        self._record_cycle_rejection(
                            cycle_rejections,
                            symbol,
                            f"funding_unfavorable_short({funding_rate:.5f}<-{funding_threshold:.5f})",
                        )
                        continue

            correlation_reason = self._is_correlation_blocked(symbol, primary_tf, corr_matrix)
            if correlation_reason:
                self._record_cycle_rejection(cycle_rejections, symbol, correlation_reason)
                continue

            if self.ai_sentiment.enabled:
                context = self._ai_context(df)
                regime = str(signal.extra.get("regime", "UNKNOWN"))
                ai_adj = self.ai_sentiment.confidence_adjustment(symbol, signal, regime, context)
                if ai_adj != 0:
                    signal.confidence = max(0.0, min(1.0, signal.confidence + ai_adj))
                    signal.reason = f"{signal.reason} | ai_adj={ai_adj:+.2f}"
                    if not self._signal_is_tradeable(signal, min_confidence):
                        log.info("[%s] AI sentiment reduced confidence below threshold", symbol)
                        rejection_reason = self._signal_rejection_reason(signal, min_confidence)
                        if rejection_reason is None:
                            rejection_reason = "ai_adjustment_rejected"
                        self._record_cycle_rejection(cycle_rejections, symbol, rejection_reason)
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
                self._record_cycle_rejection(cycle_rejections, symbol, "exchange_info_unavailable")
                continue

            position = self.risk.size_position(signal, exchange_info)
            if position is None:
                self._record_cycle_rejection(cycle_rejections, symbol, "risk_sizing_rejected")
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
                    self._record_cycle_rejection(cycle_rejections, symbol, "order_placement_failed")
                    continue

            self.portfolio.open_position(position, signal, order_ids)
            opened_symbols.add(symbol)
            notify_trade_open(
                symbol=symbol,
                direction=signal.direction.value,
                entry=position.entry_price,
                qty=position.quantity,
                notional=position.notional_usdt,
            )

        self._log_cycle_rejection_summary(cycle_rejections, opened_symbols)

        stats = self.portfolio.stats()
        log.info(
            "Cycle complete | balance=$%.2f | open=%s | trades=%s",
            stats.get("balance", 0.0),
            stats.get("open", 0),
            stats.get("trades", 0),
        )
        notify_stats(stats)

    def _generate_signal_candidate(
        self,
        symbol: str,
        multi_tf: dict,
        primary_tf: str,
        htf_1: str,
        htf_2: str,
        min_confidence: float,
    ) -> dict | None:
        df = multi_tf.get(primary_tf)
        if df is None or len(df) < 120:
            log.warning("[%s] not enough %s data to evaluate", symbol, primary_tf)
            observed = 0 if df is None else len(df)
            return {
                "symbol": symbol,
                "rejection_reason": f"insufficient_{primary_tf}_data({observed}<120)",
            }

        strategy = self.strategy_cls()
        signal = strategy.generate(
            symbol=symbol,
            df=df,
            htf_df=multi_tf.get(htf_1),
            htf_df2=multi_tf.get(htf_2),
        )
        if signal is None:
            skip_reason = str(getattr(strategy, "last_skip_reason", "")).strip()
            return {
                "symbol": symbol,
                "rejection_reason": skip_reason or "strategy_no_signal",
            }

        rejection_reason = self._signal_rejection_reason(signal, min_confidence)
        if rejection_reason:
            log.info(
                "[%s] signal rejected | conf=%.2f | rr=%.2f | reason=%s",
                symbol,
                signal.confidence,
                signal.risk_reward,
                rejection_reason,
            )
            return {
                "symbol": symbol,
                "rejection_reason": rejection_reason,
            }

        return {
            "symbol": symbol,
            "signal": signal,
            "df": df,
        }

    def _signal_is_tradeable(self, signal, min_confidence: float) -> bool:
        return self._signal_rejection_reason(signal, min_confidence) is None

    @staticmethod
    def _signal_rejection_reason(signal, min_confidence: float) -> str | None:
        min_conf = float(min_confidence)
        min_rr = float(CONFIG.risk.min_rr_ratio)

        if signal.direction == Direction.FLAT:
            return "direction_flat"
        if signal.confidence < min_conf:
            return f"confidence_below_min({signal.confidence:.2f}<{min_conf:.2f})"
        if signal.risk_reward < min_rr:
            return f"rr_below_min({signal.risk_reward:.2f}<{min_rr:.2f})"
        if signal.stop_loss <= 0:
            return "invalid_stop_loss"
        if signal.take_profit_1 <= 0:
            return "invalid_take_profit_1"
        if signal.take_profit_2 <= 0:
            return "invalid_take_profit_2"
        return None

    @staticmethod
    def _record_cycle_rejection(cycle_rejections: dict[str, str], symbol: str, reason: str) -> None:
        if symbol not in cycle_rejections:
            cycle_rejections[symbol] = reason

    def _log_cycle_rejection_summary(self, cycle_rejections: dict[str, str], opened_symbols: set[str]) -> None:
        skipped_entries: list[str] = []
        for symbol in self.symbols:
            if symbol in opened_symbols:
                continue
            reason = cycle_rejections.get(symbol)
            if reason is None:
                reason = "no_entry_taken"
            skipped_entries.append(f"{symbol}:{reason}")

        if not skipped_entries:
            log.info("Cycle rejection summary | no skipped symbols")
            return

        log.info("Cycle rejection summary | %s", " | ".join(skipped_entries))

    def _is_correlation_blocked(
        self,
        candidate_symbol: str,
        interval: str,
        corr_matrix: dict[tuple[str, str], float] | None = None,
    ) -> str | None:
        if not bool(CONFIG.risk.correlation_management_enabled):
            return None

        open_symbols = [s for s in self.portfolio.open_positions.keys() if s != candidate_symbol]
        if not open_symbols:
            return None

        threshold = float(self._effective_correlation_threshold())
        lookback = int(CONFIG.risk.correlation_lookback)
        max_correlated = self._effective_max_correlated_positions()

        correlated = 0
        for open_symbol in open_symbols:
            corr = None
            if corr_matrix:
                corr = corr_matrix.get((candidate_symbol, open_symbol))
                if corr is None:
                    corr = corr_matrix.get((open_symbol, candidate_symbol))

            if corr is None:
                corr = self.fetcher.get_close_correlation(candidate_symbol, open_symbol, interval, lookback)

            if corr is None:
                continue
            if abs(corr) >= threshold:
                correlated += 1
                log.info(
                    "[%s] high correlation with %s: %.2f",
                    candidate_symbol,
                    open_symbol,
                    corr,
                )

        if correlated >= max_correlated:
            log.warning(
                "[%s] skipped due correlation cap | correlated=%s threshold=%.2f",
                candidate_symbol,
                correlated,
                threshold,
            )
            return (
                f"correlation_cap(correlated={correlated},"
                f"max={max_correlated},threshold={threshold:.2f})"
            )
        return None

    @staticmethod
    def _ai_context(df) -> dict:
        signal_idx = -2
        context = {
            "close": float(df["close"].iloc[signal_idx]),
            "volume": float(df["volume"].iloc[signal_idx]),
        }
        if "taker_ratio" in df.columns:
            context["taker_ratio"] = float(df["taker_ratio"].iloc[signal_idx])
        if "body" in df.columns:
            context["body"] = float(df["body"].iloc[signal_idx])
        return context

    def shutdown(self, close_positions: bool, reason: str = "shutdown") -> None:
        positions = dict(self.portfolio.open_positions)
        if not positions:
            self.fetcher.stop_price_stream()
            notify_event("AlphaBot Shutdown", f"No open positions. Reason: {reason}")
            return

        closed_count = 0
        for symbol, pos in positions.items():
            if close_positions and not self.dry_run:
                self._cancel_protective_orders(symbol, pos)

                qty = float(pos.get("quantity", 0.0))
                if qty > 0:
                    self.executor.close_position_market(symbol, pos.get("direction", "LONG"), qty)
                exit_price = self.fetcher.get_current_price(symbol) or float(pos.get("entry_price", 0.0))
                self.portfolio.close_position(symbol, float(exit_price), reason="SHUTDOWN_EXIT")
                closed_count += 1

        self.portfolio._save()
        self.fetcher.stop_price_stream()
        notify_event(
            "AlphaBot Shutdown",
            (
                f"Reason: {reason}. Open positions: {len(positions)}. "
                f"Closed on shutdown: {closed_count}."
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaBot paper-trading entrypoint")
    parser.add_argument(
        "--symbols",
        default=os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT"),
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--strategy",
        default=os.getenv("ACTIVE_STRATEGY", "adx_trend"),
        choices=list(STRATEGY_MAP.keys()),
        help="Strategy to run",
    )
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=int(os.getenv("ANALYSIS_INTERVAL", str(CONFIG.trading.analysis_interval))),
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

    shutdown_state = {"requested": False, "reason": "manual"}

    def _request_shutdown(reason: str) -> None:
        if shutdown_state["requested"]:
            return
        shutdown_state["requested"] = True
        shutdown_state["reason"] = reason
        log.warning("Shutdown requested: %s", reason)
        notify_event("AlphaBot", f"Shutdown requested: {reason}")

    def _handle_signal(signum, _frame) -> None:
        try:
            name = os_signal.Signals(signum).name
        except Exception:
            name = str(signum)
        _request_shutdown(f"signal:{name}")

    if hasattr(os_signal, "SIGTERM"):
        os_signal.signal(os_signal.SIGTERM, _handle_signal)
    if hasattr(os_signal, "SIGINT"):
        os_signal.signal(os_signal.SIGINT, _handle_signal)

    bot = TradingBot(symbols=symbols, strategy_name=args.strategy, dry_run=dry_run)

    cycle = 0
    try:
        while True:
            if shutdown_state["requested"]:
                break

            cycle += 1
            log.info("Starting cycle %s", cycle)
            bot.run_cycle()

            if cycle_target > 0 and cycle >= cycle_target:
                break

            if shutdown_state["requested"]:
                break

            sleep_for = max(5, args.analysis_interval)
            log.info("Sleeping %ss before next cycle", sleep_for)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        _request_shutdown("keyboard_interrupt")

    bot.shutdown(close_positions=bool(CONFIG.trading.close_on_shutdown), reason=shutdown_state["reason"])

    final_stats = bot.portfolio.stats()
    log.info("Final stats: %s", final_stats)


if __name__ == "__main__":
    main()
