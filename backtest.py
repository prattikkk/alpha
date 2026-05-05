"""
backtest.py — Offline backtester using mainnet historical data.
Run this BEFORE going live to validate the strategy on your chosen symbols.

Usage:
    python backtest.py --symbol BTCUSDT --tf 15m --days 90
"""
from __future__ import annotations
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import CONFIG
from core.signal import Direction
from strategies.ensemble import EnsembleStrategy
from strategies.supertrend_rsi import SuperTrendRSIStrategy
from strategies.ema_adx_volume import EMAAdxVolumeStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from utils.logger import get_logger

log = get_logger("Backtest")

STRATEGY_MAP = {
    "ensemble":          EnsembleStrategy,
    "supertrend_rsi":    SuperTrendRSIStrategy,
    "ema_adx_volume":    EMAAdxVolumeStrategy,
    "breakout_momentum": BreakoutMomentumStrategy,
}

INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}


def fetch_historical(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch up to `days` days of candles from Binance mainnet in chunks."""
    import requests
    limit = 1000
    ms_per_candle = INTERVAL_MINUTES.get(interval, 15) * 60 * 1000
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 24 * 3600 * 1000

    all_rows = []
    session = requests.Session()
    while start_ts < end_ts:
        try:
            r = session.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={
                    "symbol": symbol, "interval": interval,
                    "startTime": start_ts, "limit": limit,
                },
                timeout=15,
            )
            data = r.json()
            if not data:
                break
            all_rows.extend(data)
            start_ts = data[-1][0] + ms_per_candle
            time.sleep(0.1)
        except Exception as e:
            log.error(f"Fetch error: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols).drop(columns=["_ignore"])
    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df["taker_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    return df.drop_duplicates()


class Backtest:
    def __init__(
        self,
        symbol: str,
        interval: str,
        days: int,
        strategy_name: str,
        initial_capital: float = 1000.0,
        risk_per_trade: float = 0.015,
        leverage: int = 5,
        slippage_bps: float | None = None,
        spread_bps: float | None = None,
        max_hold_hours: float | None = None,
    ):
        self.symbol          = symbol
        self.interval        = interval
        self.days            = days
        self.strategy        = STRATEGY_MAP.get(strategy_name, EnsembleStrategy)()
        self.capital         = initial_capital
        self.risk_per_trade  = risk_per_trade
        self.leverage        = leverage
        self.sl_mult         = CONFIG.risk.atr_sl_multiplier
        self.tp1_mult        = CONFIG.risk.atr_tp1_multiplier
        self.tp2_mult        = CONFIG.risk.atr_tp2_multiplier
        self.slippage_bps    = (
            float(slippage_bps)
            if slippage_bps is not None
            else float(CONFIG.trading.backtest_slippage_bps)
        )
        self.spread_bps      = (
            float(spread_bps)
            if spread_bps is not None
            else float(CONFIG.trading.backtest_spread_bps)
        )
        hold_hours = (
            float(max_hold_hours)
            if max_hold_hours is not None
            else float(CONFIG.trading.backtest_max_hold_hours)
        )
        self.max_hold_bars = self._bars_from_hours(hold_hours)

    def run(self) -> dict:
        log.info(f"Fetching {self.days}d of {self.symbol} {self.interval} candles…")
        df = fetch_historical(self.symbol, self.interval, self.days)
        if df.empty or len(df) < 100:
            log.error("Not enough data")
            return {}

        log.info(f"Got {len(df)} candles. Running backtest…")

        trades = []
        balance = self.capital
        warmup = 60  # bars needed before signaling

        for i in range(warmup + 1, len(df) - 1):
            # Keep one extra bar in the window so strategy closed-bar indexing (-2)
            # aligns with live behavior where newest candle can still be in-flight.
            window = df.iloc[: i + 1].copy()
            signal = self.strategy.generate(self.symbol, window)
            if signal is None or not signal.is_valid:
                continue

            # Position sizing
            entry   = self._apply_entry_cost(signal.entry_price, signal.direction)
            sl      = signal.stop_loss
            risk_amount = balance * self.risk_per_trade
            risk_per_unit = abs(entry - sl)
            if risk_per_unit <= 0:
                continue
            qty     = (risk_amount * self.leverage) / risk_per_unit
            notional = qty * entry

            # Simulate exit using future candles
            outcome = self._simulate_exit(df, i, signal, qty, entry)
            if outcome is None:
                continue

            pnl, exit_price, bars_held, reason = outcome
            balance += pnl

            trades.append({
                "entry_time":  df.index[i],
                "exit_time":   df.index[min(i + bars_held, len(df) - 1)],
                "direction":   signal.direction.value,
                "confidence":  signal.confidence,
                "entry":       entry,
                "entry_signal": signal.entry_price,
                "exit":        exit_price,
                "pnl":         pnl,
                "slippage_bps": self.slippage_bps,
                "spread_bps": self.spread_bps,
                "balance":     balance,
                "reason":      reason,
                "rr":          signal.risk_reward,
            })

            if balance <= 0:
                log.warning("Account blown!")
                break

        return self._report(trades, balance)

    def _simulate_exit(self, df, entry_bar, signal, qty, entry_price):
        """Walk forward to find first SL/TP hit."""
        direction = signal.direction
        entry     = entry_price
        sl        = signal.stop_loss
        tp1       = signal.take_profit_1
        tp2       = signal.take_profit_2
        tp1_hit   = False
        partial_pnl = 0.0

        max_exit_bar = min(entry_bar + self.max_hold_bars, len(df) - 1)
        for j in range(entry_bar + 1, max_exit_bar + 1):
            bar = df.iloc[j]
            h, l = bar["high"], bar["low"]
            bars = j - entry_bar

            if direction == Direction.LONG:
                if not tp1_hit and h >= tp1:
                    tp1_fill = self._apply_exit_cost(tp1, direction, is_stop=False)
                    partial_pnl = (tp1_fill - entry) * qty * 0.5
                    qty *= 0.5
                    tp1_hit = True
                if h >= tp2:
                    tp2_fill = self._apply_exit_cost(tp2, direction, is_stop=False)
                    total = partial_pnl + (tp2_fill - entry) * qty
                    return total, tp2_fill, bars, "TP2"
                if l <= sl:
                    sl_fill = self._apply_exit_cost(sl, direction, is_stop=True)
                    total = partial_pnl + (sl_fill - entry) * qty
                    return total, sl_fill, bars, "SL"
            else:
                if not tp1_hit and l <= tp1:
                    tp1_fill = self._apply_exit_cost(tp1, direction, is_stop=False)
                    partial_pnl = (entry - tp1_fill) * qty * 0.5
                    qty *= 0.5
                    tp1_hit = True
                if l <= tp2:
                    tp2_fill = self._apply_exit_cost(tp2, direction, is_stop=False)
                    total = partial_pnl + (entry - tp2_fill) * qty
                    return total, tp2_fill, bars, "TP2"
                if h >= sl:
                    sl_fill = self._apply_exit_cost(sl, direction, is_stop=True)
                    total = partial_pnl + (entry - sl_fill) * qty
                    return total, sl_fill, bars, "SL"

        # Timeout — exit at last close
        last = df.iloc[max_exit_bar]
        close = last["close"]
        exit_fill = self._apply_exit_cost(close, direction, is_stop=False)
        bars_held = max_exit_bar - entry_bar
        if direction == Direction.LONG:
            total = partial_pnl + (exit_fill - entry) * qty
        else:
            total = partial_pnl + (entry - exit_fill) * qty
        return total, exit_fill, bars_held, "TIMEOUT"

    def _report(self, trades: list, final_balance: float) -> dict:
        if not trades:
            print("\n❌ No trades generated.")
            return {}

        tdf = pd.DataFrame(trades)
        wins   = tdf[tdf["pnl"] > 0]
        losses = tdf[tdf["pnl"] <= 0]
        wr     = len(wins) / len(tdf) * 100
        pf     = abs(wins["pnl"].sum() / losses["pnl"].sum()) if not losses.empty else float("inf")
        max_dd = self._max_drawdown(tdf["balance"].tolist())

        report = {
            "symbol":        self.symbol,
            "interval":      self.interval,
            "strategy":      self.strategy.name,
            "days":          self.days,
            "slippage_bps":  self.slippage_bps,
            "spread_bps":    self.spread_bps,
            "max_hold_bars": self.max_hold_bars,
            "total_trades":  len(tdf),
            "win_rate":      round(wr, 1),
            "profit_factor": round(pf, 2),
            "total_pnl":     round(tdf["pnl"].sum(), 2),
            "avg_win":       round(wins["pnl"].mean(), 2) if not wins.empty else 0,
            "avg_loss":      round(losses["pnl"].mean(), 2) if not losses.empty else 0,
            "max_drawdown":  round(max_dd, 2),
            "final_balance": round(final_balance, 2),
            "return_pct":    round((final_balance - self.capital) / self.capital * 100, 1),
            "tp2_rate":      round(len(tdf[tdf["reason"]=="TP2"]) / len(tdf) * 100, 1),
            "sl_rate":       round(len(tdf[tdf["reason"]=="SL"]) / len(tdf) * 100, 1),
        }

        # Print
        print("\n" + "=" * 55)
        print(f"  📊 BACKTEST RESULTS — {self.symbol} {self.interval}")
        print("=" * 55)
        for k, v in report.items():
            print(f"  {k:<20}: {v}")
        print("=" * 55)

        # Save CSV
        out = Path(f"data/backtest_{self.symbol}_{self.interval}.csv")
        out.parent.mkdir(exist_ok=True)
        tdf.to_csv(out, index=False)
        print(f"\n  Detailed trades → {out}\n")

        return report

    @staticmethod
    def _max_drawdown(balances: list) -> float:
        peak = balances[0]
        max_dd = 0.0
        for b in balances:
            if b > peak:
                peak = b
            dd = (peak - b) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _bars_from_hours(self, hours: float) -> int:
        if hours <= 0:
            return 200
        minutes = INTERVAL_MINUTES.get(self.interval, 15)
        return max(1, int((hours * 60) / minutes))

    def _apply_entry_cost(self, price: float, direction: Direction) -> float:
        half_spread = self.spread_bps / 20000.0
        slip = self.slippage_bps / 10000.0
        if direction == Direction.LONG:
            return price * (1 + half_spread + slip)
        return price * (1 - half_spread - slip)

    def _apply_exit_cost(self, price: float, direction: Direction, is_stop: bool) -> float:
        half_spread = self.spread_bps / 20000.0
        slip = self.slippage_bps / 10000.0
        impact = slip * (1.5 if is_stop else 1.0)
        if direction == Direction.LONG:
            return price * (1 - half_spread - impact)
        return price * (1 + half_spread + impact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaBot Backtester")
    parser.add_argument("--symbol",   default="BTCUSDT")
    parser.add_argument("--tf",       default="15m")
    parser.add_argument("--days",     type=int, default=60)
    parser.add_argument("--strategy", default="ensemble",
                        choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--capital",  type=float, default=1000.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=CONFIG.trading.backtest_slippage_bps,
        help="One-way slippage in basis points applied to entry/exit fills",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=CONFIG.trading.backtest_spread_bps,
        help="Bid/ask spread in basis points",
    )
    parser.add_argument(
        "--max-hold-hours",
        type=float,
        default=CONFIG.trading.backtest_max_hold_hours,
        help="Force timeout exit after this many hours",
    )
    args = parser.parse_args()

    bt = Backtest(
        symbol=args.symbol,
        interval=args.tf,
        days=args.days,
        strategy_name=args.strategy,
        initial_capital=args.capital,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        max_hold_hours=args.max_hold_hours,
    )
    bt.run()
