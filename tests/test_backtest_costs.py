import unittest

from backtest import Backtest
from core.signal import Direction


class BacktestCostTests(unittest.TestCase):
    def test_entry_cost_directionality(self):
        bt = Backtest(
            symbol="BTCUSDT",
            interval="15m",
            days=1,
            strategy_name="ensemble",
            slippage_bps=5,
            spread_bps=2,
            max_hold_hours=8,
        )
        long_entry = bt._apply_entry_cost(100.0, Direction.LONG)
        short_entry = bt._apply_entry_cost(100.0, Direction.SHORT)
        self.assertGreater(long_entry, 100.0)
        self.assertLess(short_entry, 100.0)

    def test_exit_cost_stop_worse_than_tp(self):
        bt = Backtest(
            symbol="BTCUSDT",
            interval="15m",
            days=1,
            strategy_name="ensemble",
            slippage_bps=5,
            spread_bps=2,
            max_hold_hours=8,
        )
        long_tp_exit = bt._apply_exit_cost(100.0, Direction.LONG, is_stop=False)
        long_sl_exit = bt._apply_exit_cost(100.0, Direction.LONG, is_stop=True)
        self.assertLess(long_sl_exit, long_tp_exit)


if __name__ == "__main__":
    unittest.main()
