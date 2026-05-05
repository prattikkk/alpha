import unittest

from core.signal import Direction, Signal


class SignalTests(unittest.TestCase):
    def test_risk_reward_long(self):
        sig = Signal(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            confidence=0.9,
            strategy="unit",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=105.0,
            take_profit_2=110.0,
            atr=1.0,
        )
        self.assertAlmostEqual(sig.risk_reward, 2.0)

    def test_risk_reward_short(self):
        sig = Signal(
            symbol="BTCUSDT",
            direction=Direction.SHORT,
            confidence=0.9,
            strategy="unit",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit_1=95.0,
            take_profit_2=90.0,
            atr=1.0,
        )
        self.assertAlmostEqual(sig.risk_reward, 2.0)

    def test_signal_validity(self):
        sig = Signal(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            confidence=0.95,
            strategy="unit",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=106.0,
            take_profit_2=110.0,
            atr=1.0,
        )
        self.assertTrue(sig.is_valid)


if __name__ == "__main__":
    unittest.main()
