import unittest

from core.regime import MarketRegime
from strategies.ensemble import EnsembleStrategy, REGIME_ALLOWLIST


class EnsembleTests(unittest.TestCase):
    def test_dynamic_min_agreement(self):
        self.assertEqual(EnsembleStrategy._min_agree(1), 1)
        self.assertEqual(EnsembleStrategy._min_agree(2), 1)
        self.assertEqual(EnsembleStrategy._min_agree(3), 2)
        self.assertEqual(EnsembleStrategy._min_agree(4), 3)

    def test_ranging_allowlist_includes_mean_reversion(self):
        allowed = REGIME_ALLOWLIST[MarketRegime.RANGING]
        self.assertIn("mean_reversion", allowed)


if __name__ == "__main__":
    unittest.main()
