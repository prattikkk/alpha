import unittest

import numpy as np
import pandas as pd

from core.indicators import adx, atr, supertrend


def _sample_df(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="15min")
    base = np.linspace(100.0, 120.0, rows)
    wobble = np.sin(np.linspace(0, 8, rows)) * 0.7
    close = base + wobble
    open_ = close - 0.2
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.linspace(1000, 2000, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


class IndicatorTests(unittest.TestCase):
    def test_supertrend_direction_values(self):
        df = _sample_df()
        st_line, st_dir = supertrend(df, period=10, multiplier=3.0)
        self.assertEqual(len(st_line), len(df))
        self.assertEqual(len(st_dir), len(df))
        valid = set(int(v) for v in st_dir.dropna().unique())
        self.assertTrue(valid.issubset({-1, 1}))

    def test_atr_outputs(self):
        df = _sample_df()
        out = atr(df, period=14)
        self.assertEqual(len(out), len(df))
        self.assertFalse(out.dropna().empty)

    def test_adx_outputs(self):
        df = _sample_df()
        adx_line, di_pos, di_neg = adx(df, period=14)
        self.assertEqual(len(adx_line), len(df))
        self.assertEqual(len(di_pos), len(df))
        self.assertEqual(len(di_neg), len(df))


if __name__ == "__main__":
    unittest.main()
