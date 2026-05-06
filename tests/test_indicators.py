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


def _volatile_df(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="15min")
    rng = np.random.default_rng(7)

    drift = np.linspace(0.0, 12.0, rows)
    close = 100.0 + drift + rng.normal(0.0, 1.2, rows).cumsum() * 0.18
    open_ = close + rng.normal(0.0, 0.5, rows)

    hi_pad = np.abs(rng.normal(0.8, 0.35, rows))
    lo_pad = np.abs(rng.normal(0.8, 0.35, rows))
    high = np.maximum(open_, close) + hi_pad
    low = np.minimum(open_, close) - lo_pad
    volume = rng.integers(900, 2200, size=rows).astype(float)

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

    def test_atr_uses_wilder_smoothing(self):
        df = _volatile_df()
        out = atr(df, period=14)

        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

        expected_wilder = tr.ewm(com=13, adjust=False).mean()
        expected_span = tr.ewm(span=14, adjust=False).mean()

        diff_wilder = (out - expected_wilder).abs().dropna().mean()
        diff_span = (out - expected_span).abs().dropna().mean()

        self.assertLess(diff_wilder, 1e-12)
        self.assertGreater(diff_span, diff_wilder)

    def test_adx_prefers_wilder_pipeline(self):
        df = _volatile_df()
        out_adx, out_di_pos, out_di_neg = adx(df, period=14)

        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_h = h.shift(1)
        prev_l = l.shift(1)
        prev_c = c.shift(1)

        up_move = h - prev_h
        down_move = prev_l - l

        pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

        atr_wilder = tr.ewm(com=13, adjust=False).mean()
        atr_span = tr.ewm(span=14, adjust=False).mean()

        sp_wilder = pos_dm.ewm(com=13, adjust=False).mean()
        sn_wilder = neg_dm.ewm(com=13, adjust=False).mean()
        di_pos_wilder = 100 * sp_wilder / atr_wilder.replace(0, np.nan)
        di_neg_wilder = 100 * sn_wilder / atr_wilder.replace(0, np.nan)
        dx_wilder = 100 * (di_pos_wilder - di_neg_wilder).abs() / (di_pos_wilder + di_neg_wilder).replace(0, np.nan)
        adx_wilder = dx_wilder.ewm(com=13, adjust=False).mean()

        sp_span = pos_dm.ewm(span=14, adjust=False).mean()
        sn_span = neg_dm.ewm(span=14, adjust=False).mean()
        di_pos_span = 100 * sp_span / atr_span.replace(0, np.nan)
        di_neg_span = 100 * sn_span / atr_span.replace(0, np.nan)
        dx_span = 100 * (di_pos_span - di_neg_span).abs() / (di_pos_span + di_neg_span).replace(0, np.nan)
        adx_span = dx_span.ewm(span=14, adjust=False).mean()

        adx_diff_wilder = (out_adx - adx_wilder).abs().dropna().mean()
        adx_diff_span = (out_adx - adx_span).abs().dropna().mean()
        di_pos_diff_wilder = (out_di_pos - di_pos_wilder).abs().dropna().mean()
        di_pos_diff_span = (out_di_pos - di_pos_span).abs().dropna().mean()
        di_neg_diff_wilder = (out_di_neg - di_neg_wilder).abs().dropna().mean()
        di_neg_diff_span = (out_di_neg - di_neg_span).abs().dropna().mean()

        self.assertLess(adx_diff_wilder, adx_diff_span)
        self.assertLess(di_pos_diff_wilder, di_pos_diff_span)
        self.assertLess(di_neg_diff_wilder, di_neg_diff_span)

    def test_supertrend_flips_on_reversal(self):
        rows = 180
        idx = pd.date_range("2025-02-01", periods=rows, freq="1h")
        up_leg = np.linspace(100, 125, rows // 2)
        down_leg = np.linspace(125, 92, rows - rows // 2)
        close = np.concatenate([up_leg, down_leg])
        open_ = close - 0.2
        high = np.maximum(open_, close) + 0.7
        low = np.minimum(open_, close) - 0.7
        volume = np.linspace(1000, 1400, rows)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )

        st_line, st_dir = supertrend(df, period=10, multiplier=3.0)
        self.assertFalse(st_line.iloc[5:].isna().any())
        self.assertFalse(st_dir.isna().any())

        direction_changes = (st_dir.shift(1) != st_dir).sum()
        self.assertGreaterEqual(int(direction_changes), 2)


if __name__ == "__main__":
    unittest.main()
