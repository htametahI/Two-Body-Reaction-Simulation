"""Check live-time normalization, uncertainty propagation, and safe CSV reuse."""
from pathlib import Path
import io
import math
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Plotting"))
from ssb_astro_spectroscopic_factors import apply_emma_live_time_correction


class SpectroscopicLiveTime(unittest.TestCase):
    def setUp(self):
        # Two independent yields with a shared observation efficiency.
        self.original = pd.DataFrame({
            "MCMC_yield": [100.0, 50.0],
            "total_observed_efficiency": [0.25, 0.25],
            "unit_strength_observed_yield": [1000.0, 1000.0],
            "alpha_spectroscopic_factor": [0.1, 0.05],
            "alpha_spectroscopic_factor_fit_minus": [0.01, 0.005],
            "alpha_spectroscopic_factor_fit_plus": [0.02, 0.01],
            "total_relative_uncertainty_minus": [0.2, 0.3],
            "total_relative_uncertainty_plus": [0.3, 0.4],
            "alpha_spectroscopic_factor_total_minus": [0.02, 0.015],
            "alpha_spectroscopic_factor_total_plus": [0.03, 0.02],
        })

    def test_prediction_preserved_and_uncertainty_added(self):
        corrected = apply_emma_live_time_correction(self.original, 900, 1000)
        np.testing.assert_allclose(
            corrected.alpha_spectroscopic_factor * corrected.unit_strength_observed_yield,
            self.original.MCMC_yield,
        )
        np.testing.assert_allclose(corrected.total_observed_efficiency, 0.225)
        np.testing.assert_allclose(
            corrected.alpha_spectroscopic_factor_fit_plus,
            self.original.alpha_spectroscopic_factor_fit_plus / 0.9,
        )
        sigma = math.sqrt(0.9 * 0.1 / 1000)
        s = 0.1 / 0.9
        # Directly vary the efficiency denominator to derive its asymmetric effect.
        expected_minus = math.hypot(0.02 / 0.9, s - 0.1 / (0.9 + sigma))
        expected_plus = math.hypot(0.03 / 0.9, 0.1 / (0.9 - sigma) - s)
        self.assertAlmostEqual(corrected.iloc[0].alpha_spectroscopic_factor_total_minus, expected_minus)
        self.assertAlmostEqual(corrected.iloc[0].alpha_spectroscopic_factor_total_plus, expected_plus)
        self.assertAlmostEqual(
            corrected.iloc[0].alpha_spectroscopic_factor / corrected.iloc[1].alpha_spectroscopic_factor,
            2.0,
        )
        self.assertNotIn("emma_live_fraction", self.original)

    def test_csv_reuse_does_not_repeat_correction(self):
        first = apply_emma_live_time_correction(self.original, 32524147, 32979454)
        from_csv = pd.read_csv(io.StringIO(first.to_csv(index=False)))
        second = apply_emma_live_time_correction(from_csv, 32524147, 32979454)
        pd.testing.assert_frame_equal(second, from_csv)
        self.assertAlmostEqual(first.iloc[0].emma_live_fraction, 0.9861942226211508)
        self.assertAlmostEqual(first.iloc[0].emma_live_fraction_stat_unc, 0.0000203184198138401)

    def test_mismatched_counts_or_mixed_corrections_rejected(self):
        corrected = apply_emma_live_time_correction(self.original, 900, 1000)
        with self.assertRaisesRegex(ValueError, "different or undocumented"):
            apply_emma_live_time_correction(corrected, 950, 1000)
        corrected.loc[0, "emma_livetime_correction_applied"] = False
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            apply_emma_live_time_correction(corrected, 900, 1000)

    def test_invalid_counts_and_empty_table_rejected(self):
        for accepted, presented in ((0, 100), (-1, 100), (101, 100), (1, 0)):
            with self.subTest(accepted=accepted, presented=presented):
                with self.assertRaises(ValueError):
                    apply_emma_live_time_correction(self.original, accepted, presented)
        with self.assertRaises(ValueError):
            apply_emma_live_time_correction(self.original.iloc[:0], 90, 100)


if __name__ == "__main__":
    unittest.main()
