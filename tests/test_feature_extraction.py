from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from adore_dpv.feature_extraction import extract_features, extract_features_from_file


class FeatureExtractionTest(unittest.TestCase):
    def test_valley_to_valley_delta_i(self) -> None:
        trace = pd.DataFrame(
            {
                "voltage": np.array([0.0, 1.0, 2.0]),
                "current": np.array([1.0, 5.0, 1.0]),
            }
        )

        features = extract_features(trace, trace_id="test_trace", channel="mb", blank_current=10.0)

        self.assertAlmostEqual(features.Delta_I, 6.0)
        self.assertAlmostEqual(features.Ep, 1.0)
        self.assertAlmostEqual(features.Ah, 4.0)
        self.assertAlmostEqual(features.FWHM, 1.0)

    def test_baseline_uses_voltage_coordinates(self) -> None:
        trace = pd.DataFrame(
            {
                "voltage": np.array([0.0, 1.0, 3.0, 4.0, 6.0]),
                "current": np.array([1.0, 0.0, 5.0, 2.0, 1.0]),
            }
        )

        features = extract_features(trace, trace_id="nonuniform", channel="mb", blank_current=10.0)

        self.assertAlmostEqual(features.Delta_I, 5.4)

    def test_rejects_nonmonotonic_voltage(self) -> None:
        trace = pd.DataFrame(
            {
                "voltage": np.array([0.0, 1.0, 0.5]),
                "current": np.array([1.0, 5.0, 1.0]),
            }
        )

        with self.assertRaisesRegex(ValueError, "strictly monotonic"):
            extract_features(trace, trace_id="invalid", channel="mb", blank_current=10.0)

    def test_example_feature_extraction_matches_reference(self) -> None:
        manifest = pd.read_csv(REPO_ROOT / "examples" / "dpv_example_manifest.csv")
        expected = pd.read_csv(REPO_ROOT / "examples" / "expected_example_features.csv")

        rows = []
        for row in manifest.to_dict(orient="records"):
            rows.append(
                extract_features_from_file(
                    REPO_ROOT / "examples" / row["trace_file"],
                    trace_id=row["trace_id"],
                    channel=row["channel"],
                    blank_current=float(row["blank_current"]),
                    current_scale=float(row.get("current_scale", 1.0)),
                )
            )

        observed = pd.DataFrame(rows)
        pd.testing.assert_frame_equal(observed, expected, check_exact=False, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
