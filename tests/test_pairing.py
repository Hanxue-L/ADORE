from __future__ import annotations

import unittest

import pandas as pd

from adore_dpv.pairing import build_feature_table


def feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trace_id": "mb_1", "Delta_I": 1.0, "Ep": 2.0, "Ah": 3.0, "FWHM": 4.0},
            {"trace_id": "l012_1", "Delta_I": 5.0, "Ep": 6.0, "Ah": 7.0, "FWHM": 8.0},
        ]
    )


class PairingTest(unittest.TestCase):
    def test_builds_one_natural_pair(self) -> None:
        pairing = pd.DataFrame(
            [
                {"trace_id": "mb_1", "sample_id": "H-1", "group": "Healthy", "analyte": "miR21"},
                {"trace_id": "l012_1", "sample_id": "H-1", "group": "Healthy", "analyte": "miR92a"},
            ]
        )

        observed = build_feature_table(feature_rows(), pairing)

        self.assertEqual(observed.loc[0, "sample_id"], "H-1")
        self.assertEqual(observed.loc[0, "miR21_Delta_I"], 1.0)
        self.assertEqual(observed.loc[0, "miR92a_Delta_I"], 5.0)

    def test_rejects_missing_analyte(self) -> None:
        pairing = pd.DataFrame(
            [{"trace_id": "mb_1", "sample_id": "H-1", "group": "Healthy", "analyte": "miR21"}]
        )
        features = feature_rows().iloc[[0]].copy()

        with self.assertRaisesRegex(ValueError, "exactly one miR21 trace and one miR92a trace"):
            build_feature_table(features, pairing)

    def test_rejects_duplicate_analyte(self) -> None:
        pairing = pd.DataFrame(
            [
                {"trace_id": "mb_1", "sample_id": "H-1", "group": "Healthy", "analyte": "miR21"},
                {"trace_id": "l012_1", "sample_id": "H-1", "group": "Healthy", "analyte": "miR21"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "exactly one miR21 trace and one miR92a trace"):
            build_feature_table(feature_rows(), pairing)


if __name__ == "__main__":
    unittest.main()
