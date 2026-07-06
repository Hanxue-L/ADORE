from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import pandas as pd

from adore_dpv.modeling import FEATURE_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[1]


class SVMPredictionTest(unittest.TestCase):
    def test_script_uses_fitted_svm_classification_rule(self) -> None:
        feature_table = pd.DataFrame(
            [
                {
                    "sample_id": "example_001",
                    "group": "Healthy",
                    "miR92a_Delta_I": 0.21776911764705875,
                    "miR21_Delta_I": 0.1895653164556962,
                    "miR92a_Ep": 0.572,
                    "miR21_Ep": -0.236,
                    "miR92a_Ah": 0.028430158823529398,
                    "miR21_Ah": 0.050102987341772154,
                    "miR92a_FWHM": 0.09729020461072568,
                    "miR21_FWHM": 0.08805174474244387,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            temp_input = temp_dir / "features.csv"
            temp_output = temp_dir / "predictions.csv"
            feature_table.to_csv(temp_input, index=False)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "predict_with_svm_model.py"),
                    "--input",
                    str(temp_input),
                    "--output",
                    str(temp_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            observed = pd.read_csv(temp_output)

        model = joblib.load(REPO_ROOT / "models" / "adore_svm_model.joblib")
        expected = model.predict(feature_table[FEATURE_COLUMNS].to_numpy())
        self.assertEqual(observed["predicted_class"].tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
