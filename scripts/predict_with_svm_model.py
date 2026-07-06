from __future__ import annotations

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

from adore_dpv.modeling import FEATURE_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the fitted ADORE SVM model to a feature table.")
    parser.add_argument("--input", required=True, help="CSV file containing the ADORE eight-feature schema.")
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "adore_svm_model.joblib"))
    parser.add_argument("--output", default="results/svm_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    sample_ids = df["sample_id"].astype(str) if "sample_id" in df.columns else pd.Series(range(len(df))).astype(str)
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Feature values must be finite")

    model = joblib.load(args.model)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Model classes must contain disease class 1; received {classes}")

    probability = model.predict_proba(X)[:, classes.index(1)]
    prediction = model.predict(X).astype(int)

    output = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "probability_disease": probability,
            "predicted_class": prediction,
            "predicted_label": ["Disease" if value == 1 else "Healthy" for value in prediction],
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
