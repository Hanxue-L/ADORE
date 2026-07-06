from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from adore_dpv.pairing import build_feature_table


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired ADORE feature table from per-trace DPV features.")
    parser.add_argument("--features", default=str(REPO_ROOT / "examples" / "example_features.csv"))
    parser.add_argument("--pairing", default=str(REPO_ROOT / "examples" / "example_pairing_manifest.csv"))
    parser.add_argument("--output", default=str(REPO_ROOT / "examples" / "example_feature_table.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    pairing = pd.read_csv(args.pairing)
    output = build_feature_table(features, pairing)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
