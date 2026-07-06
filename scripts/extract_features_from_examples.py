from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from adore_dpv.feature_extraction import extract_features_from_file


def main() -> None:
    manifest_path = REPO_ROOT / "examples" / "dpv_example_manifest.csv"
    output_path = REPO_ROOT / "examples" / "example_features.csv"
    manifest = pd.read_csv(manifest_path)

    rows = []
    for row in manifest.to_dict(orient="records"):
        trace_path = REPO_ROOT / "examples" / row["trace_file"]
        rows.append(
            extract_features_from_file(
                trace_path,
                trace_id=row["trace_id"],
                channel=row["channel"],
                blank_current=float(row["blank_current"]),
                current_scale=float(row.get("current_scale", 1.0)),
            )
        )

    output = pd.DataFrame(rows)
    output.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
