from __future__ import annotations

import numpy as np
import pandas as pd


ANALYTE_PREFIX = {
    "miR92a": "miR92a",
    "miR21": "miR21",
}

FEATURES = ["Delta_I", "Ep", "Ah", "FWHM"]

OUTPUT_COLUMNS = [
    "sample_id",
    "group",
    "miR92a_Delta_I",
    "miR21_Delta_I",
    "miR92a_Ep",
    "miR21_Ep",
    "miR92a_Ah",
    "miR21_Ah",
    "miR92a_FWHM",
    "miR21_FWHM",
]


def _duplicate_values(df: pd.DataFrame, column: str) -> list[str]:
    duplicated = df.loc[df[column].duplicated(keep=False), column]
    return sorted(duplicated.astype(str).unique())


def build_feature_table(features: pd.DataFrame, pairing: pd.DataFrame) -> pd.DataFrame:
    required_feature_columns = {"trace_id", *FEATURES}
    required_pairing_columns = {"trace_id", "sample_id", "group", "analyte"}

    missing_features = required_feature_columns - set(features.columns)
    missing_pairing = required_pairing_columns - set(pairing.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")
    if missing_pairing:
        raise ValueError(f"Missing pairing columns: {sorted(missing_pairing)}")
    if features.empty or pairing.empty:
        raise ValueError("Feature and pairing tables must contain at least one row")

    if features["trace_id"].isna().any():
        raise ValueError("Feature column 'trace_id' contains missing values")
    for column in required_pairing_columns:
        if pairing[column].isna().any():
            raise ValueError(f"Pairing column '{column}' contains missing values")

    duplicate_features = _duplicate_values(features, "trace_id")
    duplicate_pairing = _duplicate_values(pairing, "trace_id")
    if duplicate_features:
        raise ValueError(f"Duplicate trace_id values in feature table: {duplicate_features}")
    if duplicate_pairing:
        raise ValueError(f"Duplicate trace_id values in pairing table: {duplicate_pairing}")

    feature_ids = set(features["trace_id"].astype(str))
    pairing_ids = set(pairing["trace_id"].astype(str))
    missing_feature_ids = sorted(pairing_ids - feature_ids)
    orphan_feature_ids = sorted(feature_ids - pairing_ids)
    if missing_feature_ids:
        raise ValueError(f"Missing feature rows for trace_id values: {missing_feature_ids}")
    if orphan_feature_ids:
        raise ValueError(f"Feature rows without pairing records: {orphan_feature_ids}")

    normalized_features = features.copy()
    normalized_pairing = pairing.copy()
    normalized_features["trace_id"] = normalized_features["trace_id"].astype(str)
    normalized_pairing["trace_id"] = normalized_pairing["trace_id"].astype(str)
    normalized_pairing["analyte"] = normalized_pairing["analyte"].astype(str)

    merged = normalized_pairing.merge(normalized_features, on="trace_id", how="inner", validate="one_to_one")
    merged[FEATURES] = merged[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(merged[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite")

    rows = []
    expected_analytes = set(ANALYTE_PREFIX)
    for sample_id, group_df in merged.groupby("sample_id", sort=True):
        groups = sorted(group_df["group"].astype(str).unique())
        if len(groups) != 1:
            raise ValueError(f"sample_id {sample_id} has multiple group labels: {groups}")

        analyte_counts = group_df["analyte"].value_counts().to_dict()
        if set(analyte_counts) != expected_analytes or any(count != 1 for count in analyte_counts.values()):
            raise ValueError(
                f"sample_id {sample_id} must contain exactly one miR21 trace and one miR92a trace; "
                f"received {analyte_counts}"
            )

        output_row = {"sample_id": sample_id, "group": groups[0]}
        for record in group_df.to_dict(orient="records"):
            prefix = ANALYTE_PREFIX[record["analyte"]]
            for feature in FEATURES:
                output_row[f"{prefix}_{feature}"] = record[feature]
        rows.append(output_row)

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    feature_output = output[OUTPUT_COLUMNS[2:]].to_numpy(dtype=float)
    if not np.isfinite(feature_output).all():
        raise ValueError("Paired feature table contains non-finite values")
    return output
