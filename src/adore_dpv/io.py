from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def _parse_numeric_pair(line: str) -> tuple[float, float] | None:
    parts = [part for part in re.split(r"[\s,]+", line.strip()) if part]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def read_dpv_trace(path: str | Path) -> pd.DataFrame:
    """Read a DPV trace into voltage/current columns."""
    path = Path(path)
    rows: list[tuple[float, float]] = []
    data_started = False
    content_seen = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip().lstrip("\ufeff")
            if line == "":
                continue

            lower = line.lower()
            if lower.startswith("potential/v") or lower.startswith("voltage"):
                data_started = True
                content_seen = True
                continue

            parsed = _parse_numeric_pair(line)
            if data_started and parsed is not None:
                rows.append(parsed)
            elif not content_seen and parsed is not None:
                data_started = True
                rows.append(parsed)

            content_seen = True

    if len(rows) == 0:
        raise ValueError(f"No voltage/current pairs were parsed from {path}")

    return pd.DataFrame(rows, columns=["voltage", "current"])
