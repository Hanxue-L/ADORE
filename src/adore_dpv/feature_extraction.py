from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_dpv_trace


@dataclass(frozen=True)
class DPVFeatures:
    trace_id: str
    channel: str
    Delta_I: float
    Ep: float
    Ah: float
    FWHM: float


@dataclass(frozen=True)
class ValleyToValleyResult:
    corrected_current: np.ndarray
    peak_index: int
    left_valley_index: int
    right_valley_index: int
    ip: float


def valley_to_valley_correction(voltage: np.ndarray, current: np.ndarray) -> ValleyToValleyResult:
    """Apply valley-to-valley baseline correction and extract the peak current."""
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.ndim != 1 or current.ndim != 1:
        raise ValueError("Voltage and current must be one-dimensional arrays")
    if len(voltage) != len(current) or len(voltage) < 3:
        raise ValueError("Voltage and current must contain at least three paired values")
    if not np.isfinite(voltage).all() or not np.isfinite(current).all():
        raise ValueError("Voltage and current values must be finite")

    voltage_steps = np.diff(voltage)
    if not (np.all(voltage_steps > 0) or np.all(voltage_steps < 0)):
        raise ValueError("Voltage values must be strictly monotonic")

    peak_index = int(np.argmax(current))
    if peak_index == 0 or peak_index == len(current) - 1:
        raise ValueError("The extracted peak must have data points on both sides")

    left_valley_index = int(np.argmin(current[: peak_index + 1]))
    right_valley_index = int(peak_index + np.argmin(current[peak_index:]))
    if left_valley_index == peak_index or right_valley_index == peak_index:
        raise ValueError("Valley-to-valley correction requires valleys on both sides of the peak")

    left_voltage = float(voltage[left_valley_index])
    right_voltage = float(voltage[right_valley_index])
    voltage_span = right_voltage - left_voltage
    if voltage_span == 0:
        raise ValueError("Valley voltages must be distinct")

    left_current = float(current[left_valley_index])
    right_current = float(current[right_valley_index])
    fraction = (voltage - left_voltage) / voltage_span
    baseline = left_current + fraction * (right_current - left_current)
    corrected_current = current - baseline
    ip = float(corrected_current[peak_index])
    if not np.isfinite(ip) or ip <= 0:
        raise ValueError("Valley-to-valley correction must produce a positive peak current")

    return ValleyToValleyResult(
        corrected_current=corrected_current,
        peak_index=peak_index,
        left_valley_index=left_valley_index,
        right_valley_index=right_valley_index,
        ip=ip,
    )


def _interpolate_half_height_voltage(
    voltage: np.ndarray,
    current: np.ndarray,
    lower_index: int,
    upper_index: int,
    half_height: float,
) -> float:
    x0 = float(voltage[lower_index])
    x1 = float(voltage[upper_index])
    y0 = float(current[lower_index])
    y1 = float(current[upper_index])
    if y1 == y0:
        return x0
    fraction = (half_height - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0)


def calculate_fwhm(
    voltage: np.ndarray,
    corrected_current: np.ndarray,
    *,
    peak_index: int,
    left_bound: int,
    right_bound: int,
) -> float:
    peak_height = float(corrected_current[peak_index])

    if peak_height <= 0:
        return 0.0

    half_height = peak_height / 2.0
    left_cross = left_bound
    for index in range(peak_index - 1, left_bound - 1, -1):
        if corrected_current[index] <= half_height:
            left_cross = index
            break

    right_cross = right_bound
    for index in range(peak_index + 1, right_bound + 1):
        if corrected_current[index] <= half_height:
            right_cross = index
            break

    left_voltage = (
        _interpolate_half_height_voltage(voltage, corrected_current, left_cross, left_cross + 1, half_height)
        if left_cross < peak_index
        else float(voltage[left_cross])
    )
    right_voltage = (
        _interpolate_half_height_voltage(voltage, corrected_current, right_cross - 1, right_cross, half_height)
        if right_cross > peak_index
        else float(voltage[right_cross])
    )

    return float(abs(right_voltage - left_voltage))


def calculate_area(voltage: np.ndarray, corrected_current: np.ndarray, *, left_bound: int, right_bound: int) -> float:
    local_voltage = voltage[left_bound : right_bound + 1]
    local_current = np.maximum(corrected_current[left_bound : right_bound + 1], 0.0)
    try:
        area = np.trapezoid(local_current, local_voltage)
    except AttributeError:
        area = np.trapz(local_current, local_voltage)
    return float(abs(area))


def extract_features(
    trace: pd.DataFrame,
    *,
    trace_id: str,
    channel: str,
    blank_current: float,
    current_scale: float = 1.0,
) -> DPVFeatures:
    required_columns = {"voltage", "current"}
    missing_columns = required_columns - set(trace.columns)
    if missing_columns:
        raise ValueError(f"Missing trace columns: {sorted(missing_columns)}")
    if not np.isfinite(blank_current):
        raise ValueError("blank_current must be finite")
    if not np.isfinite(current_scale) or current_scale <= 0:
        raise ValueError("current_scale must be a positive finite value")

    voltage = trace["voltage"].to_numpy(dtype=float)
    current = trace["current"].to_numpy(dtype=float) * float(current_scale)

    corrected = valley_to_valley_correction(voltage, current)
    ep = float(voltage[corrected.peak_index])
    delta_i = float(blank_current - corrected.ip)

    return DPVFeatures(
        trace_id=trace_id,
        channel=channel,
        Delta_I=delta_i,
        Ep=ep,
        Ah=calculate_area(
            voltage,
            corrected.corrected_current,
            left_bound=corrected.left_valley_index,
            right_bound=corrected.right_valley_index,
        ),
        FWHM=calculate_fwhm(
            voltage,
            corrected.corrected_current,
            peak_index=corrected.peak_index,
            left_bound=corrected.left_valley_index,
            right_bound=corrected.right_valley_index,
        ),
    )


def extract_features_from_file(
    path: str | Path,
    *,
    trace_id: str,
    channel: str,
    blank_current: float,
    current_scale: float = 1.0,
) -> dict[str, float | str]:
    trace = read_dpv_trace(path)
    features = extract_features(
        trace,
        trace_id=trace_id,
        channel=channel,
        blank_current=blank_current,
        current_scale=current_scale,
    )
    return asdict(features)
