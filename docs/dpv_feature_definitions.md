# DPV Feature Definitions

Each DPV trace is represented by four scalar features.

| Feature | Definition |
| --- | --- |
| `Delta_I` | Blank reference current minus the valley-to-valley baseline-corrected peak current (`ip`). |
| `Ep` | Potential at the extracted DPV peak. |
| `Ah` | Absolute trapezoidal area of the valley-to-valley baseline-corrected peak region. |
| `FWHM` | Full width at half maximum calculated from the valley-to-valley baseline-corrected peak region. |

The example workflow reads voltage/current pairs from `examples/raw_dpv/` and uses `examples/dpv_example_manifest.csv` for the blank reference current values. The `current_scale` column converts raw current values into the current unit used for `blank_current`.

For `Delta_I`, the workflow uses:

```text
Delta_I = blank_current - ip
```

The `ip` value is extracted from the raw DPV trace after valley-to-valley baseline correction. Set `blank_current` according to the blank or control current measured in the assay.

## Input and Units

The input contains two columns: voltage and current. Voltage is expressed in volts. `current_scale` converts the raw current into the unit used for `blank_current` and `Delta_I`. `Ep` and `FWHM` are reported in volts, while `Ah` is reported as current unit multiplied by volts.

The valley-to-valley baseline is linearly interpolated in voltage coordinates. Input voltage values must be strictly monotonic and contain at least three finite voltage/current pairs. The extracted upward peak must have a valley on each side.
