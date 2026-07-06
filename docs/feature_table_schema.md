# Feature Table Schema

The classification script accepts one CSV row per paired sample.

```csv
sample_id,group,miR92a_Delta_I,miR21_Delta_I,miR92a_Ep,miR21_Ep,miR92a_Ah,miR21_Ah,miR92a_FWHM,miR21_FWHM
```

| Column | Description |
| --- | --- |
| `sample_id` | Anonymous paired sample identifier. |
| `group` | Class label: `Healthy` or `Disease`. |
| `miR92a_Delta_I` | Current-response feature from the L-012/miR-92a trace. |
| `miR21_Delta_I` | Current-response feature from the MB/miR-21 trace. |
| `miR92a_Ep` | Peak-potential feature from the L-012/miR-92a trace. |
| `miR21_Ep` | Peak-potential feature from the MB/miR-21 trace. |
| `miR92a_Ah` | Peak-area feature from the L-012/miR-92a trace. |
| `miR21_Ah` | Peak-area feature from the MB/miR-21 trace. |
| `miR92a_FWHM` | Peak-width feature from the L-012/miR-92a trace. |
| `miR21_FWHM` | Peak-width feature from the MB/miR-21 trace. |

Feature order in the model matrix:

```text
miR92a_Delta_I
miR21_Delta_I
miR92a_Ep
miR21_Ep
miR92a_Ah
miR21_Ah
miR92a_FWHM
miR21_FWHM
```

Build a paired feature table from per-trace feature output:

```bash
python scripts/build_feature_table.py \
  --features examples/example_features.csv \
  --pairing examples/example_pairing_manifest.csv \
  --output examples/example_feature_table.csv
```

The pairing manifest records the sample identifier, group label, and analyte assigned to each DPV trace.
Each `sample_id` must contain exactly one `miR21` trace and one `miR92a` trace.
