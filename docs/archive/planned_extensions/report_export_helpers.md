# Report Export Helpers

This extension implements a first pass at lightweight report exports for reproducible research workflows.

## Implemented

### `ReportExporter`

The exporter can render:

- Markdown model cards from model metadata.
- JSON run manifests from parameters, outputs, and metrics.
- Markdown figure indexes from generated figure paths and captions.

```python
from stats_transformer.reporting import ReportExporter

exporter = ReportExporter()
metadata = {"r_squared": 0.82, "model": "ols"}

exporter.write_model_card(metadata, "reports/model_card.md", title="OLS Model")

manifest = exporter.run_manifest(
    "okuns-law",
    parameters={"model": "ols"},
    outputs={"summary": "reports/model_summary.json"},
    metrics={"r_squared": 0.82},
)
exporter.write_json_manifest(manifest, "reports/run_manifest.json")
```

## Still Planned

- Pipeline integration after model and visualization stages.
- LaTeX regression-table exports.
- Richer model-card templates by estimator family.
- Links from run manifests to generated figures and table artifacts.
