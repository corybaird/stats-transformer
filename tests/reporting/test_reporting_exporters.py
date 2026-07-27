import json

from stats_transformer.reporting import ReportExporter


def test_report_exporter_renders_model_card_markdown():
    exporter = ReportExporter()
    content = exporter.model_card_markdown(
        {"r_squared": 0.91, "features": ["x1", "x2"]},
        title="Regression Model",
        notes="Smoke-test model card.",
    )

    assert "# Regression Model" in content
    assert "Smoke-test model card." in content
    assert "- **features**: x1, x2" in content
    assert "- **r_squared**: 0.91" in content


def test_report_exporter_writes_json_manifest(tmp_path):
    exporter = ReportExporter()
    manifest = exporter.run_manifest(
        "okuns-law",
        parameters={"model": "ols"},
        outputs={"summary": "reports/model_summary.json"},
        metrics={"r_squared": 0.8},
    )

    path = exporter.write_json_manifest(manifest, tmp_path / "manifest.json")
    loaded = json.loads(path.read_text())

    assert loaded["name"] == "okuns-law"
    assert loaded["parameters"]["model"] == "ols"
    assert loaded["metrics"]["r_squared"] == 0.8
    assert "created_at" in loaded


def test_report_exporter_writes_figure_index(tmp_path):
    exporter = ReportExporter()
    figures = [
        {
            "label": "IRF",
            "path": "reports/irf.png",
            "caption": "Impulse response estimate.",
        }
    ]

    path = exporter.write_figure_index(figures, tmp_path / "figures.md")
    content = path.read_text()

    assert "# Figure Index" in content
    assert "**IRF**" in content
    assert "`reports/irf.png`" in content
    assert "Impulse response estimate." in content
