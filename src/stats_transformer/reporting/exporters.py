import json
from datetime import datetime, timezone
from pathlib import Path


class ReportExporter:
    """Export model metadata, run manifests, and figure indexes."""

    def model_card_markdown(self, metadata, title="Model Card", notes=None):
        """Render model metadata as a markdown model card."""
        lines = [f"# {title}", ""]
        if notes:
            lines.extend([notes, ""])

        lines.extend(["## Metadata", ""])
        for key, value in sorted(metadata.items()):
            lines.append(f"- **{key}**: {self._format_value(value)}")
        lines.append("")
        return "\n".join(lines)

    def write_model_card(self, metadata, output_path, title="Model Card", notes=None):
        """Write a markdown model card and return its path."""
        content = self.model_card_markdown(metadata, title=title, notes=notes)
        return self._write_text(output_path, content)

    def run_manifest(self, name, parameters=None, outputs=None, metrics=None):
        """Build a JSON-serializable run manifest."""
        return {
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parameters": parameters or {},
            "outputs": outputs or {},
            "metrics": metrics or {},
        }

    def write_json_manifest(self, manifest, output_path):
        """Write a JSON manifest and return its path."""
        path = self._prepare_path(output_path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    def figure_index_markdown(self, figures, title="Figure Index"):
        """Render a markdown index for generated figures."""
        lines = [f"# {title}", ""]
        for figure in figures:
            path = figure.get("path")
            caption = figure.get("caption", "")
            label = figure.get("label", path)
            lines.append(f"- **{label}**: `{path}`")
            if caption:
                lines.append(f"  {caption}")
        lines.append("")
        return "\n".join(lines)

    def write_figure_index(self, figures, output_path, title="Figure Index"):
        """Write a markdown figure index and return its path."""
        content = self.figure_index_markdown(figures, title=title)
        return self._write_text(output_path, content)

    def _write_text(self, output_path, content):
        path = self._prepare_path(output_path)
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _prepare_path(output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _format_value(self, value):
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, tuple)):
            return ", ".join(self._format_value(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        return str(value)
