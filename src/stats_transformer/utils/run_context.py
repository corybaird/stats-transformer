"""Per-run output folder bookkeeping and manifest.json writing for pipeline runs."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from stats_transformer.utils.config import Config


def sha256(path: str) -> str:
    """Hex sha256 digest of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunContext:
    """Tracks a pipeline run's output folder, stage log, produced files and manifest."""

    def __init__(self, config_fn: Callable[[], Config], params_path: Optional[str]) -> None:
        self._config_fn = config_fn
        self._params_path = params_path
        self.run_dir: Optional[str] = None
        self.files: List[str] = []
        self.shapes: Dict[str, Any] = {}
        self.stages_run: List[str] = []
        self.start: Optional[str] = None
        self.full_run_active: bool = False

    def output_root(self) -> str:
        """The configured output root (default data/pipeline)."""
        return self._config_fn().to_dict().get("output", {}).get("root") or "data/pipeline"

    def begin_run(self) -> str:
        """Create (or reuse) this run's per-run output folder."""
        if self.run_dir is None:
            stem = os.path.splitext(os.path.basename(self._params_path or "pipeline"))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate = os.path.join(self.output_root(), stem, ts)
            suffix = 1
            while os.path.isdir(candidate) and os.listdir(candidate):
                candidate = os.path.join(self.output_root(), stem, f"{ts}_{suffix}")
                suffix += 1
            os.makedirs(candidate, exist_ok=True)
            self.run_dir = candidate
            self.files = []
            self.shapes = {}
            self.stages_run = []
            self.start = datetime.now(timezone.utc).isoformat()
        return self.run_dir

    def end_stage(self, stage: str) -> None:
        """Record a finished stage; close the run folder and write the manifest for single-stage runs."""
        if self.run_dir is None:
            return
        self.stages_run.append(stage)
        if not self.full_run_active:
            self.write_manifest()
            self.run_dir = None

    def write_manifest(self) -> None:
        """Write manifest.json into the current run folder (no-op if no run is open)."""
        if self.run_dir is None:
            return
        params = self._config_fn().to_dict()
        inputs: Dict[str, str] = {}
        data_config = params.get("data", {})
        input_paths = (
            [ds.get("path") for ds in data_config.get("datasets", [])] + [data_config.get("raw_data_file")]
        )
        for path in input_paths:
            if path and os.path.isfile(path) and path not in inputs:
                try:
                    inputs[path] = sha256(path)
                except OSError:
                    pass
        try:
            git_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip() or None
        except Exception:
            git_sha = None
        config_sha256: Optional[str] = None
        if self._params_path and os.path.isfile(self._params_path):
            try:
                config_sha256 = sha256(self._params_path)
            except OSError:
                config_sha256 = None
        try:
            from stats_transformer import __version__ as package_version
        except Exception:
            package_version = "unknown"
        manifest = {
            "stages_run": list(self.stages_run),
            "started_at_utc": self.start,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": params,
            "config_sha256": config_sha256,
            "inputs_sha256": inputs,
            "git_sha": git_sha,
            "package_version": package_version,
            "files": list(self.files),
            "shapes": dict(self.shapes),
        }
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)