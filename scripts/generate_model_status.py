"""Render the model validation status docs from docs/validation/model_status.yaml.

The YAML file is the single source of truth for model validation status. This
script rewrites the generated section of docs/validation/benchmarks.md (between
BEGIN/END markers around the section 2.1 verified-models table) and prints the
LaTeX body rows of the paper's Table 1 (or writes them to --tex-out).

Usage:
    uv run python scripts/generate_model_status.py
    uv run python scripts/generate_model_status.py --tex-out table1_rows.tex
"""

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "docs" / "validation" / "model_status.yaml"
BENCHMARKS_PATH = ROOT / "docs" / "validation" / "benchmarks.md"
BEGIN_MARK = "<!-- BEGIN GENERATED: model-status -->"
END_MARK = "<!-- END GENERATED -->"

MD_HEADER = (
    "| Family | Model | Target Software / Function | Benchmark Dataset | Verification Status & Max Discrepancy |\n"
    "| --- | --- | --- | --- | --- |"
)
STATUS_WORDS = {
    "verified": "Verified",
    "known_issue": "Known issue",
    "python_tested": "Python-tested",
    "unresolved": "Unresolved",
    "none": "---",
}


def load_entries() -> list[dict[str, Any]]:
    return yaml.safe_load(YAML_PATH.read_text())


def _benchmark_md(entry: dict[str, Any]) -> str:
    benchmark = entry.get("benchmark") or {}
    cell = benchmark.get("name", "---")
    if benchmark.get("path"):
        cell += f" (`{benchmark['path']}`)"
    if benchmark.get("extra"):
        cell += benchmark["extra"]
    return cell


def _target_md(target: dict[str, str]) -> str:
    if target.get("desc"):
        return f"{target['software']} ({target['desc']})"
    cell = f"{target['software']} (`{target['func']}`"
    if target.get("note"):
        cell += f", {target['note']}"
    return cell + ")"


def _status_md(entry: dict[str, Any]) -> str:
    word = STATUS_WORDS[entry["status"]]
    if entry["status"] == "verified" and entry.get("tolerance"):
        return f"**{word}** ({entry['tolerance']})"
    return f"**{word}**"


def render_markdown(entries: list[dict[str, Any]]) -> str:
    """Render the benchmarks.md section 2.1 table (verified entries only)."""
    lines = [MD_HEADER]
    for entry in entries:
        if entry["status"] != "verified":
            continue
        models = ", ".join(f"`{model}`" for model in entry["models"])
        targets = ", ".join(_target_md(target) for target in entry["targets"])
        lines.append(f"| **{entry['family']}** | {models} | {targets} | {_benchmark_md(entry)} | {_status_md(entry)} |")
    return "\n".join(lines)


def _tex_escape(text: str) -> str:
    for char, repl in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_")):
        text = text.replace(char, repl)
    return text


def _tex_code(text: str) -> str:
    """Turn `code` spans into \\texttt{}; $...$ math passes through as LaTeX."""
    return re.sub(r"`([^`]+)`", lambda match: f"\\texttt{{{_tex_escape(match.group(1))}}}", text)


def _benchmark_tex(entry: dict[str, Any]) -> str:
    benchmark = entry.get("benchmark") or {}
    if benchmark.get("tex"):
        return benchmark["tex"]
    return _tex_escape(benchmark["name"]) if benchmark else "---"


def _target_tex(target: dict[str, str]) -> str:
    if target.get("desc"):
        cell = f"{target['software']}: {_tex_escape(target['desc'])}"
    else:
        cell = f"{target['software']}: \\texttt{{{_tex_escape(target['func'])}}}"
    if target.get("note"):
        # Single-token notes are code identifiers (e.g. gk2015_iv); prose notes are not.
        if re.fullmatch(r"\S+", target["note"]):
            cell += f" (\\texttt{{{_tex_escape(target['note'])}}})"
        else:
            cell += f" ({_tex_escape(target['note'])})"
    return cell


def _model_tex(entry: dict[str, Any]) -> str:
    if entry.get("label"):
        return _tex_escape(entry["label"])
    return ", ".join(f"\\texttt{{{model}}}" for model in entry["models"])


def _status_tex(entry: dict[str, Any]) -> str:
    word = STATUS_WORDS[entry["status"]]
    if entry["status"] == "verified" and entry.get("tolerance"):
        return f"{word} ({_tex_code(entry['tolerance'])})"
    return word


def render_tex(entries: list[dict[str, Any]]) -> str:
    """Render the LaTeX body rows of the paper's Table 1 (all entries)."""
    lines = []
    previous_family = None
    for entry in entries:
        if previous_family is not None and entry["family"] != previous_family:
            lines.append("    \\midrule")
        family = _tex_escape(entry["family"]) if entry["family"] != previous_family else ""
        row = f"    {family} & {_model_tex(entry)} & {_benchmark_tex(entry)} & {'; '.join(_target_tex(t) for t in entry['targets']) or '---'} & {_status_tex(entry)} \\\\"
        lines.append(row)
        previous_family = entry["family"]
    return "\n".join(lines)


def update_benchmarks(entries: list[dict[str, Any]]) -> None:
    text = BENCHMARKS_PATH.read_text()
    start = text.index(BEGIN_MARK)
    end = text.index(END_MARK) + len(END_MARK)
    text = text[:start] + BEGIN_MARK + "\n" + render_markdown(entries) + "\n" + END_MARK + text[end:]
    BENCHMARKS_PATH.write_text(text)
    print(f"Rewrote {BENCHMARKS_PATH} generated section")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex-out", metavar="PATH", help="write the LaTeX Table 1 rows to PATH instead of stdout")
    args = parser.parse_args()
    entries = load_entries()
    update_benchmarks(entries)
    tex = render_tex(entries)
    if args.tex_out:
        Path(args.tex_out).write_text(tex + "\n")
        print(f"Wrote LaTeX Table 1 rows to {args.tex_out}")
    else:
        print(tex)


if __name__ == "__main__":
    main()