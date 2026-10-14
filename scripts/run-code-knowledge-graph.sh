#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tool_name="code-review-graph"
tool_version="2.3.9"
graph_dir="$repo_root/.code-review-graph"

usage() {
  printf 'Usage: %s build|update|status|clean|install-mcp <claude|codex|cursor|...>\n' "$0" >&2
  exit 2
}

run_tool() {
  if command -v uvx >/dev/null 2>&1; then
    (cd "$repo_root" && uvx --from "$tool_name==$tool_version" "$tool_name" "$@")
  elif command -v "$tool_name" >/dev/null 2>&1; then
    (cd "$repo_root" && "$tool_name" "$@")
  else
    printf 'Need uvx (https://docs.astral.sh/uv/) or an installed %s\n' "$tool_name" >&2
    exit 1
  fi
}

[ "$#" -ge 1 ] || usage

case "$1" in
  build)
    run_tool build
    ;;
  update)
    run_tool update
    ;;
  status)
    run_tool status
    if [ -d "$graph_dir" ]; then
      du -sh "$graph_dir"
    fi
    ;;
  clean)
    rm -rf "$graph_dir"
    printf 'Removed %s\n' "$graph_dir"
    ;;
  install-mcp)
    [ "$#" -eq 2 ] || usage
    run_tool install --platform "$2"
    ;;
  *)
    usage
    ;;
esac
