#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
skill_name="stats-transformer-architecture"
source_dir="$repo_root/.agents/skills/$skill_name"

if [ ! -f "$source_dir/SKILL.md" ]; then
  printf 'Missing canonical skill at %s\n' "$source_dir/SKILL.md" >&2
  exit 1
fi

usage() {
  printf 'Usage: %s claude|codex|kilo|all\n' "$0" >&2
  exit 2
}

install_one() {
  tool="$1"

  case "$tool" in
    claude) target_root=".claude" ;;
    codex) target_root=".codex" ;;
    kilo) target_root=".kilo" ;;
    *) usage ;;
  esac

  target_dir="$repo_root/$target_root/skills/$skill_name"
  mkdir -p "$target_dir"
  cp "$source_dir"/SKILL.md "$target_dir"/SKILL.md
  cp "$source_dir"/architecture.md "$target_dir"/architecture.md
  cp "$source_dir"/models.md "$target_dir"/models.md
  cp "$source_dir"/pipeline-stages.md "$target_dir"/pipeline-stages.md
  cp "$source_dir"/transformations.md "$target_dir"/transformations.md

  if [ "$tool" = "claude" ]; then
    tmp_file=$(mktemp)
    awk '
      NR == 1 && $0 == "---" {
        print
        next
      }
      $0 == "---" && inserted != 1 {
        print "paths:"
        print "  - src/stats_transformer/**"
        print "  - references/configs/**"
        print "  - dvc.yaml"
        print "  - params.yaml"
        inserted = 1
        print
        next
      }
      { print }
    ' "$target_dir/SKILL.md" > "$tmp_file"
    mv "$tmp_file" "$target_dir/SKILL.md"
  fi

  printf 'Installed %s for %s at %s\n' "$skill_name" "$tool" "$target_dir"
}

[ "$#" -eq 1 ] || usage

case "$1" in
  all)
    install_one claude
    install_one codex
    install_one kilo
    ;;
  claude | codex | kilo)
    install_one "$1"
    ;;
  *)
    usage
    ;;
esac
