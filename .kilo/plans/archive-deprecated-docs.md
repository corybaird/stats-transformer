# Plan: Archive Deprecated Docs & Improve Gitignore

## Goal
Archive `planned_extensions`, `research_standards` docs, and `agentic_examples.md` to `docs/archive/`, add clear comments to `.gitignore`, and work on branch `refactor/docs-gitignore`.

## Files to Archive

Move these to `docs/archive/`:
1. `docs/planned_extensions/` (entire folder - 7 files)
2. `docs/research_standards/` (entire folder - 2 files)
3. `docs/validation/agentic_examples.md` (single file)

## Steps

### 1. Create branch
```bash
git checkout -b refactor/docs-gitignore
```

### 2. Create `docs/archive/` and move content
```bash
mkdir -p docs/archive
git mv docs/planned_extensions docs/archive/
git mv docs/research_standards docs/archive/
git mv docs/validation/agentic_examples.md docs/archive/
```

### 3. Update `.gitignore`

**Add new entry for docs/archive:**
```gitignore
# Archived/deprecated documentation (not tracked, preserved locally)
docs/archive/
```

**Add clarifying comments to existing entries that lack them:**
- `src/stats_transformer/visualization/expansion/` - generated visualization outputs
- `references/dictionaries/` - generated dictionary mappings
- `references/viz/` - generated visualization configs
- `archive/` - local archive of old project materials
- `src/examples/agent_reviews/` - generated/example review files
- `/data/notebook_exports/` - notebook export outputs
- `/data/pipeline/` - DVC pipeline intermediates
- `/data/final/` - final processed data outputs
- `/data/temp/` - temporary working files
- `/data/raw/` - immutable raw data (never modified)
- `reports/` - generated analysis reports and plots

### 4. Commit (Multiple Atomic Commits)

**Commit 1 - Archive deprecated documentation:**
```bash
git add -A
git commit -m "ARCHIVE move planned_extensions and research_standards to docs/archive"
```

**Commit 2 - Update gitignore with clear comments:**
```bash
git add .gitignore
git commit -m "DOCS add clarifying comments to gitignore entries"
```

## Verification
- [ ] `docs/archive/` contains `planned_extensions/`, `research_standards/`, and `agentic_examples.md`
- [ ] Original locations no longer exist
- [ ] `docs/archive/` is gitignored
- [ ] `.gitignore` has clear comments explaining purpose of entries
- [ ] 2 separate commits exist (archive move + gitignore update)