# Repository structure

The repository follows a Cookiecutter Data Science-style separation between reusable library code, empirical inputs, executable examples, tests, configuration, and generated outputs.

## Contents

1. [Repository tree](#repository-tree)
2. [Core package](#core-package)
3. [Configuration and artifacts](#configuration-and-artifacts)
4. [Validation boundary](#validation-boundary)
5. [Agent skill](#agent-skill)
6. [Related documentation](#related-documentation)

```text
.
├── .agents/                       # project instructions and agent skills
├── data/
│   ├── examples/                  # bundled and validation example data
│   ├── final/                     # engineered analysis-ready data
│   ├── pipeline/                  # resampled and merged artifacts
│   ├── raw/                       # source data
│   └── temp/                      # disposable local data
├── docs/
│   ├── archive/                   # historical planning and refactoring records
│   ├── library/                   # architecture and repository documentation
│   └── validation/                # test, replication, and comparator guides
├── models/                        # serialized model artifacts when produced
├── notebooks/                     # exploratory demonstrations
├── references/
│   ├── configs/                   # YAML specifications
│   └── dictionaries/              # constant mappings
├── reports/
│   ├── figures/                   # generated visual outputs
│   ├── tables/                    # generated table outputs
│   └── visualizations/            # EDA and model visualizations
├── src/
│   ├── examples/                  # executable demonstrations and comparisons
│   ├── stats_transformer/         # installable Python package
│   └── temp/                      # disposable development scripts
└── tests/
    ├── verification/              # cross-language verification utilities
    └── test_*.py                  # automated unit and integration tests
```

## Core package

`src/stats_transformer/` contains reusable code only. It is organized by capability:

- `featurization/` for transformation, resampling, merging, and event-study construction.
- `models/` for regression, discrete, unsupervised, and time-series implementations.
- `visualization/` for reusable charts, EDA, model visualizers, styling, and tables.
- `reporting/` for export helpers.
- `data/` for packaged example data and panel utilities.
- `pipeline.py` for YAML-driven orchestration.

Paper-specific logic and data should remain in an example, notebook, or downstream research project rather than in the installable package.

## Configuration and artifacts

`references/configs/` is the control center for reproducible YAML specifications. Configurations select raw data, transformation settings, model options, and output paths. The pipeline writes generated artifacts to `data/pipeline/`, `data/final/`, and `reports/` rather than modifying source data.

## Validation boundary

Automated tests belong in `tests/`. Cross-language checks that require local software, data, or licensing belong in `tests/verification/` and are documented separately. Academic examples under `src/examples/academic/` are executable demonstrations, not substitutes for the automated test suite.

## Agent skill

The project-specific architecture skill is stored in `.agents/skills/stats-transformer-architecture/`. It gives compatible coding agents the same file-routing and extension conventions described here. Install or refresh tool-specific copies with `scripts/install-agent-skill.sh`.
