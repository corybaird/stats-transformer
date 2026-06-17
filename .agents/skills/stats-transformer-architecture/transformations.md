# Transformations & DataMerger

The feature-engineering layer is two classes:

- `FeatureEngineer` — `src/stats_transformer/featurization/feature_engineering.py:14`. Owns data loading, resampling, and the `transformations` chain.
- `DataMerger` — `src/stats_transformer/featurization/data_merger.py:8`. Joins resampled frames into the master panel.

## `FeatureEngineer`

### Construction

```python
FeatureEngineer(params_path="params.yaml")
# or, programmatic:
FeatureEngineer(
    transformations=["log", "lag1", "zscore"],
    entity_column="country",
    date_column="date",
    period="annual",
    data_columns=["gdp_growth", "inflation"],
    country_list=["USA", "JPN"],
    start_date="2000-01-01",
)
```

If `params_path` is provided, all of the above are read from the YAML's `data.featurization` block (`feature_engineering.py:21`–`31`).

### API

- `load_data(path)` — `feature_engineering.py:87`. CSV or parquet only. Returns a DataFrame.
- `resample_dataset(df, dataset_config)` — `feature_engineering.py:401`. Resamples a single dataset to its configured frequency.
- `fit_transform(df)` — `feature_engineering.py:345`. Applies the `transformations` chain in order, returns a new DataFrame.
- `transform(df)` — `feature_engineering.py:300`. Apply a previously-fitted transformation set to new data.

### Transformation vocabulary

`fit_transform` dispatches on string names. The valid base list is at `feature_engineering.py:309`:

- `changepct` — period-over-period percent change.
- `changepct_yoy` — year-over-year percent change.
- `changeraw` — period-over-period raw (level) change.
- `rollingmean` — rolling mean window (configure window via suffix or config).
- `log` — natural log of `1 + x` (safe for zeros).
- `zscore` — z-score within entity (demean + scale).

`fit_transform` also recognizes `lag<N>` and `lead<N>` patterns (`feature_engineering.py:263`–`270` and `312`–`316`). Use these for `lag1`, `lag2`, `lead1`, etc.

### YAML `data.featurization`

```yaml
data:
  featurization:
    entity_column: country
    date_column: date
    period: annual                # annual | quarterly | monthly | daily
    transformations: [log, lag1, lag2, zscore]
    column_mapping:               # optional: rename raw columns
      gdp: gdp_growth
    country_list: [USA, JPN, DEU]
    start_date: "2000-01-01"
    data_columns: [gdp_growth, inflation]
    output_path: data/final/features.csv
```

### Resampling & frequency alignment

`FeatureEngineer` uses these constants (`feature_engineering.py:9`–`12`):

- `SUFFIX_TO_MONTHS = {"Y": 12, "Q": 3, "M": 1, "D": 1}` — pandas-style frequency suffix → months.
- `TARGET_TO_MONTHS = {"annual": 12, "quarterly": 3, "monthly": 1, "daily": 1/30}` — period name → months.
- `PERIOD_TO_SUFFIX = {"daily": "D", "monthly": "M", "quarterly": "Q", "annual": "Y"}` — period name → pandas suffix.
- `FREQ_ORDER = {"daily": 0, "monthly": 1, "quarterly": 2, "annual": 3}` — ordering for upsampling.

`resample_dataset` is the per-dataset entry point. It honors `dataset_config.frequency` (e.g. `Q` for quarterly) and aligns upward to the target period specified by `period` in the featurization config.

## `DataMerger`

Source: `src/stats_transformer/featurization/data_merger.py:8`.

```python
DataMerger(params_path="params.yaml")
dm.merge(df_left, df_right, on=["country", "date"], how="outer")
```

`merge` is a thin wrapper around `pandas.merge` that:

1. Joins on the given `on` columns (defaults to `["country", "date"]`).
2. Logs left/right/merged row counts and entity counts (`data_merger.py:34`).
3. Warns if any entities from the left side are dropped.

YAML keys consumed:

```yaml
data:
  merge:
    on: [country, date]
    how: outer                # outer | inner | left | right
    output_path: data/pipeline/resampled_merged.parquet
```

## Adding a new transformation

1. Add the new string to `valid_base_transformations` (`feature_engineering.py:309`) if it is a base (non-`lag`/`lead`) transform.
2. Add the dispatch branch in `fit_transform` alongside the existing `if "log"`, `if "zscore"` checks.
3. If the transform is parameterized (e.g. window size), document the suffix convention (e.g. `rollingmean_3`) in the dispatch.
4. Update the transformation vocabulary section above.
5. Add a smoke test under `tests/` (mirroring the test fixture `tests/data/test_data.csv` referenced by `references/configs/test_pipeline.yaml:5`).

## Adding a new join strategy

`DataMerger.merge` is intentionally thin. If you need a non-pandas join (e.g. temporal tolerance), subclass `DataMerger` or add a sibling class in `featurization/` rather than overloading `merge`.
