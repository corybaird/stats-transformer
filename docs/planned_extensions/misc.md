# Miscellaneous Planned Extensions

## Granger Causality Tester

A generalized `GrangerCausalityTester` should be implemented in the future to abstract away the paper-specific setup used for Granger causality testing (e.g., resampling, z-scoring, and running region-by-region checks).

Currently, this logic lives in paper-specific pipelines (such as `notebooks.paper_econsurveys.modules.utils.granger_causality_tester.py`), but the underlying pattern of formatting panels for `statsmodels.tsa.stattools.grangercausalitytests` could be generalized as an extension to `stats_transformer.models.timeseries`.
