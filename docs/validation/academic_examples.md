# Academic and numerical validation

## Contents

1. [Evidence ladder](#evidence-ladder)
2. [Direct estimator comparisons](#direct-estimator-comparisons)
3. [Academic feature-transformation examples](#academic-feature-transformation-examples)
4. [Structural VAR and local-projection demonstrations](#structural-var-and-local-projection-demonstrations)
5. [MATLAB comparison](#matlab-comparison)
6. [Reporting validation claims](#reporting-validation-claims)

Validation in this repository has three distinct levels. They should not be conflated.

1. **Automated tests** establish that supported interfaces execute and return expected structures on controlled data.
2. **Direct Python comparisons** check wrapper parity with an underlying Python estimator on the same prepared data.
3. **Cross-language comparisons** compare a defined numerical object with an external implementation such as MATLAB VAR-Toolbox.

A script that runs successfully is an executable demonstration. It becomes a replication claim only when its data, transformations, estimator specification, and numerical outputs are compared with a stated benchmark.

## Direct estimator comparisons

`src/examples/timeseries/macro_var.py` constructs $100\Delta\log(x_t)$ for real GDP, real consumption, and real investment from the quarterly `statsmodels` macroeconomic dataset. It estimates the same VAR(2) with a constant directly through `statsmodels` and through `VARModel`.

```bash
/opt/homebrew/bin/uv run python -m src.examples.timeseries.macro_var
```

The example compares coefficient and standard-error arrays. It does not validate structural identification or impulse-response confidence intervals.

## Academic feature-transformation examples

The following scripts compare `FeatureEngineer` transformations with paper-specific processing logic or supplied replication outputs:

| Example | Script | Comparison target | Scope |
| --- | --- | --- | --- |
| Nakamura--Steinsson (2018) | `src.examples.academic.nakamura_steinsson` | Supplied Stata `master.dta` output | Daily first difference of a nominal yield |
| Bauer--Swanson (2023) | `src.examples.academic.bauer_swanson` | MATLAB-inspired log-difference and lag logic | Monthly transformations |
| Bauer--Bernanke--Milstein (2023) | `src.examples.academic.bauer_bernanke_milstein` | Explicit Python difference and percentage-change calculations | Daily transformations |

These examples require the corresponding research data under `data/examples/academic/`; the data are not installed as a package dependency. Run an example only after obtaining data that may be redistributed under the relevant source terms:

```bash
/opt/homebrew/bin/uv run python -m src.examples.academic.nakamura_steinsson
/opt/homebrew/bin/uv run python -m src.examples.academic.bauer_swanson
/opt/homebrew/bin/uv run python -m src.examples.academic.bauer_bernanke_milstein
```

The scripts print sample overlap and maximum absolute differences. Record those values with the data version used. Do not treat historical outputs from a different dataset revision as a current validation result.

## Structural VAR and local-projection demonstrations

`src/examples/academic/var/` contains executable demonstrations based on bundled VAR-Toolbox data:

- `stock_watson_2001.py` for a reduced-form VAR.
- `blanchard_quah_1989.py` for long-run identification.
- `gertler_karadi_2015.py` for an external-instrument example.
- `jorda_taylor_2025.py` for an IV local-projection example.

The automated extension tests confirm that these classes can fit their bundled data. At present, only the Blanchard--Quah structural impact matrix has a documented MATLAB numerical comparison. The other scripts should be described as translations or demonstrations until their full variable ordering, controls, normalization, inference, and published outputs have been compared.

## MATLAB comparison

The [MATLAB comparator](matlab_comparator.md) runs the defined Blanchard--Quah impact-matrix comparison against a local checkout of VAR-Toolbox. It is opt-in because it requires MATLAB, MATLAB Engine for Python, and a local VAR-Toolbox path.

The associated MATLAB, Dynare, and VAR-Toolbox versions are recorded in the comparator's [Software environment](matlab_comparator.md#software-environment). Dynare is available in the recorded environment but is not currently used by this comparison.

## Reporting validation claims

When documenting a result, state all of the following:

- The input data and its version or source.
- The exact transformation and sample rules.
- The estimator, lag order, deterministic terms, and identification assumptions.
- The object compared, such as coefficients, standard errors, an impact matrix, or an impulse response.
- The numerical tolerance and observed discrepancy.

This standard keeps a machine-precision comparison distinct from a smoke test and a descriptive figure distinct from a causal estimate.
