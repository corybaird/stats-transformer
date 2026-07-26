from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr

from stats_transformer.models.timeseries.decompositions import TimeSeriesDecompositions
from stats_transformer.reporting.timeseries.results import TimeSeriesReportData


def _build_specification_frame(model_name, result, variables, identification):
    rows = [
        ["model", model_name],
        ["variables", ", ".join(variables)],
        ["observations", int(result.nobs)],
        ["lag_order", int(result.k_ar)],
        ["identification", identification],
    ]
    for metric in ["aic", "bic", "hqic", "fpe"]:
        value = getattr(result, metric, None)
        if value is not None:
            rows.append([metric, float(value)])
    return pd.DataFrame(rows, columns=["statistic", "value"])


def _build_coefficient_frame(result):
    rows = []
    equations = list(result.params.columns)
    for equation in equations:
        for term in result.params.index:
            rows.append([
                equation,
                term,
                float(result.params.loc[term, equation]),
                float(result.stderr.loc[term, equation]),
                float(result.tvalues.loc[term, equation]),
                float(result.pvalues.loc[term, equation]),
            ])
    return pd.DataFrame(rows, columns=["equation", "term", "estimate", "std_error", "statistic", "p_value"])


def _build_irf_dataset(result, impact, variables, horizons):
    responses = result.ma_rep(maxn=horizons)
    structural_responses = np.einsum("hij,jk->hik", responses, impact)
    da = xr.DataArray(
        structural_responses,
        coords=[np.arange(structural_responses.shape[0]), variables, variables],
        dims=["horizon", "response", "shock"],
        name="estimate"
    )
    return da.to_dataset()


def _build_fevd_dataset(result, impact, variables, horizons):
    values = TimeSeriesDecompositions(result, B_0=impact).compute_fevd(steps=horizons + 1)
    da = xr.DataArray(
        values,
        coords=[np.arange(values.shape[0]), variables, variables],
        dims=["horizon", "response", "shock"],
        name="share"
    )
    return da.to_dataset()


def _build_decomposition_datasets(model, result, impact, variables):
    historical, shocks = TimeSeriesDecompositions(result, B_0=impact).compute_hd()
    dates = _resolve_dates(model, result)
    
    historical_da = xr.DataArray(
        historical,
        coords=[dates, variables, variables],
        dims=["date", "response", "shock"],
        name="contribution"
    )
    reconstructed = historical.sum(axis=2)
    reconstructed_da = xr.DataArray(
        reconstructed,
        coords=[dates, variables],
        dims=["date", "response"],
        name="reconstructed"
    )
    historical_ds = xr.merge([historical_da, reconstructed_da])
    
    shocks_da = xr.DataArray(
        shocks,
        coords=[dates, variables],
        dims=["date", "shock"],
        name="value"
    )
    return historical_ds, shocks_da.to_dataset()


def _resolve_dates(model, result):
    residual_index = result.resid.index
    if model.date_column and model.date_column in model.df_clean.columns:
        return model.df_clean.loc[residual_index, model.date_column].tolist()
    return residual_index.tolist()


def _normal_95_interval(estimate, standard_error):
    critical_value = 1.959963984540054
    return estimate - critical_value * standard_error, estimate + critical_value * standard_error


class TimeSeriesResultAdapter(ABC):

    def __init__(self, model, horizons=20):
        self.model = model
        self.horizons = horizons

    @abstractmethod
    def build(self):
        pass


class VARResultAdapter(TimeSeriesResultAdapter):

    def build(self):
        result = self.model.model
        if result is None:
            raise ValueError("VARModel must be fitted before reporting")
        variables = list(self.model.target_variables)
        covariance = np.asarray(result.sigma_u)
        impact = np.linalg.cholesky(covariance)
        historical, shocks = _build_decomposition_datasets(self.model, result, impact, variables)
        return TimeSeriesReportData(
            specification=_build_specification_frame("VAR", result, variables, "recursive Cholesky"),
            coefficients=_build_coefficient_frame(result),
            irfs=_build_irf_dataset(result, impact, variables, self.horizons),
            fevd=_build_fevd_dataset(result, impact, variables, self.horizons),
            historical_decomposition=historical,
            structural_shocks=shocks,
        )


class BlanchardQuahResultAdapter(TimeSeriesResultAdapter):

    def build(self):
        result = self.model.var_result
        if result is None or self.model.B_0 is None:
            raise ValueError("BlanchardQuahModel must be fitted before reporting")
        variables = list(self.model.target_variables)
        impact = np.asarray(self.model.B_0)
        historical, shocks = _build_decomposition_datasets(self.model, result, impact, variables)
        return TimeSeriesReportData(
            specification=_build_specification_frame("Blanchard-Quah SVAR", result, variables, "long-run restrictions"),
            coefficients=_build_coefficient_frame(result),
            irfs=_build_irf_dataset(result, impact, variables, self.horizons),
            fevd=_build_fevd_dataset(result, impact, variables, self.horizons),
            historical_decomposition=historical,
            structural_shocks=shocks,
        )


class LocalProjectionsResultAdapter(TimeSeriesResultAdapter):

    def build(self):
        source = self.model.compute_irf()
        rows = []
        for row in source.itertuples(index=False):
            rows.append([int(row.horizon), self.model.target, self.model.shock_var, float(row.effect), float(row.stderr), float(row.lower_ci), float(row.upper_ci), float(row.pvalue)])
        
        irfs_df = pd.DataFrame(rows, columns=["horizon", "response", "shock", "estimate", "std_error", "lower", "upper", "p_value"])
        irfs_ds = irfs_df.set_index(["horizon", "response", "shock"]).to_xarray()
        
        specification = pd.DataFrame([
            ["model", "Local projections"],
            ["response", self.model.target],
            ["shock", self.model.shock_var],
            ["horizons", int(self.model.horizon)],
            ["covariance", "HC3"],
        ], columns=["statistic", "value"])
        return TimeSeriesReportData(specification=specification, irfs=irfs_ds)


class LocalProjectionsIVResultAdapter(TimeSeriesResultAdapter):

    def build(self):
        if not self.model.irf_coefficients:
            raise ValueError("LocalProjectionsIVModel must be fitted before reporting")
        rows = []
        for horizon, estimate in enumerate(self.model.irf_coefficients):
            standard_error = self.model.irf_std_errors[horizon]
            lower, upper = _normal_95_interval(estimate, standard_error)
            rows.append([horizon, self.model.target_variable, self.model.shock_variable, estimate, standard_error, lower, upper])
            
        irfs_df = pd.DataFrame(rows, columns=["horizon", "response", "shock", "estimate", "std_error", "lower", "upper"])
        irfs_ds = irfs_df.set_index(["horizon", "response", "shock"]).to_xarray()
        
        specification = pd.DataFrame([
            ["model", "Local projections IV"],
            ["response", self.model.target_variable],
            ["shock", self.model.shock_variable],
            ["instrument", self.model.instrument_variable],
            ["horizons", int(self.model.horizons)],
            ["confidence_interval", "normal approximation, 95%"],
        ], columns=["statistic", "value"])
        return TimeSeriesReportData(specification=specification, irfs=irfs_ds)
