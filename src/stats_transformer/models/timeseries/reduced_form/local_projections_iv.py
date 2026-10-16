import pandas as pd
from linearmodels.iv import IV2SLS
from stats_transformer.models.base import ModelBase

class LocalProjectionsIVModel(ModelBase):

    """Local projections model with an instrumental variable."""
    def __init__(self, target_variable=None, shock_variable=None, instrument_variable=None, control_variables=None, horizons=10, date_column=None, cumulative=False, multiplier=False, lags=0, cov_type="nonrobust", **kwargs):
        if bool(cumulative) and bool(multiplier):
            raise ValueError("cumulative and multiplier are mutually exclusive")
        controls = control_variables or []
        super().__init__(target=target_variable or "dummy", independent_variables=[shock_variable or "dummy"] + controls, **kwargs)
        self.target_variable = target_variable
        self.shock_variable = shock_variable
        self.instrument_variable = instrument_variable
        self.control_variables = controls
        self.horizons = horizons
        self.date_column = date_column
        self.time_column = date_column
        self.cumulative = cumulative
        self.multiplier = multiplier
        self.lags = int(lags)
        self.cov_type = cov_type
        self.irf_coefficients = []
        self.irf_std_errors = []

    def _get_required_columns(self):
        cols = [self.target_variable, self.shock_variable, self.instrument_variable] + list(self.control_variables)
        if self.date_column:
            cols.append(self.date_column)
        return list(dict.fromkeys(cols))

    def build_model(self):
        """Build the underlying model object from the loaded data."""
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        self._estimate_lp_iv()
        return self

    def _build_exog(self, df):
        """Exogenous regressors shared across horizons: controls, optional lag
        blocks (target, shock, instrument, controls), and a constant."""
        exog = pd.DataFrame(index=df.index)
        for c in self.control_variables:
            exog[c] = df[c]
        for lag in range(1, self.lags + 1):
            for col in [self.target_variable, self.shock_variable, self.instrument_variable] + list(self.control_variables):
                exog[f"{col}_lag{lag}"] = df[col].shift(lag)
        exog["const"] = 1.0
        return exog

    def _outcome_and_endog(self, df, h):
        """Dependent variable and endogenous treatment at horizon h."""
        target, shock = self.target_variable, self.shock_variable
        if self.multiplier:
            # Ramey-Zubairy (2018) integral multiplier: sum of y and x over j=0..h
            y_h = df[target].shift(0)
            x_h = df[shock].shift(0)
            for j in range(1, h + 1):
                y_h = y_h + df[target].shift(-j)
                x_h = x_h + df[shock].shift(-j)
            return y_h, x_h.rename(shock)
        if self.cumulative:
            return df[target].shift(-h) - df[target].shift(1), df[shock]
        return df[target].shift(-h), df[shock]

    def _fit(self, y_h, exog, endog, instrument, h):
        cov_type_norm = self.cov_type.upper() if isinstance(self.cov_type, str) else str(self.cov_type)
        if cov_type_norm in ("HAC", "NW", "NEWEY_WEST", "NEWEYWEST"):
            return IV2SLS(y_h, exog, endog, instrument).fit(cov_type="kernel", kernel="bartlett", bandwidth=h + 1)
        if cov_type_norm == "HC3":
            raise ValueError("HC3 is not supported for IV; use HC1 or HAC")
        if cov_type_norm in ("HC1", "ROBUST"):
            return IV2SLS(y_h, exog, endog, instrument).fit(cov_type="robust", debiased=True)
        return IV2SLS(y_h, exog, endog, instrument).fit(cov_type="unadjusted", debiased=True)

    def _estimate_lp_iv(self):
        self.irf_coefficients = []
        self.irf_std_errors = []
        df = self.df_clean
        exog = self._build_exog(df)
        for h in range(self.horizons + 1):
            y_h, endog = self._outcome_and_endog(df, h)
            data = pd.concat([y_h.rename("__outcome"), exog, endog, df[[self.instrument_variable]]], axis=1).dropna()
            res = self._fit(data["__outcome"], data[exog.columns], data[[self.shock_variable]], data[[self.instrument_variable]], h)
            self.irf_coefficients.append(float(res.params[self.shock_variable]))
            self.irf_std_errors.append(float(res.std_errors[self.shock_variable]))

    def get_summary(self):
        """Return a text summary of the fitted model."""
        return f"Local Projections IV Model\nHorizons: {self.horizons}\nIRF Coefficients: {self.irf_coefficients}"

    def get_model_metrics(self):
        """Return key fit metrics as a dictionary."""
        return {
            "horizons": self.horizons,
            "irf_h0": self.irf_coefficients[0] if self.irf_coefficients else None,
            "irf_h_max": self.irf_coefficients[-1] if self.irf_coefficients else None
        }

    def run(self, data):
        """Run the full workflow end to end."""
        self.fit(data)
        return self.get_model_metadata()
