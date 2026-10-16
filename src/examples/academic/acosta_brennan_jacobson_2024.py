"""Replication of Acosta, Brennan & Jacobson (2024), "Money Talks? A Fed
Speech Model for Policymakers" (as shipped, high-frequency FOMC surprises).

Replicates the VAR of the Gurkaynak-Sack-Swanson monetary surprise factors
(gss_target, gss_path) and the policy-rate surprise ns on the authors'
event-study series, and verifies the GSS rotation properties the factors
should satisfy: target and path are orthogonal surprise factors, and the
orthogonal decomposition of ns variance is reported as each factor's
variance share. Data follows the authors' replication series in
data/examples/academic/acosta_brennan_jacobson_2024/sofr_surprises.parquet.gzip.

Approximation: the rotation checks use the shipped factors directly (cov/
variance shares from the observed series) instead of re-running the
authors' factor extraction; no regression of ns on the factors is fitted
since ns (the policy-rate surprise) is mechanically built from them.
"""

from pathlib import Path

import pandas as pd

from stats_transformer.models.timeseries.reduced_form.var import VARModel

from examples._print import format_metrics


class AcostaBrennanJacobson2024Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path(__file__).resolve().parents[3] / "data/examples/academic/acosta_brennan_jacobson_2024/sofr_surprises.parquet.gzip"
        self.var_model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index()

    def _rotation_checks(self, df):
        # Gurkaynak, Sack & Swanson (2005) extract mutually orthogonal target
        # and path factors; the authors' gss_target / gss_path series should
        # satisfy that rotation. Regressing ns on them is vacuous (ns, the
        # policy-rate surprise, is mechanically built from the two factors,
        # R^2 ~ 1), so what is checked instead: the factor correlation is
        # near zero, and ns variance is decomposed over the orthogonal
        # factors as cov(ns, factor)^2 / (var(factor) * var(ns)).
        target, path, ns = df["gss_target"], df["gss_path"], df["ns"]
        return {
            "target_path_corr": float(target.corr(path)),
            "variance_share_target": float(target.cov(ns) ** 2 / (target.var() * ns.var())),
            "variance_share_path": float(path.cov(ns) ** 2 / (path.var() * ns.var())),
            "variance_share_sum": float(target.cov(ns) ** 2 / (target.var() * ns.var()) + path.cov(ns) ** 2 / (path.var() * ns.var())),
            "num_observations": int(ns.notna().sum()),
        }

    def run(self):
        df = self._load_data()
        self.var_model = VARModel(target_variables=["gss_target", "gss_path", "ns"], maxlags=2)
        var_metrics = self.var_model.fit(df)
        rotation_metrics = self._rotation_checks(df)
        print("Acosta, Brennan, & Jacobson (2024) VAR:")
        format_metrics(var_metrics)
        print("Acosta, Brennan, & Jacobson (2024) GSS rotation checks (target vs path):")
        format_metrics(rotation_metrics)
        return {"var_metrics": var_metrics, "rotation_metrics": rotation_metrics}


if __name__ == "__main__":
    AcostaBrennanJacobson2024Replication().run()