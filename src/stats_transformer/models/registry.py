from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.iv import IV2SLSModel
from stats_transformer.models.regression.panel_iv import PanelIV2SLSModel
from stats_transformer.models.discrete.logit import LogitModel
from stats_transformer.models.unsupervised.unsupervised import PCAModel, KMeansModel
from stats_transformer.models.timeseries import (
    BlanchardQuahModel,
    LocalProjectionsIVModel,
    ProxySVARModel,
    SignZeroSVARModel,
    VolatilitySVARModel,
    IndependenceSVARModel,
    SVARModel,
    VARModel,
    VECMModel,
)

MODEL_REGISTRY = {
    "ols": {"cls": RegressionModel, "kind": "single_equation"},
    "robust_ols": {"cls": RobustOLSModel, "kind": "single_equation"},
    "logit": {"cls": LogitModel, "kind": "single_equation"},
    "panel_ols": {"cls": PanelRegressionModel, "kind": "panel"},
    "iv": {"cls": IV2SLSModel, "kind": "iv"},
    "panel_iv": {"cls": PanelIV2SLSModel, "kind": "panel_iv"},
    "pca": {"cls": PCAModel, "kind": "unsupervised"},
    "kmeans": {"cls": KMeansModel, "kind": "unsupervised"},
    "var": {"cls": VARModel, "kind": "svar_family"},
    "vecm": {"cls": VECMModel, "kind": "svar_family"},
    "svar": {"cls": SVARModel, "kind": "svar_family"},
    "blanchard_quah": {"cls": BlanchardQuahModel, "kind": "svar_family"},
    "proxy_svar": {"cls": ProxySVARModel, "kind": "svar_family"},
    "sign_restrictions": {"cls": SignZeroSVARModel, "kind": "svar_family"},
    "volatility_svar": {"cls": VolatilitySVARModel, "kind": "svar_family"},
    "independence_svar": {"cls": IndependenceSVARModel, "kind": "svar_family"},
    "lp_iv": {"cls": LocalProjectionsIVModel, "kind": "lp_iv"},
}

MODEL_TYPE_ALIASES = {
    "iv_2sls": "iv",
    "2sls": "iv",
    "sign_zero": "sign_restrictions",
}

# ModelBase subclasses that are intentionally not reachable via Pipeline's
# model_type dispatch (e.g. abstract bases, or models requiring construction
# paths the pipeline does not support). Checked by the registry-completeness
# test so a newly added model is never silently unreachable by omission.
NOT_PIPELINE_EXPOSED = {
    "UnsupervisedModel",  # abstract base for PCAModel/KMeansModel; not directly instantiable
    "LocalProjectionsModel",  # non-IV local projections; no pipeline entry point defined yet
}
