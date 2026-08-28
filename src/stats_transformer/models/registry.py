from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.iv import IV2SLSModel
from stats_transformer.models.regression.panel_iv import PanelIV2SLSModel
from stats_transformer.models.regression.gmm import GMMModel
from stats_transformer.models.regression.did import DiDModel
from stats_transformer.models.regression.spec_runner import SpecificationRunner
from stats_transformer.models.discrete.logit import LogitModel
from stats_transformer.models.discrete.probit import ProbitModel
from stats_transformer.models.unsupervised.unsupervised import PCAModel, KMeansModel
from stats_transformer.models.timeseries import (
    BlanchardQuahModel,
    DynamicFactorModel,
    BVARModel,
    LocalProjectionsModel,
    LocalProjectionsIVModel,
    ProxySVARModel,
    SignZeroSVARModel,
    VolatilitySVARModel,
    IndependenceSVARModel,
    CVMSVARModel,
    NonGaussianSVARModel,
    SVARModel,
    VARModel,
    VECMModel,
    SVECModel,
    TVARModel,
    TVECMModel,
    STVARModel,
    ARIMAModel,
)

MODEL_REGISTRY = {
    # regression models
    "ols": {"cls": RegressionModel, "kind": "single_equation"}, 
    "robust_ols": {"cls": RobustOLSModel, "kind": "single_equation"}, 
    "arima": {"cls": ARIMAModel, "kind": "svar_family"}, 
    "panel_ols": {"cls": PanelRegressionModel, "kind": "panel"}, 
    "iv": {"cls": IV2SLSModel, "kind": "iv"}, 
    "panel_iv": {"cls": PanelIV2SLSModel, "kind": "panel_iv"}, 
    "gmm": {"cls": GMMModel, "kind": "iv"}, 
    "did": {"cls": DiDModel, "kind": "did"}, 
    "spec_runner": {"cls": SpecificationRunner, "kind": "spec_runner"},
    # discrete models
    "logit": {"cls": LogitModel, "kind": "single_equation"}, 
    "probit": {"cls": ProbitModel, "kind": "single_equation"},
    # unsupervised models
    "pca": {"cls": PCAModel, "kind": "unsupervised"},
    "kmeans": {"cls": KMeansModel, "kind": "unsupervised"},
    # timeseries models
    "var": {"cls": VARModel, "kind": "svar_family"},
    "vecm": {"cls": VECMModel, "kind": "svar_family"},
    "svec": {"cls": SVECModel, "kind": "svar_family"},
    "svar": {"cls": SVARModel, "kind": "svar_family"},
    "blanchard_quah": {"cls": BlanchardQuahModel, "kind": "svar_family"},
    "proxy_svar": {"cls": ProxySVARModel, "kind": "svar_family"},
    "sign_restrictions": {"cls": SignZeroSVARModel, "kind": "svar_family"},
    "volatility_svar": {"cls": VolatilitySVARModel, "kind": "svar_family"},
    "independence_svar": {"cls": IndependenceSVARModel, "kind": "svar_family"},
    "cvm_svar": {"cls": CVMSVARModel, "kind": "svar_family"},
    "non_gaussian_svar": {"cls": NonGaussianSVARModel, "kind": "svar_family"},
    "dynamic_factor": {"cls": DynamicFactorModel, "kind": "svar_family"},
    "bvar": {"cls": BVARModel, "kind": "svar_family"},
    "tvar": {"cls": TVARModel, "kind": "svar_family"},
    "tvecm": {"cls": TVECMModel, "kind": "svar_family"},
    "stvar": {"cls": STVARModel, "kind": "svar_family"},
    "local_projections": {"cls": LocalProjectionsModel, "kind": "lp"},
    "lp_iv": {"cls": LocalProjectionsIVModel, "kind": "lp_iv"},
}

MODEL_TYPE_ALIASES = {
    "iv_2sls": "iv",
    "2sls": "iv",
    "sign_zero": "sign_restrictions",
    "cvm": "cvm_svar",
    "non_gaussian": "non_gaussian_svar",
}

# ModelBase subclasses that are intentionally not reachable via Pipeline's
# model_type dispatch (e.g. abstract bases, or models requiring construction
# paths the pipeline does not support). Checked by the registry-completeness
# test so a newly added model is never silently unreachable by omission.
NOT_PIPELINE_EXPOSED = {
    "UnsupervisedModel",  # abstract base for PCAModel/KMeansModel; not directly instantiable
}
