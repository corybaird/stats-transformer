from dataclasses import dataclass

@dataclass
class SvarsUsaBenchmark:
    """R svars::USA dataset, svars JSS vignette, changes-in-volatility SVAR."""
    # VAR(p=6, type="const") baseline
    var_p6_loglevel = -564.30
    
    # id.cv(SB=c(1979,3)) structural identification
    lambda_diag = (0.393, 0.192, 1.244)
    
    # Wald test: lambda1 = lambda3
    wald_lambda1_eq_lambda3_stat = 7.66
    wald_lambda1_eq_lambda3_pvalue = 0.01
    
    # Restricted (over-identified) model
    restricted_lr_stat = 8.734
    restricted_lr_pvalue = 0.033
