import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase
from stats_transformer.models.timeseries.reduced_form.forecasting import VARForecaster

class SignZeroSVARModel(ModelBase):
    """
    SVAR Model with Sign, Zero, and Narrative restrictions.
    Evaluates configurations loaded from a YAML schema.
    """
    def __init__(self, target_variables=None, config_path=None, date_column=None, maxlags=1, max_draws=10000, required_accepts=100, **kwargs):
        target = target_variables[0] if target_variables else "dummy"
        indep = target_variables[1:] if target_variables and len(target_variables) > 1 else ["dummy"]
        super().__init__(target=target, independent_variables=indep, **kwargs)
        self.target_variables = target_variables or []
        self.config_path = config_path
        self.date_column = date_column
        self.time_column = date_column
        self.maxlags = maxlags
        self.max_draws = max_draws
        self.required_accepts = required_accepts
        
        self.var_result = None
        self.accepted_rotations = []
        self.accepted_irfs = []
        self.config = {}
        self.restrictions = []
        self.narrative_restrictions = []
        
        if self.config_path:
            self._load_config()

    def _load_config(self):
        """Loads and parses the YAML restriction schema."""
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.shocks = self.config.get("shocks", [])
        self.restrictions = self.config.get("restrictions", [])
        self.narrative_restrictions = self.config.get("narrative_restrictions", [])
        
        # Verify variable match
        conf_vars = self.config.get("variables", [])
        if conf_vars and set(conf_vars) != set(self.target_variables):
            raise ValueError(f"Config variables {conf_vars} do not match model variables {self.target_variables}")
            
    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def build_model(self):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
            
        self.y = self.df_clean[self.target_variables]
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        
        if self.restrictions or self.narrative_restrictions:
            self._draw_restrictions()
            
        return self.var_result

    def _draw_restrictions(self):
        """
        Samples orthogonal rotations and evaluates them against configured restrictions.
        """
        k = self.y.shape[1]
        sigma_u = self.var_result.sigma_u
        if type(sigma_u) == pd.DataFrame:
            sigma_u = sigma_u.values
        
        # Lower triangular Cholesky
        p_chol = np.linalg.cholesky(sigma_u)
        
        # We need MA representation (IRF matrices) to check horizon > 0 restrictions
        # Find the maximum horizon needed
        max_h = 0
        for r in self.restrictions:
            h = r.get("horizon", 0)
            if isinstance(h, list):
                max_h = max(max_h, h[1])
            else:
                max_h = max(max_h, h)
                
        # Get Phi matrices (MA representation) for unrotated VAR
        phis = self._get_ma_matrices(max_h + 1)
        
        self.accepted_rotations = []
        self.accepted_irfs = []
        
        draws = 0
        accepts = 0
        
        # Precompute variable and shock indices
        var_to_idx = {v: i for i, v in enumerate(self.target_variables)}
        shock_to_idx = {s: i for i, s in enumerate(self.shocks)}
        
        np.random.seed(42) # For reproducibility in the sampling
        
        while draws < self.max_draws and accepts < self.required_accepts:
            draws += 1
            # 1. Generate random orthogonal matrix (Haar measure via QR)
            W = np.random.normal(size=(k, k))
            Q, R = np.linalg.qr(W)
            # Normalize sign of diagonal of R to ensure uniform Haar measure
            Q = Q @ np.diag(np.sign(np.diag(R)))
            
            # Candidate impact matrix
            impact = p_chol @ Q
            
            # Candidate structural IRF matrices up to max_h
            # Theta_h = Phi_h @ Impact
            candidate_irfs = np.zeros((max_h + 1, k, k))
            for h in range(max_h + 1):
                candidate_irfs[h] = phis[h] @ impact
                
            # 2. Evaluate constraints
            valid = True
            for r in self.restrictions:
                s_idx = shock_to_idx[r["shock"]]
                v_idx = var_to_idx[r["response"]]
                r_type = r["type"]
                
                h_def = r.get("horizon", 0)
                horizons_to_check = []
                if isinstance(h_def, list):
                    horizons_to_check = list(range(h_def[0], h_def[1] + 1))
                else:
                    horizons_to_check = [h_def]
                    
                for h in horizons_to_check:
                    val = candidate_irfs[h, v_idx, s_idx]
                    
                    if r_type == "sign":
                        sign_val = r["value"]
                        if sign_val == "+" and val <= 0:
                            valid = False
                            break
                        elif sign_val == "-" and val >= 0:
                            valid = False
                            break
                    elif r_type == "zero":
                        # Due to random rotations, exact zeroes are impossible almost surely unless 
                        # generated via restricted subspace sampling (e.g. Arias et al 2018).
                        # For Phase 2 frequentist testing, we allow an epsilon tolerance or assert
                        # it requires the deterministic zero sampling method. 
                        # We use a strict tolerance here for now.
                        if abs(val) > 1e-10:
                            valid = False
                            break
                            
                if not valid:
                    break
                    
            if valid:
                # Need to check narrative restrictions if any
                if self.narrative_restrictions:
                    if "p_chol_inv" not in locals():
                        p_chol_inv = np.linalg.inv(p_chol)
                    valid = self._check_narrative(Q, p_chol_inv)
                    
            if valid:
                self.accepted_rotations.append(Q)
                self.accepted_irfs.append(candidate_irfs)
                accepts += 1
                
        self.total_draws = draws
        
    def _check_narrative(self, Q, p_chol_inv):
        """Checks narrative restrictions on the historical structural shocks."""
        # e_t = Q' P^{-1} u_t
        # self.var_result.resid is (T, K)
        # u_t is (K, T)
        u_t = self.var_result.resid.values.T if isinstance(self.var_result.resid, pd.DataFrame) else self.var_result.resid.T
        
        # e_t is (K, T). e_t = Q' @ p_chol_inv @ u_t
        e_t = Q.T @ p_chol_inv @ u_t
        
        dates = self.df_clean[self.date_column].values if self.date_column else np.arange(u_t.shape[1])
        date_to_idx = {d: i for i, d in enumerate(dates[-u_t.shape[1]:])}
        
        shock_to_idx = {s: i for i, s in enumerate(self.shocks)}
        
        for r in self.narrative_restrictions:
            s_idx = shock_to_idx[r["shock"]]
            target_date = r["date"]
            
            # Allow strings to match numpy datetime64 or strings
            if target_date not in date_to_idx:
                # Try to parse or find the closest match, but for now exact match
                # Convert date to whatever format the index is in
                if isinstance(dates[0], np.datetime64):
                    target_date = np.datetime64(target_date)
                    
            if target_date not in date_to_idx:
                raise ValueError(f"Date {target_date} not found in model residuals for narrative restriction.")
                
            t_idx = date_to_idx[target_date]
            val = e_t[s_idx, t_idx]
            
            r_type = r["type"]
            if r_type == "sign":
                sign_val = r["value"]
                if sign_val == "+" and val <= 0:
                    return False
                elif sign_val == "-" and val >= 0:
                    return False
                    
        return True
        
    def _get_ma_matrices(self, steps):
        """Computes Phi matrices of the VAR up to `steps`."""
        forecaster = VARForecaster(self.var_result)
        return forecaster._compute_ma_rep(steps)

    def get_summary(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        acc = len(self.accepted_rotations)
        return f"Sign/Zero SVAR Model\nAccepted Draws: {acc} / {self.total_draws}\n\nVAR Summary:\n{self.var_result.summary()}"

    def get_model_metrics(self):
        if self.var_result is None:
            raise ValueError("Model not trained")
        return {
            "nobs": int(self.var_result.nobs),
            "accepted_draws": len(self.accepted_rotations),
            "total_draws": getattr(self, "total_draws", 0)
        }

    def get_representative_draw(self):
        """
        Selects the single accepted draw whose IRF is closest to the point-wise median
        IRF (Fry and Pagan 2011 "Median Target").
        """
        if not self.accepted_irfs:
            raise ValueError("No accepted draws available.")
            
        # accepted_irfs is a list of arrays of shape (H+1, K, K)
        all_irfs = np.array(self.accepted_irfs) # (N, H+1, K, K)
        
        # Point-wise median across draws
        median_irf = np.median(all_irfs, axis=0) # (H+1, K, K)
        
        # Calculate sum of squared standardized deviations from median
        # Standardize by point-wise standard deviation to avoid scale issues
        std_irf = np.std(all_irfs, axis=0)
        std_irf[std_irf == 0] = 1e-10 # prevent div by zero
        
        distances = np.sum(((all_irfs - median_irf) / std_irf)**2, axis=(1, 2, 3))
        best_idx = np.argmin(distances)
        
        return {
            "index": best_idx,
            "rotation": self.accepted_rotations[best_idx],
            "irf": self.accepted_irfs[best_idx]
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metrics()
