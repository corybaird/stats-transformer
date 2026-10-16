import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase
from stats_transformer.models.timeseries.reduced_form.forecasting import VARForecaster

def _zero_restriction_bases(restrictions, shock_to_idx, var_to_idx, phis, p_mat, k):
    """
    Builds the stacked zero-restriction matrix Z_j F(θ) for each shock j.
    Rows are the unit vectors selecting the restricted responses pre-multiplied
    by the structural IRFs: Z_j F(θ) = e_v' Φ_h P for impact (h = 0) and higher
   horizon zero restrictions (Arias, Rubio-Ramírez & Waggoner 2018, Section 4).
    """
    rows_by_shock = {j: [] for j in range(k)}
    for r in restrictions:
        if r.get("type") != "zero":
            continue
        if r["shock"] not in shock_to_idx:
            raise ValueError(f"Unknown shock {r['shock']} in zero restriction")
        if r["response"] not in var_to_idx:
            raise ValueError(f"Unknown response variable {r['response']} in zero restriction")
        h_def = r.get("horizon", 0)
        horizons = range(h_def[0], h_def[1] + 1) if isinstance(h_def, list) else [h_def]
        for h in horizons:
            if h >= len(phis):
                raise ValueError(f"Zero restriction horizon {h} exceeds computed IRF horizons")
            rows_by_shock[shock_to_idx[r["shock"]]].append(phis[h][var_to_idx[r["response"]]][None, :] @ p_mat)
    return [np.vstack(rows_by_shock[j]) if rows_by_shock[j] else np.zeros((0, k)) for j in range(k)]

def _sample_zero_restricted_q(zero_bases, k, rng):
    """
    Constructs an orthogonal Q satisfying the zero restrictions by construction,
    following Arias, Rubio-Ramírez & Waggoner (2018, Econometrica), Algorithm 2 /
    Section 4: shocks are ordered by decreasing number of zero restrictions
    (ARW require z_j <= n - j) and column j is drawn uniformly from the unit
    sphere in the null space of [Z_j F(θ); q_1', ..., q_{j-1}']. Sign and
    narrative restrictions must afterwards be checked as usual.

    Raises ValueError if the ARW ordering condition z_j <= n - j is violated.
    """
    order = sorted(range(k), key=lambda j: -zero_bases[j].shape[0])
    Q = np.zeros((k, k))
    prev = np.zeros((0, k))
    for j in order:
        aug = np.vstack([zero_bases[j], prev]) if prev.shape[0] else zero_bases[j]
        if aug.shape[0] == 0:
            null_basis = np.eye(k)
        else:
            _, s, vt = np.linalg.svd(aug)
            rank = int(np.sum(s > max(aug.shape) * np.finfo(float).eps * s.max()))
            null_basis = vt[rank:].T  # (k, k - rank) orthonormal null-space basis
        if null_basis.shape[1] < 1:
            raise ValueError(
                "Zero restrictions violate the Arias, Rubio-Ramírez & Waggoner (2018) "
                "ordering condition z_j <= n - j: no rotation satisfies them."
            )
        g = rng.standard_normal(null_basis.shape[1])
        q_j = null_basis @ g
        q_j /= np.linalg.norm(q_j)
        Q[:, j] = q_j
        prev = np.vstack([prev, q_j[None, :]])
    return Q

class SignZeroSVARModel(ModelBase):
    """
    SVAR Model with Sign, Zero, and Narrative restrictions.
    Evaluates configurations loaded from a YAML schema.
    """
    _is_multivariate = True

    def __init__(self, target_variables=None, config_path=None, date_column=None, maxlags=1, max_draws=10000, required_accepts=100, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.target_variables = target_variables or getattr(self, "target_variables", [])
        self.config_path = self._param("config_path", config_path, None)
        self.date_column = self._param("date_column", date_column, None)
        self.time_column = self.date_column
        self.maxlags = self._param("maxlags", maxlags, 1)
        self.max_draws = self._param("max_draws", max_draws, 10000)
        self.required_accepts = self._param("required_accepts", required_accepts, 100)
        self.seed = self._param("seed", seed, 42)
        
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
        """Build the underlying model object from the loaded data."""
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

        Zero restrictions are imposed by construction (Arias, Rubio-Ramírez &
        Waggoner 2018, Algorithm 2); if no zero restrictions are configured, draws
        are Haar-uniform rotations and behaviour is unchanged.
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
        
        # Precompute variable and shock indices
        var_to_idx = {v: i for i, v in enumerate(self.target_variables)}
        shock_to_idx = {s: i for i, s in enumerate(self.shocks)}

        has_zero_restrictions = any(r.get("type") == "zero" for r in self.restrictions)
        zero_bases = _zero_restriction_bases(self.restrictions, shock_to_idx, var_to_idx, phis, p_chol, k) if has_zero_restrictions else None
        
        self.accepted_rotations = []
        self.accepted_irfs = []
        
        draws = 0
        accepts = 0
        
        rng = np.random.default_rng(self.seed) # For reproducibility in the sampling

        while draws < self.max_draws and accepts < self.required_accepts:
            draws += 1
            if zero_bases is not None:
                # Zero restrictions imposed by construction (Arias et al. 2018)
                Q = _sample_zero_restricted_q(zero_bases, k, rng)
            else:
                # 1. Generate random orthogonal matrix (Haar measure via QR)
                W = rng.normal(size=(k, k))
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
                        # Zero restrictions are imposed by construction (Arias et
                        # al. 2018); this checks the construction, it does not
                        # filter accepts.
                        assert abs(val) <= 1e-10, (
                            f"Constructed rotation violates zero restriction: |{val}| > 1e-10"
                        )
                            
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
        """Return a text summary of the fitted model."""
        if self.var_result is None:
            raise ValueError("Model not trained")
        acc = len(self.accepted_rotations)
        return f"Sign/Zero SVAR Model\nAccepted Draws: {acc} / {self.total_draws}\n\nVAR Summary:\n{self.var_result.summary()}"

    def get_model_metrics(self):
        """Return key fit metrics as a dictionary."""
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
        """Run the full workflow end to end."""
        self.fit(data)
        return self.get_model_metrics()
