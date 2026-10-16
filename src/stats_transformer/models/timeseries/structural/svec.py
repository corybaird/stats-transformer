import numpy as np
import pandas as pd
from scipy.optimize import minimize, root
from scipy.linalg import cholesky, null_space
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.base import ModelBase


def _orthogonal_complement(M: np.ndarray) -> np.ndarray:
    K, r = M.shape
    if r == 0:
        return np.eye(K)
    if r >= K:
        return np.empty((K, 0))
    q, _ = np.linalg.qr(M, mode="complete")
    return q[:, r:]


def _restriction_matrix(Xi: np.ndarray, SR: np.ndarray, LR: np.ndarray) -> np.ndarray:
    """Rows of E impose the restrictions on vec(B) (row-major, index i*K+j).

    SR entries give unit rows (B[i, j] = SR[i, j]); an LR restriction
    (Xi B)[i, j] = LR[i, j] is column j of B dotted with row i of Xi, so its
    row places Xi[i, :] on the entries of column j (indices j, K+j, ...)."""
    K = SR.shape[0]
    sr_mask = ~np.isnan(SR)
    lr_mask = ~np.isnan(LR)
    E = np.zeros((0, K * K))
    for i in range(K):
        for j in range(K):
            if lr_mask[i, j]:
                row = np.zeros(K * K)
                row[j::K] = Xi[i, :]
                E = np.vstack([E, row])
            if sr_mask[i, j]:
                row = np.zeros(K * K)
                row[i * K + j] = 1.0
                E = np.vstack([E, row])
    return E


def _restriction_violation(vec_B: np.ndarray, Xi: np.ndarray, SR: np.ndarray, LR: np.ndarray) -> float:
    """Max absolute violation of the SR/LR restrictions for vec(B)."""
    K = SR.shape[0]
    B = vec_B.reshape((K, K))
    sr_mask = ~np.isnan(SR)
    lr_mask = ~np.isnan(LR)
    violations = [abs(B[i, j] - float(SR[i, j])) for i in range(K) for j in range(K) if sr_mask[i, j]]
    violations += [abs((Xi @ B)[i, j] - float(LR[i, j])) for i in range(K) for j in range(K) if lr_mask[i, j]]
    return max(violations, default=0.0)


class SVEC:
    """
    Structural Vector Error Correction (SVEC) model.
    Estimates short-run (SR) and long-run (LR) structural impact matrices
    via Maximum Likelihood matching R vars::SVEC (King et al. 1991).
    """
    def __init__(self, vecm_result, SR=None, LR=None, max_iter=1000, r=None):
        self.vecm = vecm_result
        self.sigma_u = np.asarray(vecm_result.sigma_u)
        self.neqs = self.sigma_u.shape[0]
        self.nobs = getattr(vecm_result, "nobs", 100)

        self.coint_rank = r if r is not None else getattr(vecm_result, "coint_rank", 1)
        self.SR = np.asarray(SR, dtype=float) if SR is not None else np.full((self.neqs, self.neqs), np.nan)
        self.LR = np.asarray(LR, dtype=float) if LR is not None else np.full((self.neqs, self.neqs), np.nan)
        self.max_iter = max_iter

        self.beta = getattr(vecm_result, "beta", None)
        self.alpha = getattr(vecm_result, "alpha", None)
        self.gamma = getattr(vecm_result, "gamma", None)

        self.Xi = self._compute_long_run_impact_multiplier()
        self._fit_structural()

    def _compute_long_run_impact_multiplier(self) -> np.ndarray:
        K = self.neqs
        r = self.coint_rank
        if r == 0 or r >= K or self.alpha is None or self.beta is None:
            return np.eye(K)

        alpha_mat = np.asarray(self.alpha)
        beta_mat = np.asarray(self.beta)
        if alpha_mat.ndim == 1:
            alpha_mat = alpha_mat.reshape(K, r)
        if beta_mat.ndim == 1:
            beta_mat = beta_mat.reshape(K, r)

        alpha_orth = _orthogonal_complement(alpha_mat)
        beta_orth = _orthogonal_complement(beta_mat)

        if alpha_orth.shape[1] == 0 or beta_orth.shape[1] == 0:
            return np.eye(K)

        gamma_sum = np.zeros((K, K))
        if self.gamma is not None:
            gamma_arr = np.asarray(self.gamma)
            if gamma_arr.ndim == 3:
                gamma_sum = np.sum(gamma_arr, axis=0)
            elif gamma_arr.ndim == 2:
                if gamma_arr.shape == (K, K):
                    gamma_sum = gamma_arr
                elif gamma_arr.shape[0] == K and gamma_arr.shape[1] % K == 0:
                    p_diff = gamma_arr.shape[1] // K
                    for l in range(p_diff):
                        gamma_sum += gamma_arr[:, l * K : (l + 1) * K]
                elif gamma_arr.shape[1] == K and gamma_arr.shape[0] % K == 0:
                    p_diff = gamma_arr.shape[0] // K
                    for l in range(p_diff):
                        gamma_sum += gamma_arr[l * K : (l + 1) * K, :]

        Gamma = np.eye(K) - gamma_sum
        mid = alpha_orth.T @ Gamma @ beta_orth
        try:
            mid_inv = np.linalg.inv(mid)
            Xi = beta_orth @ mid_inv @ alpha_orth.T
        except np.linalg.LinAlgError:
            Xi = beta_orth @ np.linalg.pinv(mid) @ alpha_orth.T
        return Xi

    def _fit_structural(self):
        K = self.neqs
        T = self.nobs
        sigma_u = self.sigma_u
        Xi = self.Xi

        sr_mask = ~np.isnan(self.SR)
        lr_mask = ~np.isnan(self.LR)

        try:
            L = cholesky(sigma_u, lower=True)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(sigma_u)
            eigvals = np.maximum(eigvals, 1e-6)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # Deterministic starts plus seeded random starts.
        rng = np.random.default_rng(0)
        starts = [
            L.flatten(),
            np.diag(np.sqrt(np.maximum(np.diag(sigma_u), 0.0))).flatten(),
            rng.normal(0, 1, K * K),
            rng.normal(0, 1, K * K),
        ]

        # Restriction matrix on vec(B) (row-major); the free directions of Sb
        # remove any linear dependence between the restriction rows.
        E = _restriction_matrix(Xi, self.SR, self.LR)
        Sb = null_space(E) if E.shape[0] else np.eye(K * K)
        # Non-zero restriction targets are shifted into the constrained part;
        # the free directions only span the homogeneous system.
        sr_mask = ~np.isnan(self.SR)
        lr_mask = ~np.isnan(self.LR)
        rhs = []
        for i in range(K):
            for j in range(K):
                if lr_mask[i, j]:
                    rhs.append(float(self.LR[i, j]))
                if sr_mask[i, j]:
                    rhs.append(float(self.SR[i, j]))
        b_shift = np.linalg.lstsq(E, np.array(rhs), rcond=None)[0] if E.shape[0] else np.zeros(K * K)

        def objective(b_flat):
            B = b_flat.reshape((K, K))
            try:
                sign, logdet = np.linalg.slogdet(B @ B.T)
                if sign <= 0 or logdet <= -100:
                    return 1e10
                inv_BB = np.linalg.inv(B @ B.T)
                nll = 0.5 * T * (logdet + np.trace(inv_BB @ sigma_u))
                return nll
            except (np.linalg.LinAlgError, ValueError):
                return 1e10

        def violation(b_flat):
            return _restriction_violation(b_flat, Xi, self.SR, self.LR)

        self.restr_tol = 1e-6
        candidates = []  # (objective, vec(B)) of restriction-satisfying fits
        best_viol = np.inf
        # Just identified: the concentrated likelihood only depends on B B', whose
        # unique minimum over positive-definite choices is B B' = sigma_u, so the
        # ML solution solves the system {restrictions, B B' = sigma_u} directly.
        if Sb.shape[1] == K * (K + 1) // 2:
            def direct_residuals(b_flat):
                B = b_flat.reshape((K, K))
                out = []
                for i in range(K):
                    for j in range(K):
                        if sr_mask[i, j]:
                            out.append(B[i, j] - float(self.SR[i, j]))
                        if lr_mask[i, j]:
                            out.append((Xi @ B)[i, j] - float(self.LR[i, j]))
                triu = np.triu_indices(K)
                out.extend((B @ B.T)[triu] - sigma_u[triu])
                return np.array(out)

            for b0 in starts:
                sol = root(direct_residuals, b0, method="lm")
                if not sol.success:
                    continue
                viol = violation(sol.x)
                best_viol = min(best_viol, viol)
                if viol <= self.restr_tol:
                    candidates.append((objective(sol.x), sol.x))

        # Fallback (over-identified, or the system solve failed): minimise the
        # concentrated likelihood along the free directions from several
        # deterministic starts, keeping the best fit. Finite-difference BFGS
        # at tight tolerances can stop with a precision-loss flag while still at
        # a good point, so a fit is accepted by what it achieves — restriction
        # violation within tolerance and a finite objective — not by the
        # success flag or by an improvement over its start (BFGS can leave a
        # degenerate start and still return an unusable point, and vice versa).
        # The concentrated likelihood min value over PD B is the unrestricted
        # one 0.5*T*(K + logdet(sigma_u)), reached at BB' = sigma_u; anything
        # below it (near-singular BB', logdet just above the -100 floor) is not
        # a likelihood at all, so it fails the finiteness gate too.
        free_starts = [np.linalg.lstsq(Sb, s0, rcond=None)[0] for s0 in starts[:2]]
        rng = np.random.default_rng(0)
        free_starts += [rng.normal(0.0, 1.0, Sb.shape[1]) for _ in range(2)]
        failed_violation = failed_objective = 0
        for g0 in free_starts:
            g_obj = lambda g: objective(Sb @ g + b_shift)
            # Scaling the free component leaves the restrictions intact
            # (E @ Sb = 0), so rescale each start onto the likelihood-relevant
            # scale t^2 = tr((B0 B0')^-1 Sigma_u) / K; without this BFGS never
            # escapes the degenerate small-|B| cone of zero-restriction fits.
            b0 = (Sb @ g0 + b_shift).reshape((K, K))
            try:
                t2 = np.trace(np.linalg.inv(b0 @ b0.T) @ sigma_u) / K
            except np.linalg.LinAlgError:
                t2 = np.nan
            if np.isfinite(t2) and t2 > 0.0:
                g0 = g0 * float(np.sqrt(min(t2, 1e300)))
            fit_res = minimize(g_obj, g0, method="BFGS", options={"maxiter": self.max_iter, "gtol": 1e-8})
            b_flat = Sb @ fit_res.x + b_shift
            viol = violation(b_flat)
            obj = objective(b_flat)
            obj_floor = 0.5 * T * (K + np.linalg.slogdet(sigma_u)[1])
            if fit_res.success or obj <= g_obj(g0):
                best_viol = min(best_viol, viol)
            if viol > self.restr_tol:
                failed_violation += 1
            elif not np.isfinite(obj) or obj >= 1e10 or obj < obj_floor - 1e-8 * abs(obj_floor):
                failed_objective += 1
            else:
                candidates.append((obj, b_flat))

        if not candidates:
            self.max_restriction_violation = best_viol
            reasons = []
            if failed_violation:
                reasons.append(f"{failed_violation} violated the identifying restrictions beyond {self.restr_tol:g}")
            if failed_objective:
                reasons.append(f"{failed_objective} had a non-finite objective")
            reason = "; ".join(reasons) if reasons else "all optimiser runs failed"
            err = RuntimeError(
                f"SVEC estimation failed: no start satisfied the identifying "
                f"restrictions to {self.restr_tol:g} with a finite objective "
                f"({reason}; best violation {best_viol:.3g}); "
                f"tried {len(free_starts)} deterministic starts"
            )
            err.max_restriction_violation = best_viol
            raise err

        # Best objective wins; a tie within 1e-9 keeps the first (deterministic) start.
        best_obj, b_flat = candidates[0]
        for c_obj, c_flat in candidates[1:]:
            if c_obj < best_obj - 1e-9:
                best_obj, b_flat = c_obj, c_flat
        B_est = b_flat.reshape((K, K))
        self.max_restriction_violation = violation(b_flat)
        for i in range(K):
            for j in range(K):
                if sr_mask[i, j]:
                    B_est[i, j] = self.SR[i, j]

        self.SR_est = B_est
        self.LR_est = Xi @ B_est
        self.bse = np.full((K, K), np.nan)

        try:
            sign, logdet_bb = np.linalg.slogdet(B_est @ B_est.T)
            inv_bb = np.linalg.inv(B_est @ B_est.T)
            self.llf = -0.5 * T * (K * np.log(2 * np.pi) + logdet_bb + np.trace(inv_bb @ sigma_u))
        except (np.linalg.LinAlgError, ValueError):
            self.llf = -np.inf

        sign_s, logdet_s = np.linalg.slogdet(sigma_u)
        self.llf_unrestricted = -0.5 * T * (K * np.log(2 * np.pi) + logdet_s + K)
        self.lr_stat = max(0.0, 2.0 * (self.llf_unrestricted - self.llf))

    def get_structural_matrices(self):
        return {
            "SR": self.SR_est,
            "LR": self.LR_est
        }


class SVECModel(ModelBase):
    _is_multivariate = True

    def __init__(self, target_variables=None, date_column=None, k_ar_diff=1, coint_rank=1, deterministic="n", SR=None, LR=None, max_iter=1000, **kwargs):
        super().__init__(**kwargs)
        if target_variables:
            self.target_variables = target_variables
        elif self.params:
            self.target_variables = self.params.get("model", {}).get("target_variables", [])
            if not self.target_variables and self.target:
                self.target_variables = [self.target] + (self.independent_variables or [])
        else:
            self.target_variables = []

        model_params = self.params.get("model", {})
        self.date_column = date_column or model_params.get("date_column")
        self.k_ar_diff = model_params.get("k_ar_diff", k_ar_diff)
        self.coint_rank = model_params.get("coint_rank", coint_rank)
        self.deterministic = model_params.get("deterministic", deterministic)
        self.SR = SR if SR is not None else model_params.get("SR")
        self.LR = LR if LR is not None else model_params.get("LR")
        self.max_iter = model_params.get("max_iter", max_iter)
        self.svec_fit = None

    def _get_required_columns(self):
        cols = list(self.target_variables)
        if self.date_column:
            cols.append(self.date_column)
        return cols

    def build_model(self, drop_na=True):
        if getattr(self, "df_clean", None) is None:
            raise ValueError("No cleaned data available")
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)

        self.y = self.df_clean[self.target_variables]
        self.vecm_spec = VECM(self.y, k_ar_diff=self.k_ar_diff, coint_rank=self.coint_rank, deterministic=self.deterministic)
        self.vecm_result = self.vecm_spec.fit()
        self.model = self.vecm_result
        self.svec_fit = SVEC(self.vecm_result, SR=self.SR, LR=self.LR, max_iter=self.max_iter, r=self.coint_rank)
        self.B_0 = self.svec_fit.SR_est
        self.LR_est = self.svec_fit.LR_est
        return self.svec_fit

    def fit(self, df, drop_na=True):
        required_cols = self._get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        self.df_clean = df.copy()
        if drop_na:
            self.df_clean = self.df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)

        self.build_model(drop_na=False)
        return self.get_model_metrics()

    def get_summary(self):
        if self.svec_fit is None:
            raise ValueError("Model not trained")
        summary_text = (
            f"SVEC Structural Error Correction Model\n"
            f"Cointegration Rank: {self.coint_rank}\n"
            f"Log-Likelihood: {self.svec_fit.llf:.4f}\n"
            f"LR Over-identification Test Stat: {self.svec_fit.lr_stat:.4f}\n\n"
            f"Estimated Short-Run Impact Matrix B (SR):\n{self.svec_fit.SR_est}\n\n"
            f"Estimated Long-Run Impact Matrix (LR = Xi @ B):\n{self.svec_fit.LR_est}"
        )
        return summary_text

    def get_model_metrics(self):
        if self.svec_fit is None:
            raise ValueError("Model not trained")
        return {
            "num_observations": len(self.y),
            "log_likelihood": float(self.svec_fit.llf),
            "coint_rank": int(self.coint_rank),
            "lr_stat": float(self.svec_fit.lr_stat)
        }
