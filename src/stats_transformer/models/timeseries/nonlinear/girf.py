import numpy as np
import pandas as pd


class GIRFEngine:
    """
    Generalized Impulse Response Functions (GIRF) Engine.
    Computes history-dependent Monte Carlo GIRFs for non-linear time-series models
    such as TVAR and STVAR (Koop, Pesaran, & Potter 1996).
    """

    def __init__(self, model, horizon=20, shock_size=1.0, num_histories=50, num_bootstrap=100, seed=42):
        self.model = model
        self.horizon = horizon
        self.shock_size = shock_size
        self.num_histories = num_histories
        self.num_bootstrap = num_bootstrap
        self.seed = seed

    def _simulate_path(self, init_history, shocks, is_stvar=False):
        p = self.model.lags
        K = len(self.model.target_variables)
        H = len(shocks)

        history_buffer = list(init_history)
        simulated = []

        for h in range(H):
            curr_lags = np.hstack(history_buffer[-p:][::-1])
            x_vec = np.hstack([[1.0], curr_lags]) if getattr(self.model, "intercept", True) else curr_lags

            if is_stvar:
                tr_val = history_buffer[-self.model.delay][0]
                z = -self.model.gamma * (tr_val - self.model.c) / max(1e-6, getattr(self.model, "scale", 1.0))
                g_wt = 1.0 / (1.0 + np.exp(np.clip(z, -50.0, 50.0)))
                cond_params = (1.0 - g_wt) * self.model.regime_1_params + g_wt * self.model.regime_2_params
            else:
                th_val = history_buffer[-self.model.delay][0]
                cond_params = self.model.regime_1_params if th_val <= self.model.gamma else self.model.regime_2_params

            y_next = x_vec @ cond_params + shocks[h]
            simulated.append(y_next)
            history_buffer.append(y_next)

        return np.array(simulated)

    def compute_girf(self, shock_variable=None, shock_magnitude=None):
        rng = np.random.default_rng(self.seed)
        var_names = self.model.target_variables
        K = len(var_names)
        p = self.model.lags
        d = getattr(self.model, "delay", 1)
        max_lag = max(p, d)

        shock_idx = 0
        if shock_variable is not None:
            if shock_variable in var_names:
                shock_idx = var_names.index(shock_variable)
            elif isinstance(shock_variable, int):
                shock_idx = shock_variable

        resids_arr = self.model.residuals.values
        resids_std = np.std(resids_arr, axis=0)

        delta_val = shock_magnitude if shock_magnitude is not None else (self.shock_size * resids_std[shock_idx])
        shock_vector = np.zeros(K)
        shock_vector[shock_idx] = delta_val

        y_clean = self.model.df_clean[var_names].values
        T = len(y_clean)

        avail_history_indices = np.arange(max_lag, T - self.horizon)
        if len(avail_history_indices) == 0:
            avail_history_indices = [max_lag]

        sample_histories = rng.choice(avail_history_indices, size=min(self.num_histories, len(avail_history_indices)), replace=False)
        is_stvar = hasattr(self.model, "transition_weights")

        all_differences = []

        for h_idx in sample_histories:
            init_hist = [y_clean[h_idx - k] for k in range(max_lag, 0, -1)]

            for b in range(self.num_bootstrap):
                rand_indices = rng.choice(len(resids_arr), size=self.horizon, replace=True)
                bootstrap_resids = resids_arr[rand_indices].copy()

                base_path = self._simulate_path(init_hist, bootstrap_resids, is_stvar=is_stvar)

                shocked_resids = bootstrap_resids.copy()
                shocked_resids[0] += shock_vector
                shock_path = self._simulate_path(init_hist, shocked_resids, is_stvar=is_stvar)

                diff = shock_path - base_path
                all_differences.append(diff)

        all_diffs_arr = np.array(all_differences)

        mean_girf = np.mean(all_diffs_arr, axis=0)
        lower_band = np.percentile(all_diffs_arr, 16.0, axis=0)
        upper_band = np.percentile(all_diffs_arr, 84.0, axis=0)

        records = []
        for h in range(self.horizon):
            for i, v in enumerate(var_names):
                records.append({
                    "horizon": h,
                    "variable": v,
                    "shock": var_names[shock_idx],
                    "girf": float(mean_girf[h, i]),
                    "ci_lower": float(lower_band[h, i]),
                    "ci_upper": float(upper_band[h, i])
                })

        girf_df = pd.DataFrame(records)
        return girf_df
