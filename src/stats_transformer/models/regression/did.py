import numpy as np
import pandas as pd
from scipy import stats
from stats_transformer.models.base import ModelBase

class DiDModel(ModelBase):
    _is_multivariate = True

    def __init__(self, entity_column=None, time_column=None, cohort_column=None, outcome_column=None, control_variables=None, control_group="never_treated", base_period="varying", **kwargs):
        self.entity_column = entity_column
        self.time_column = time_column
        self.cohort_column = cohort_column
        self.outcome_column = outcome_column
        self.control_variables = control_variables or []
        self.control_group = control_group
        self.base_period = base_period
        super().__init__(entity_column=entity_column, **kwargs)
        self.target_variables = [outcome_column]
        self.att_gt_ = None
        self.pretrend_stat_ = None
        self.pretrend_pvalue_ = None

    def _get_required_columns(self):
        cols = [self.entity_column, self.time_column, self.cohort_column, self.outcome_column] + list(self.control_variables)
        return list(dict.fromkeys(c for c in cols if c))

    def build_model(self):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")

        df = self.df_clean.copy()
        periods = sorted(df[self.time_column].unique())
        cohorts = sorted(c for c in df[self.cohort_column].unique() if c != 0 and not pd.isna(c))

        panel = df.set_index([self.entity_column, self.time_column])[self.outcome_column].unstack(self.time_column)
        cohort_map = df.groupby(self.entity_column)[self.cohort_column].first()

        results = []
        for g in cohorts:
            for t in periods:
                base_t = self._base_period(g, t, periods)
                if base_t == t:
                    continue
                att, se = self._att_gt(panel, cohort_map, g, t, base_t)
                if att is not None:
                    results.append({"group": g, "time": t, "base_period": base_t, "att": att, "se": se, "post": t >= g})

        self.att_gt_ = pd.DataFrame(results)
        self._compute_pretrend_test()
        self.simple_att_ = self._aggregate_simple()
        self.event_study_ = self._aggregate_event_study()
        self.model = {"att_gt": self.att_gt_, "simple_att": self.simple_att_}
        return self.model

    def _base_period(self, g, t, periods):
        # "varying" base period (Callaway & Sant'Anna's default): compare to
        # the period immediately before treatment for post-treatment ATT(g,t),
        # and to t-1 for pre-treatment placebo checks -- both are the standard
        # "not yet treated at g-1" comparison point.
        idx = periods.index(t)
        if t >= g:
            pre_idx = periods.index(g) - 1
            return periods[pre_idx] if pre_idx >= 0 else periods[0]
        return periods[idx - 1] if idx > 0 else periods[0]

    def _control_units(self, cohort_map, g, t):
        if self.control_group == "never_treated":
            mask = cohort_map == 0
        else:
            mask = (cohort_map == 0) | (cohort_map > max(g, t))
        return cohort_map.index[mask]

    def _att_gt(self, panel, cohort_map, g, t, base_t):
        treated_units = cohort_map.index[cohort_map == g]
        control_units = self._control_units(cohort_map, g, t)
        control_units = control_units.difference(treated_units)

        if len(treated_units) == 0 or len(control_units) == 0:
            return None, None
        if t not in panel.columns or base_t not in panel.columns:
            return None, None

        treated_diff = (panel.loc[panel.index.intersection(treated_units), t] - panel.loc[panel.index.intersection(treated_units), base_t]).dropna()
        control_diff = (panel.loc[panel.index.intersection(control_units), t] - panel.loc[panel.index.intersection(control_units), base_t]).dropna()

        if len(treated_diff) == 0 or len(control_diff) == 0:
            return None, None

        att = float(treated_diff.mean() - control_diff.mean())
        se_treated = treated_diff.var(ddof=1) / len(treated_diff) if len(treated_diff) > 1 else 0.0
        se_control = control_diff.var(ddof=1) / len(control_diff) if len(control_diff) > 1 else 0.0
        se = float(np.sqrt(se_treated + se_control))
        return att, se

    def _compute_pretrend_test(self):
        pre = self.att_gt_[~self.att_gt_["post"]] if self.att_gt_ is not None and not self.att_gt_.empty else pd.DataFrame()
        if pre.empty:
            self.pretrend_stat_, self.pretrend_pvalue_ = None, None
            return
        valid = pre[pre["se"] > 0]
        if valid.empty:
            self.pretrend_stat_, self.pretrend_pvalue_ = None, None
            return
        stat = float(np.sum((valid["att"] / valid["se"]) ** 2))
        df = len(valid)
        self.pretrend_stat_ = stat
        self.pretrend_pvalue_ = float(1 - stats.chi2.cdf(stat, df))

    def _aggregate_simple(self):
        post = self.att_gt_[self.att_gt_["post"]] if self.att_gt_ is not None and not self.att_gt_.empty else pd.DataFrame()
        if post.empty:
            return {"att": None, "se": None}
        weights = np.ones(len(post)) / len(post)
        att = float(np.sum(weights * post["att"]))
        se = float(np.sqrt(np.sum((weights * post["se"]) ** 2)))
        return {"att": att, "se": se, "n_att_gt": len(post)}

    def _aggregate_event_study(self):
        if self.att_gt_ is None or self.att_gt_.empty:
            return pd.DataFrame()
        df = self.att_gt_.copy()
        df["event_time"] = df["time"] - df["group"]
        grouped = df.groupby("event_time").apply(lambda g: pd.Series({"att": g["att"].mean(), "se": float(np.sqrt(np.sum(g["se"] ** 2)) / len(g)), "n_att_gt": len(g)}), include_groups=False)
        return grouped.reset_index()

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        lines = [f"Callaway-Sant'Anna DiD: {len(self.att_gt_)} group-time ATT(g,t) estimates"]
        lines.append(f"Simple aggregate ATT: {self.simple_att_.get('att')} (se={self.simple_att_.get('se')})")
        if self.pretrend_pvalue_ is not None:
            lines.append(f"Pre-trend test: stat={self.pretrend_stat_:.4f}, p-value={self.pretrend_pvalue_:.4f}")
        return "\n".join(lines)

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "n_cohorts": int(self.att_gt_["group"].nunique()) if not self.att_gt_.empty else 0,
            "n_att_gt": int(len(self.att_gt_)),
            "simple_att": self.simple_att_.get("att"),
            "simple_att_se": self.simple_att_.get("se"),
            "pretrend_stat": self.pretrend_stat_,
            "pretrend_pvalue": self.pretrend_pvalue_,
            "control_group": self.control_group,
        }

    def compute_att_gt(self):
        if self.att_gt_ is None:
            raise ValueError("Model not trained")
        return self.att_gt_

    def compute_event_study(self):
        if self.event_study_ is None:
            raise ValueError("Model not trained")
        return self.event_study_

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
