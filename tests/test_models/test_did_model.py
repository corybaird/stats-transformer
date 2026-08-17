import numpy as np
import pandas as pd
import pytest
from linearmodels.panel import PanelOLS
from stats_transformer.models.regression.did import DiDModel
from examples.academic.lane_2025 import Lane2025Replication

def _simulate_staggered_panel(seed=1, n_units=300, periods=range(2000, 2013), cohorts=(2005, 2008), cohort_effect=2.0, effect_growth=0.0, pretrend_violation=0.0):
    np.random.seed(seed)
    periods = list(periods)
    cohort_choices = list(cohorts) + [0]
    unit_cohort = np.random.choice(cohort_choices, size=n_units)
    unit_fe = np.random.normal(scale=2, size=n_units)

    rows = []
    for i in range(n_units):
        g = unit_cohort[i]
        for t in periods:
            time_fe = 0.2 * (t - periods[0])
            treated = (g != 0) and (t >= g)
            exposure = (t - g + 1) if treated else 0
            effect = (cohort_effect + effect_growth * (exposure - 1)) if treated else 0.0
            violation = pretrend_violation * (t - g) if (g != 0 and t < g) else 0.0
            y = unit_fe[i] + time_fe + effect + violation + np.random.normal(scale=0.4)
            rows.append({"unit": i, "year": t, "cohort": g, "y": y, "treated": int(treated), "true_effect": effect})
    return pd.DataFrame(rows)

def test_did_recovers_known_homogeneous_att():
    df = _simulate_staggered_panel(cohort_effect=2.0, effect_growth=0.0)
    model = DiDModel(entity_column="unit", time_column="year", cohort_column="cohort", outcome_column="y")
    metrics = model.fit(df)
    assert metrics["simple_att"] == pytest.approx(2.0, abs=0.1)

def test_did_pretrend_test_does_not_reject_under_parallel_trends():
    df = _simulate_staggered_panel(cohort_effect=2.0, pretrend_violation=0.0)
    model = DiDModel(entity_column="unit", time_column="year", cohort_column="cohort", outcome_column="y")
    metrics = model.fit(df)
    assert metrics["pretrend_pvalue"] > 0.05

def test_did_pretrend_test_rejects_planted_violation():
    df = _simulate_staggered_panel(cohort_effect=2.0, pretrend_violation=0.5)
    model = DiDModel(entity_column="unit", time_column="year", cohort_column="cohort", outcome_column="y")
    metrics = model.fit(df)
    assert metrics["pretrend_pvalue"] < 0.01

def test_did_outperforms_naive_twfe_under_dynamic_treatment_effects():
    df = _simulate_staggered_panel(seed=42, cohorts=(2004, 2006, 2008), cohort_effect=2.0, effect_growth=1.0)
    true_att = df.loc[df["treated"] == 1, "true_effect"].mean()

    model = DiDModel(entity_column="unit", time_column="year", cohort_column="cohort", outcome_column="y")
    metrics = model.fit(df)
    cs_att = metrics["simple_att"]

    panel = df.set_index(["unit", "year"])
    twfe = PanelOLS(panel["y"], panel[["treated"]], entity_effects=True, time_effects=True).fit()
    twfe_att = float(twfe.params["treated"])

    cs_error = abs(cs_att - true_att)
    twfe_error = abs(twfe_att - true_att)
    assert cs_error < twfe_error
    assert cs_error / true_att < 0.1

def test_did_event_study_aggregation_shape():
    df = _simulate_staggered_panel(cohorts=(2005, 2008), cohort_effect=2.0)
    model = DiDModel(entity_column="unit", time_column="year", cohort_column="cohort", outcome_column="y")
    model.fit(df)
    event_study = model.compute_event_study()
    assert "event_time" in event_study.columns
    assert (event_study["event_time"] >= 0).any()

def test_did_not_yet_treated_control_group():
    df = _simulate_staggered_panel(cohorts=(2005, 2008), cohort_effect=2.0)
    model = DiDModel(entity_column="unit", time_column="year", cohort_column="cohort", outcome_column="y", control_group="not_yet_treated")
    metrics = model.fit(df)
    assert metrics["simple_att"] == pytest.approx(2.0, abs=0.15)

def test_did_run_returns_metadata():
    df = _simulate_staggered_panel(cohorts=(2005,), cohort_effect=1.5)
    model = DiDModel(entity_column="unit", time_column="year", cohort_column="cohort", outcome_column="y")
    metadata = model.run(df)
    assert "metrics" in metadata
    assert metadata["metrics"]["simple_att"] == pytest.approx(1.5, abs=0.1)

def test_lane_2025_replication():
    result = Lane2025Replication().run()
    assert "metrics" in result
    assert "did_metrics" in result
    assert result["did_metrics"]["n_att_gt"] > 0
