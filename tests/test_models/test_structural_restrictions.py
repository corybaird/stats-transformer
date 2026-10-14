import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from stats_transformer.models.timeseries.identification.sign_zero import (
    SignZeroSVARModel,
    _sample_zero_restricted_q,
    _zero_restriction_bases,
)
from stats_transformer.models.timeseries.identification.bootstrap import SVARBootstrap

def test_sign_zero_svar_model(tmp_path):
    # Create a synthetic dataset
    np.random.seed(42)
    T = 100
    y1 = np.random.randn(T)
    y2 = 0.5 * y1 + np.random.randn(T)
    y3 = -0.3 * y1 + 0.2 * y2 + np.random.randn(T)
    
    df = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3})
    
    # Create a temporary config file
    config_content = """
variables:
  - y1
  - y2
  - y3
shocks:
  - s1
  - s2
  - s3
restrictions:
  - shock: s1
    response: y1
    type: sign
    value: "+"
    horizon: 0
  - shock: s1
    response: y2
    type: sign
    value: "-"
    horizon: 0
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    
    # Test model
    model = SignZeroSVARModel(
        target_variables=['y1', 'y2', 'y3'],
        config_path=str(config_path),
        maxlags=1,
        max_draws=500,
        required_accepts=5
    )
    
    model.df_clean = df
    model.build_model()
    
    metrics = model.get_model_metrics()
    assert metrics['accepted_draws'] > 0
    assert metrics['total_draws'] > 0
    
    rep_draw = model.get_representative_draw()
    assert 'irf' in rep_draw
    assert 'rotation' in rep_draw
    
    # Check that restrictions hold in the accepted draws
    for irf in model.accepted_irfs:
        impact = irf[0] # horizon 0
        # s1 is index 0, y1 is index 0 -> +
        assert impact[0, 0] > 0
        # s1 is index 0, y2 is index 1 -> -
        assert impact[1, 0] < 0

def test_zero_restriction_construction_helper():
    """_sample_zero_restricted_q imposes zero restrictions by construction."""
    # One impact zero restriction for shock 0 (y3 does not respond to shock 1)
    zero_bases = [np.array([[0.0, 0.0, 1.0]]), np.zeros((0, 3)), np.zeros((0, 3))]
    Q = _sample_zero_restricted_q(zero_bases, 3, np.random.default_rng(0))
    assert abs(Q[2, 0]) <= 1e-10  # zero restriction holds to 1e-10
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-10)
    assert abs(np.linalg.det(Q)) == pytest.approx(1.0)

    # A horizon-1 zero restriction stacks Z_j with Z_j F(θ) = phi_1 row
    p = np.eye(3)
    phi1 = np.array([[0.2, 0.4, 0.1], [0.0, 0.3, 0.5], [0.1, 0.2, 0.7]])
    bases_h1 = _zero_restriction_bases(
        [{"shock": "s1", "response": "y2", "type": "zero", "horizon": 1}],
        {"s1": 0}, {"y1": 0, "y2": 1, "y3": 2},
        [np.eye(3), phi1], p, 3,
    )
    assert bases_h1[0].shape == (1, 3)
    np.testing.assert_allclose(bases_h1[0], phi1[1][None, :])
    Q = _sample_zero_restricted_q(bases_h1, 3, np.random.default_rng(0))
    # impact = P Q; response of y2 to shock 1 at horizon 1 = phi1[1] @ impact[:, 0]
    assert abs(phi1[1] @ Q[:, 0]) <= 1e-10

    # Violating the ARW ordering condition z_j <= n - j raises
    over_restricted = [np.vstack([np.eye(3), [[0.0, 1.0, 0.0]]]), np.zeros((0, 3)), np.zeros((0, 3))]
    with pytest.raises(ValueError, match="ordering condition"):
        _sample_zero_restricted_q(over_restricted, 3, np.random.default_rng(0))

def test_sign_zero_model_zero_restrictions_by_construction(tmp_path):
    """Deterministic: accepted draws satisfy zero restrictions to 1e-10 and signs hold."""
    np.random.seed(42)
    T = 100
    y1 = np.random.randn(T)
    y2 = 0.5 * y1 + np.random.randn(T)
    y3 = -0.3 * y1 + 0.2 * y2 + np.random.randn(T)
    df = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})

    config_content = """
variables:
  - y1
  - y2
  - y3
shocks:
  - s1
  - s2
  - s3
restrictions:
  - shock: s1
    response: y3
    type: zero
    horizon: 0
  - shock: s2
    response: y1
    type: zero
    horizon: 1
  - shock: s1
    response: y1
    type: sign
    value: "+"
    horizon: 0
  - shock: s2
    response: y3
    type: sign
    value: "+"
    horizon: 0
"""
    config_path = tmp_path / "zero_config.yaml"
    config_path.write_text(config_content)

    model = SignZeroSVARModel(
        target_variables=["y1", "y2", "y3"],
        config_path=str(config_path),
        maxlags=1,
        max_draws=200,
        required_accepts=10,
        seed=7,
    )
    model.df_clean = df
    model.build_model()

    assert len(model.accepted_rotations) == 10
    for irf in model.accepted_irfs:
        # shock s1 (index 0) has an exact zero impact on y3 (index 2) at horizon 0
        assert abs(irf[0, 2, 0]) <= 1e-10
        # shock s2 (index 1) has an exact zero effect on y1 at horizon 1
        assert abs(irf[1, 0, 1]) <= 1e-10
        # sign restrictions hold
        assert irf[0, 0, 0] > 0
        assert irf[0, 2, 1] > 0

def test_sign_zero_model_accepts_with_zero_restrictions(tmp_path):
    """With a zero restriction the sampler accepts a reasonable share of draws
    (on origin/main it accepts essentially never, since a random Haar rotation
    hits an exact zero with probability zero)."""
    rng = np.random.default_rng(123)
    T = 60
    y1 = rng.standard_normal(T)
    y2 = 0.5 * y1 + rng.standard_normal(T)
    y3 = -0.3 * y1 + 0.2 * y2 + rng.standard_normal(T)
    df = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})

    config_content = """
variables:
  - y1
  - y2
  - y3
shocks:
  - s1
  - s2
  - s3
restrictions:
  - shock: s1
    response: y3
    type: zero
    horizon: 0
  - shock: s1
    response: y1
    type: sign
    value: "+"
    horizon: 0
"""
    config_path = tmp_path / "zero_accept_config.yaml"
    config_path.write_text(config_content)

    model = SignZeroSVARModel(
        target_variables=["y1", "y2", "y3"],
        config_path=str(config_path),
        maxlags=1,
        max_draws=300,
        required_accepts=20,
        seed=7,
    )
    model.df_clean = df
    model.build_model()

    assert len(model.accepted_rotations) == 20
    assert model.total_draws < 300  # acceptance well above zero
    for irf in model.accepted_irfs:
        assert abs(irf[0, 2, 0]) <= 1e-10

_MAIN_NOZERO_ROTATIONS = np.array(
    [
      [[0.004487307929991413, 0.9999899319830882],
      [-0.9999899319830882, 0.004487307929991571]],
      [[0.062162444997626576, -0.9980660451252296],
      [-0.9980660451252296, -0.06216244499762679]],
      [[0.0898063439217549, 0.9959592464510821],
      [-0.9959592464510821, 0.08980634392175504]],
      [[0.3671536127683299, 0.9301603220043109],
      [-0.9301603220043109, 0.36715361276832986]],
      [[0.6015038377246473, -0.7988699100620333],
      [-0.7988699100620333, -0.6015038377246472]],
      [[0.8119566602960748, -0.5837177244189562],
      [-0.5837177244189562, -0.8119566602960748]],
      [[0.2668151136691068, -0.9637477341699652],
      [-0.9637477341699652, -0.26681511366910665]],
      [[0.8768983179004786, 0.48067591999528225],
      [-0.48067591999528225, 0.8768983179004787]],
      [[0.8539092474790668, -0.5204219413800055],
      [-0.5204219413800055, -0.8539092474790667]],
      [[0.7509890978427558, -0.6603146029896082],
      [-0.6603146029896082, -0.7509890978427557]],
      [[0.2951917445854544, -0.9554380324901224],
      [-0.9554380324901224, -0.29519174458545416]],
      [[0.7268732219873064, -0.6867716644983195],
      [-0.6867716644983195, -0.7268732219873062]],
      [[0.8725083380878862, 0.48859922223343244],
      [-0.48859922223343244, 0.8725083380878861]],
      [[0.7464808722419682, -0.6654068735569768],
      [-0.6654068735569768, -0.7464808722419681]],
      [[0.182220570896785, -0.9832576791167459],
      [-0.9832576791167459, -0.1822205708967851]],
      [[0.7978201925503039, -0.6028954638732955],
      [-0.6028954638732955, -0.7978201925503038]],
      [[0.12391256625926261, 0.9922931401168931],
      [-0.9922931401168931, 0.12391256625926252]],
      [[0.4216346200201775, 0.9067658171768719],
      [-0.9067658171768719, 0.42163462002017765]],
      [[0.2204826951360821, -0.9753908863350783],
      [-0.9753908863350783, -0.22048269513608199]],
      [[0.3934227755277251, -0.9193576669044867],
      [-0.9193576669044867, -0.39342277552772514]],
    ]
)

def test_sign_zero_no_zero_restrictions_draws_unchanged(tmp_path):
    """Without zero restrictions, accepted draws for a fixed seed are unchanged
    relative to origin/main (literal values generated on origin/main)."""
    rng = np.random.default_rng(123)
    T = 60
    y1 = rng.standard_normal(T)
    y2 = 0.5 * y1 + rng.standard_normal(T)
    df = pd.DataFrame({"y1": y1, "y2": y2})

    config_content = """
variables:
  - y1
  - y2
shocks:
  - s1
  - s2
restrictions:
  - shock: s1
    response: y1
    type: sign
    value: "+"
    horizon: 0
  - shock: s1
    response: y2
    type: sign
    value: "-"
    horizon: 0
"""
    config_path = tmp_path / "sign_only_config.yaml"
    config_path.write_text(config_content)

    model = SignZeroSVARModel(
        target_variables=["y1", "y2"],
        config_path=str(config_path),
        maxlags=1,
        max_draws=200,
        required_accepts=20,
        seed=7,
    )
    model.df_clean = df
    model.build_model()

    np.testing.assert_allclose(
        np.array(model.accepted_rotations), _MAIN_NOZERO_ROTATIONS, atol=1e-10
    )

def test_svar_bootstrap(tmp_path):
    np.random.seed(42)
    T = 50
    y1 = np.random.randn(T)
    y2 = 0.5 * y1 + np.random.randn(T)
    df = pd.DataFrame({'y1': y1, 'y2': y2})
    
    config_content = """
variables:
  - y1
  - y2
shocks:
  - s1
  - s2
restrictions:
  - shock: s1
    response: y1
    type: sign
    value: "+"
    horizon: 0
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    
    model = SignZeroSVARModel(
        target_variables=['y1', 'y2'],
        config_path=str(config_path),
        maxlags=1,
        max_draws=100,
        required_accepts=1 # small for fast test
    )
    model.df_clean = df
    model.build_model()
    
    bootstrap = SVARBootstrap(model, n_bootstrap=2, seed=42)
    results = bootstrap.run()
    
    assert len(results) > 0
    
    lower, upper = bootstrap.get_confidence_intervals()
    assert lower.shape == results[0]['irf'].shape
    assert upper.shape == results[0]['irf'].shape
