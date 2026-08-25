"""Unit tests for Rouquerol BET range selection."""

import numpy as np
import pytest

from rouquerol import (
    N2_BET_FACTOR,
    diagnose_instrument_range,
    fit_bet_window,
    monolayer_pressure_theory,
    rouquerol_transform,
    select_bet_range,
    bet_sensitivity_heatmap,
)


def _ideal_bet_isotherm(C=80.0, Vm=10.0, p=None):
    if p is None:
        p = np.linspace(0.01, 0.40, 20)
    theta = C * p / ((1.0 - p) * (1.0 + (C - 1.0) * p))
    return p, Vm * theta, C, Vm


def test_rouquerol_transform_decreases_after_micropore_knee():
    p = np.array([0.01, 0.05, 0.10, 0.20, 0.40, 0.70])
    n = np.array([8.0, 9.5, 10.0, 10.4, 11.0, 12.0])
    t = rouquerol_transform(p, n)
    assert t[0] > t[-1]


def test_ideal_isotherm_recovers_C_and_Vm():
    p, n, C, Vm = _ideal_bet_isotherm()
    fit = fit_bet_window(p, n)
    assert fit["C"] == pytest.approx(C, rel=1e-6)
    assert fit["Vm"] == pytest.approx(Vm, rel=1e-6)
    assert fit["S_BET"] == pytest.approx(Vm * N2_BET_FACTOR, rel=1e-6)
    assert fit["R2"] == pytest.approx(1.0, abs=1e-10)


def test_select_range_is_rouquerol_valid_on_ideal_data():
    p, n, C, Vm = _ideal_bet_isotherm()
    result = select_bet_range(p, n)
    best = result["best"]
    assert best is not None
    assert best.valid
    assert result["n_valid"] >= 1
    assert best.C == pytest.approx(C, rel=1e-4)
    assert best.Vm == pytest.approx(Vm, rel=1e-4)
    assert best.c1_C_positive and best.c2_n1mp_increasing
    assert best.c3_nm_in_range and best.c4_pm_consistency


def test_criterion4_theory_matches_ideal_monolayer_pressure():
    C = 80.0
    p_m = monolayer_pressure_theory(C)
    assert p_m == pytest.approx(1.0 / (np.sqrt(C) + 1.0))


def test_instrument_window_diagnosis_flags_invalid_high_pp0():
    p, n, _, _ = _ideal_bet_isotherm(p=np.linspace(0.05, 0.90, 25))
    # last 4 points sit past the Rouquerol maximum for this Type-II-like curve
    win = diagnose_instrument_range(p, n, len(p) - 4, len(p) - 1)
    assert win.c2_n1mp_increasing is False or win.valid is False


def test_too_few_points_returns_none_best_only_if_empty():
    p = np.array([0.05, 0.10, 0.15])
    n = np.array([1.0, 1.2, 1.4])
    result = select_bet_range(p, n, min_points=4)
    assert result["best"] is None or result["n_candidates"] == 0


def test_noisy_type_iv_isotherm_recovers_surface_area():
    """Type IV with capillary condensation: the linearity filter must reject
    the full-range window, which passes all four Rouquerol criteria but has
    R² ≈ 0.92 and overestimates S_BET by ~30 %."""
    rng = np.random.default_rng(42)
    C_true, Vm_true = 120.0, 22.0
    p = np.array([0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.24, 0.28,
                  0.32, 0.36, 0.42, 0.48, 0.55, 0.62, 0.70, 0.80, 0.90, 0.99])
    theta = C_true * p / ((1.0 - p) * (1.0 + (C_true - 1.0) * p))
    n = Vm_true * theta
    n = n * (1.0 + 0.6 * np.clip((p - 0.45) / 0.5, 0, 1) ** 2)
    n = n * (1.0 + rng.normal(0, 0.003, len(p)))

    result = select_bet_range(p, n)
    best = result["best"]
    S_true = Vm_true * N2_BET_FACTOR
    assert best is not None and best.valid
    assert best.R2 >= 0.999
    assert abs(best.S_BET - S_true) / S_true < 0.05


def test_bet_uncertainty_propagation():
    """σ(S_BET) and σ(C) come from the linregress standard errors via
    first-order propagation: ~0 for a perfect fit, positive for noisy
    data, reproducing σ_S = S_BET·√(σ_slope² + σ_intercept²)/(slope+intercept),
    and bracketing the true surface area."""
    p, n, C, Vm = _ideal_bet_isotherm()
    fit = fit_bet_window(p, n)
    assert fit["sigma_S_BET"] == pytest.approx(0.0, abs=1e-9)
    assert fit["sigma_C"] == pytest.approx(0.0, abs=1e-9)

    rng = np.random.default_rng(7)
    n_noisy = n * (1.0 + rng.normal(0.0, 0.005, size=len(p)))
    fit = fit_bet_window(p, n_noisy)
    assert fit["sigma_slope"] > 0 and fit["sigma_intercept"] > 0
    expected = (abs(fit["S_BET"])
                * np.hypot(fit["sigma_slope"], fit["sigma_intercept"])
                / abs(fit["slope"] + fit["intercept"]))
    assert fit["sigma_S_BET"] == pytest.approx(expected, rel=1e-12)
    assert fit["sigma_C"] > 0
    S_true = Vm * N2_BET_FACTOR
    assert abs(fit["S_BET"] - S_true) < 3.0 * fit["sigma_S_BET"]


def test_bet_sensitivity_heatmap_dimensions_and_stability():
    """Heatmap returns NxN matrices; ideal isotherm gives stable S_BET."""
    p, n, C, Vm = _ideal_bet_isotherm()
    result = bet_sensitivity_heatmap(p, n)
    N = result["n_points"]
    assert result["s_bet"].shape == (N, N)
    assert result["valid"].shape == (N, N)
    assert result["r2"].shape == (N, N)
    assert result["valid"].sum() > 0
    S_true = Vm * N2_BET_FACTOR
    valid_s = result["s_bet"][result["valid"]]
    assert np.all(np.abs(valid_s - S_true) / S_true < 0.01)
    for i in range(N):
        for j in range(i):
            assert np.isnan(result["s_bet"][i, j])
