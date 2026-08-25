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
