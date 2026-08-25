"""Unit tests for Langmuir monolayer-adsorption analysis."""

import numpy as np
import pytest

from langmuir import (
    N2_LANGMUIR_FACTOR,
    fit_langmuir_window,
    format_langmuir_report,
    langmuir_linear_y,
)


def _langmuir_isotherm(n_m=10.0, K=50.0, p=None, noise=0.0, seed=0):
    if p is None:
        p = np.linspace(0.01, 0.30, 15)
    n = n_m * K * p / (1.0 + K * p)
    if noise > 0:
        rng = np.random.default_rng(seed)
        n = n * (1.0 + rng.normal(0.0, noise, size=len(p)))
    return p, n


def test_ideal_langmuir_recovers_parameters():
    n_m_true = 10.0
    K_true = 50.0
    p, n = _langmuir_isotherm(n_m=n_m_true, K=K_true)
    result = fit_langmuir_window(p, n)

    assert result["n_m"] == pytest.approx(n_m_true, rel=1e-6)
    assert result["K"] == pytest.approx(K_true, rel=1e-6)
    assert result["S_Langmuir"] == pytest.approx(
        n_m_true * N2_LANGMUIR_FACTOR, rel=1e-6
    )
    assert result["R2"] == pytest.approx(1.0, abs=1e-10)
    assert result["sigma_S_Langmuir"] == pytest.approx(0.0, abs=1e-9)
    assert result["physical_fit"] is True


def test_noisy_langmuir_has_positive_uncertainty():
    rng = np.random.default_rng(42)
    p, n = _langmuir_isotherm(n_m=10.0, K=50.0)
    n_noisy = n * (1.0 + rng.normal(0.0, 0.005, size=len(p)))
    result = fit_langmuir_window(p, n_noisy)

    assert result["sigma_S_Langmuir"] > 0
    assert result["sigma_K"] > 0
    assert result["physical_fit"] is True


def test_langmuir_rejects_invalid_inputs():
    p = np.linspace(0.01, 0.30, 15)
    n = 10.0 * 50.0 * p / (1.0 + 50.0 * p)

    # too few points
    with pytest.raises(ValueError):
        fit_langmuir_window(p[:2], n[:2])

    # n <= 0
    n_bad = n.copy()
    n_bad[0] = 0.0
    with pytest.raises(ValueError):
        fit_langmuir_window(p, n_bad)

    # p <= 0
    p_bad = p.copy()
    p_bad[0] = 0.0
    with pytest.raises(ValueError):
        fit_langmuir_window(p_bad, n)

    # unequal array lengths
    with pytest.raises(ValueError):
        fit_langmuir_window(p, n[:-1])

    with pytest.raises(ValueError, match="0 < p/p0 < 1"):
        fit_langmuir_window(
            np.array([0.05, 0.10, 1.00]),
            np.array([1.0, 2.0, 3.0]),
        )

    with pytest.raises(ValueError, match="0 < p/p0 < 1"):
        fit_langmuir_window(
            np.array([0.05, 0.10, 1.10]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_langmuir_area_uses_n2_factor():
    p, n = _langmuir_isotherm(n_m=12.0, K=40.0)
    result = fit_langmuir_window(p, n)
    assert result["S_Langmuir"] == pytest.approx(
        result["n_m"] * N2_LANGMUIR_FACTOR, rel=1e-12
    )


def test_langmuir_linear_y_is_p_over_n():
    p = np.array([0.05, 0.10, 0.20])
    n = np.array([2.0, 4.0, 5.0])
    y = langmuir_linear_y(p, n)
    assert np.allclose(y, p / n)


def test_format_langmuir_report_marks_fail_when_unphysical():
    result = {
        "p_min": 0.05,
        "p_max": 0.30,
        "n_points": 6,
        "S_Langmuir": 120.35,
        "sigma_S_Langmuir": 2.18,
        "n_m": 27.65,
        "sigma_n_m": 0.50,
        "K": 42.7,
        "sigma_K": 5.9,
        "R2": 0.999120,
        "physical_fit": False,
    }
    report = format_langmuir_report(result, "Sample")
    assert "FAIL" in report
    assert "S_Langmuir" in report
