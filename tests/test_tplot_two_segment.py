"""
test_tplot_two_segment.py — ground-truth tests for the two-segment t-plot.

The t-plot construction of Lippens & de Boer fits two lines:

    line 1 (through the origin, low-t)  -> total surface area
    line 2 (free intercept, high-t)     -> external surface area + micropore volume

These tests build synthetic t-plots from closed-form expressions so the expected
answer follows from the construction, not from a hand-placed number.
"""

import numpy as np
import pytest

from tplot_analysis import (
    N2_TPLOT_SLOPE_FACTOR,
    N2_STP_TO_LIQUID,
    fit_two_segment,
)


def _two_segment(t, slope1, slope2, t_bend):
    """Continuous two-segment t-plot: line 1 (origin) below t_bend, line 2 above."""
    intercept2 = (slope1 - slope2) * t_bend
    return np.where(t < t_bend, slope1 * t, slope2 * t + intercept2)


def _grid(t_min=3.5, t_max=6.5, n=20):
    return np.linspace(t_min, t_max, n)


def test_nonporous_single_line_recovers_no_microporosity():
    # A straight line through the origin -> non-porous: no bend, V_micro ~ 0,
    # external ~= total.
    slope = 5.0
    t = _grid()
    v = slope * t

    with pytest.warns(UserWarning):
        r = fit_two_segment(t, v, 3.5, 6.5)

    assert r["V_micro_cm3g"] == pytest.approx(0.0, abs=1e-4)
    assert r["S_external_m2g"] == pytest.approx(r["S_total_m2g"], rel=0.01)
    assert r["S_total_m2g"] == pytest.approx(slope * N2_TPLOT_SLOPE_FACTOR, rel=0.01)
    # No steeper micropore-filling segment -> flagged, not silently forced.
    assert r["flags"]["slope_order_ok"] is False
    assert r["low_confidence"] is True


def test_microporous_two_segment_recovers_quantities():
    slope1 = 8.0
    slope2 = 3.0
    t_bend = 4.5
    intercept2 = (slope1 - slope2) * t_bend

    t = _grid()
    v = _two_segment(t, slope1, slope2, t_bend)

    r = fit_two_segment(t, v, 3.5, 6.5)

    assert r["S_total_m2g"] == pytest.approx(
        slope1 * N2_TPLOT_SLOPE_FACTOR, rel=0.01)
    assert r["S_external_m2g"] == pytest.approx(
        slope2 * N2_TPLOT_SLOPE_FACTOR, rel=0.01)
    assert r["S_micro_m2g"] == pytest.approx(
        (slope1 - slope2) * N2_TPLOT_SLOPE_FACTOR, rel=0.01)
    assert r["V_micro_cm3g"] == pytest.approx(
        intercept2 * N2_STP_TO_LIQUID, rel=0.01)
    assert r["t_bend_A"] == pytest.approx(t_bend, rel=0.05)
    assert r["2t_nm"] == pytest.approx(2 * t_bend / 10.0, rel=0.05)
    # A valid construction satisfies every constraint.
    assert r["flags"]["slope_order_ok"] is True
    assert r["flags"]["intercept_ok"] is True
    assert r["flags"]["s_order_ok"] is True
    assert r["flags"]["bend_ok"] is True


def test_convex_isotherm_flags_slope_intercept_and_area():
    # Convex (upward-curving) t-plot: line 1 is *less* steep than line 2, the
    # line-2 intercept is negative, and the external area exceeds the total.
    t = _grid()
    v = (t ** 2) / 12.0

    with pytest.warns(UserWarning):
        r = fit_two_segment(t, v, 3.5, 6.5)

    assert r["flags"]["slope_order_ok"] is False
    assert r["flags"]["intercept_ok"] is False
    assert r["flags"]["s_order_ok"] is False
    assert "slope_1 <= slope_2" in r["warnings"]
    assert "intercept_2 < 0" in r["warnings"]
    assert "S_external > S_total" in r["warnings"]


def test_small_bend_flags_unreliable_diameter():
    # A bend at t_bend = 3.2 A gives 2t = 0.64 nm < 0.7 nm -> unreliable.
    # The window extends below the default floor so line 1 has >= 3 points.
    slope1 = 8.0
    slope2 = 3.0
    t_bend = 3.2
    t = _grid(t_min=2.5, t_max=6.5, n=40)
    v = _two_segment(t, slope1, slope2, t_bend)

    with pytest.warns(UserWarning, match="2t"):
        r = fit_two_segment(t, v, 2.5, 6.5)

    assert r["flags"]["bend_ok"] is False
    assert "2t < 0.7 nm" in r["warnings"]
    assert r["2t_nm"] < 0.7


def test_too_few_points_raises():
    # Fewer than 2 * MIN_SEGMENT_POINTS points in the window cannot be split.
    t = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
    v = 5.0 * t
    with pytest.raises(ValueError, match="at least 6 points"):
        fit_two_segment(t, v, 3.5, 6.5)
