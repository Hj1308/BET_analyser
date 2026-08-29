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
    TPlotAnalyser,
    fit_tplot_model,
    fit_two_segment,
    harkins_jura_t,
)


def _two_segment(t, slope1, slope2, t_bend):
    """Continuous two-segment t-plot: line 1 (origin) below t_bend, line 2 above."""
    intercept2 = (slope1 - slope2) * t_bend
    return np.where(t < t_bend, slope1 * t, slope2 * t + intercept2)


def _grid(t_min=3.5, t_max=6.5, n=20):
    return np.linspace(t_min, t_max, n)


def test_nonporous_single_line_recovers_no_microporosity():
    # A straight line through the origin -> non-porous: no bend, V_micro = 0,
    # external == total.
    slope = 5.0
    t = _grid()
    v = slope * t

    r = fit_tplot_model(t, v, 3.5, 6.5)

    assert r["model"] == "single_line"
    assert r["bend_detected"] is False
    assert r["S_total_m2g"] == pytest.approx(slope * N2_TPLOT_SLOPE_FACTOR, rel=0.01)
    assert r["S_external_m2g"] == pytest.approx(r["S_total_m2g"], rel=0.01)
    assert r["S_micro_m2g"] == 0.0
    assert r["V_micro_cm3g"] == 0.0
    assert r["2t_nm"] is None
    # No steeper micropore-filling segment -> no bend is reported at all.
    assert r["no_bend_reason"]


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


def test_convex_isotherm_rejects_two_segment_and_falls_back():
    # Convex (upward-curving) t-plot: line 1 is *less* steep than line 2, so no
    # physically valid two-segment decomposition exists. The low-level fit
    # returns None and the model falls back to a single line — S_external >
    # S_total is never reported.
    t = _grid()
    v = (t ** 2) / 12.0

    assert fit_two_segment(t, v, 3.5, 6.5) is None

    r = fit_tplot_model(t, v, 3.5, 6.5)
    assert r["model"] == "single_line"
    assert r["bend_detected"] is False
    assert r["S_external_m2g"] == pytest.approx(r["S_total_m2g"], rel=0.01)


def test_small_bend_flags_unreliable_diameter():
    # A bend at t_bend = 3.2 A gives 2t = 0.64 nm < 0.7 nm -> unreliable.
    # The window and split floor extend below 3.5 A so the split can sit at 3.2.
    slope1 = 8.0
    slope2 = 3.0
    t_bend = 3.2
    t = _grid(t_min=2.5, t_max=6.5, n=40)
    v = _two_segment(t, slope1, slope2, t_bend)

    with pytest.warns(UserWarning, match="2t"):
        r = fit_two_segment(t, v, 2.5, 6.5, split_t_min=2.5)

    assert r["flags"]["bend_ok"] is False
    assert "2t < 0.7 nm" in r["warnings"]
    assert r["2t_nm"] < 0.7


def test_split_floor_confines_line2():
    # With split_t_min at the Harkins-Jura floor, the split never puts a point
    # below 3.5 A into line 2.
    slope1 = 8.0
    slope2 = 3.0
    t_bend = 4.5
    t = _grid(t_min=2.5, t_max=6.5, n=40)
    v = _two_segment(t, slope1, slope2, t_bend)

    r = fit_two_segment(t, v, 2.5, 6.5, split_t_min=3.5)
    # the split index sits at or above the floor
    assert t[r["split_index"]] >= 3.5
    assert r["n_points_1"] >= 3 and r["n_points_2"] >= 3


def test_too_few_points_raises():
    # Fewer than 2 * MIN_SEGMENT_POINTS points in the window cannot be split.
    t = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
    v = 5.0 * t
    with pytest.raises(ValueError, match="at least 6 points"):
        fit_two_segment(t, v, 3.5, 6.5)


def test_sufficiency_gate_refuses_micropore_when_undersampled():
    # Only 2 adsorption points below p/p0 = 0.08 -> micropore quantities are
    # returned as None (never 0.0) and the reason names the required region.
    p = np.array([0.05, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    v = np.linspace(5.0, 100.0, len(p))
    tp = TPlotAnalyser(p, v, s_bet=100.0, total_pore_volume=0.3)

    r = tp.full_tplot_report()

    assert r["micropore_analysis_possible"] is False
    assert r["n_points_below_pp008"] == 2
    assert r["V_micro_cm3g"] is None
    assert r["S_total_m2g"] is None
    assert r["S_micro_m2g"] is None
    assert r["t_bend_A"] is None
    assert r["2t_nm"] is None
    # external area is still reported from line 2
    assert r["S_external_m2g"] is not None
    assert "p/p0 = 0.08" in r["micropore_analysis_reason"]


def test_sufficiency_gate_passes_with_enough_low_pressure_points():
    p = np.array([0.001, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # A physical two-segment t-plot (steep origin line below t_bend = 4.5 A,
    # gentler free line above) so the fit is physical and emits no warnings.
    t = harkins_jura_t(p)
    v = _two_segment(t, slope1=8.0, slope2=3.0, t_bend=4.5)
    tp = TPlotAnalyser(p, v, s_bet=100.0, total_pore_volume=0.3)

    r = tp.full_tplot_report()

    assert r["micropore_analysis_possible"] is True
    assert r["V_micro_cm3g"] is not None
    assert r["S_total_m2g"] is not None
    assert r["t_bend_A"] is not None


def test_gate_requires_primary_filling_point():
    # 3+ points below 0.08 but none below 0.015 (9.xls-like) -> refused: the
    # wider-micropore points do not carry the steep primary-filling slope.
    p = np.array([0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    v = np.linspace(5.0, 100.0, len(p))
    tp = TPlotAnalyser(p, v, s_bet=100.0, total_pore_volume=0.3)

    r = tp.full_tplot_report()

    assert r["micropore_analysis_possible"] is False
    assert r["n_points_below_pp008"] == 3
    assert r["n_points_below_pp0015"] == 0
    assert r["V_micro_cm3g"] is None
    assert "p/p0 = 0.015" in r["micropore_analysis_reason"]
    assert "p/p0 = 0.08" not in r["micropore_analysis_reason"]  # first layer passed
