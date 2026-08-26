"""
test_isotherm_classification.py — ground-truth tests for classify_isotherm.

`bet_analysis.py` currently has no test coverage at all. These tests close that
gap for the isotherm classifier using synthetic isotherms whose IUPAC type
follows from the generating equation (see synthetic_isotherms.py).
"""

import numpy as np
import pytest

from bet_analysis import classify_isotherm
from synthetic_isotherms import CASES, EXPECTED, build, negligible_desorption


# --------------------------------------------------------------------------
# Cases the classifier already handles correctly — these guard against
# regressions when D2 / S9 are fixed.
# --------------------------------------------------------------------------
WORKING = ["TypeIa_noHyst", "TypeIb_noHyst", "TypeIII_noHyst",
           "TypeVI_noHyst", "TypeIV_H1", "TypeV_H2"]


@pytest.mark.parametrize("case", WORKING)
def test_classifier_matches_ground_truth(case):
    ads, des = build(case)
    assert classify_isotherm(ads, des)["type"] == EXPECTED[case]


# --------------------------------------------------------------------------
# S9 — Type II is reachable for a genuine unrestricted multilayer isotherm.
#
# classify_isotherm reaches "Type II" whenever the low-p/p0 region is concave
# (strong adsorbate–adsorbent interaction); IUPAC 2015 defines Type II as
# *unrestricted* monolayer-multilayer adsorption, so uptake rises without limit
# as p/p0 -> 1 and no plateau exists.
# --------------------------------------------------------------------------
def test_type_II_is_reachable():
    ads, des = build("TypeII_noHyst")
    assert classify_isotherm(ads, des)["type"] == "Type II"


def test_type_III_requires_a_convex_isotherm():
    """Type III means weak adsorbate-adsorbent interaction (convex isotherm).

    A strongly concave isotherm (BET, C = 120) must NOT be reported as
    Type III, whatever its high-pressure plateau.
    """
    ads, des = build("TypeII_noHyst")
    result = classify_isotherm(ads, des)
    assert result["concave_low"] is True
    assert result["type"] != "Type III"


# --------------------------------------------------------------------------
# D2 — hysteresis is inferred from loop area, not array presence.
#
# `has_hyst` must respond to the normalised loop area, not merely to a
# non-empty desorption array. A desorption branch 0.1% above the adsorption
# branch is not a hysteresis loop by any physical definition (normalised area
# ~4e-4).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["TypeIa_noHyst", "TypeIb_noHyst"])
def test_negligible_loop_does_not_force_type_IV(case):
    ads, _ = build(case)
    des = negligible_desorption(ads)
    assert classify_isotherm(ads, des)["type"] == EXPECTED[case]


def test_negligible_loop_area_is_genuinely_negligible():
    """Guards the probe itself: if this loop were real, the D2 tests prove nothing."""
    ads, _ = build("TypeIa_noHyst")
    des = negligible_desorption(ads)
    gap = np.clip(np.interp(des[:, 0], ads[:, 0], ads[:, 1]) * 0 +
                  des[:, 1] - np.interp(des[:, 0], ads[:, 0], ads[:, 1]), 0, None)
    norm_area = float(np.trapezoid(gap, des[:, 0])) / ads[:, 1].max()
    assert norm_area < 0.001, f"probe loop is not negligible: {norm_area:.6f}"


# --------------------------------------------------------------------------
# Threshold sanity (S2) — the absolute cutoffs should not silently exclude
# whole classes of sample.
# --------------------------------------------------------------------------
def test_steep_init_requires_at_least_two_points_below_pp0_0_1():
    """`very_low_mask.sum() > 1` means a sparse low-pressure run disables the
    micropore test entirely, regardless of the sample. Documented, not asserted
    as correct behaviour."""
    ads, _ = build("TypeIa_noHyst")
    assert (ads[:, 0] < 0.1).sum() > 1, "fixture must exercise the micropore path"
