"""
tplot_analysis.py — T-Plot Analysis (two-segment construction)
==============================================================
Determines:
  - Total surface area        (line 1, through the origin)
  - External surface area     (line 2 slope; meso + macro)
  - Micropore surface area    (total - external)
  - Micropore volume          (line 2 intercept, via the Gurvich factor)
  - Mean pore diameter (2t)   (bend point of the two lines)

Method (two-segment t-plot)
---------------------------
The t-method fits *two* lines to the adsorption isotherm plotted as adsorbed
amount V against statistical film thickness t (a "t-plot"):

    Line 1 — through the ORIGIN, low-t region:
             slope -> TOTAL surface area
    Line 2 — after micropore filling is complete:
             slope     -> EXTERNAL surface area
             intercept -> MICROPORE VOLUME (Gurvich factor)
    Micropore surface area = total - external
    Bend point (intersection of lines 1 and 2) -> mean pore radius t;
             2t = mean pore diameter, unreliable if 2t < 0.7 nm

A non-porous sample gives a single straight line through the origin (line 1
only); a mesoporous sample deviates *upward* from line 1 as capillary
condensation sets in. This module fits both segments and reports the derived
quantities; a sample with no detectable micropore region is reported with a
"no meaningful bend" flag rather than a forced split.

Data-sufficiency gate
---------------------
Line 1 needs adsorption points in the micropore-filling region (p/p0 < 0.08,
Thommes et al. 2015 §6.1). If the data has fewer than ``MIN_LINE1_POINTS``
points there, the micropore quantities (V_micro, S_micro, S_total, t_bend, 2t)
cannot be determined and are returned as ``None`` with
``micropore_analysis_possible`` False; the external surface area (line 2) is
still reported on its own. This is a *measurement limitation*, not a value of
zero — returning 0.0 here would read as "confirmed no micropores" when the
truth is "this measurement cannot answer the question".

Reference t-curve
-----------------
The reference t-curve must be measured on a non-porous sample with the *same
surface chemistry* as the analyte (Microtrac AppNote B-AD-009). Model formulas
are offered alongside measured references:

    Equation 1 (definition of the reference t-curve):
        t = 0.354 * V/Vm   [nm]   (N2 @ 77.4 K, hexagonal close packing)

Two model formulas are implemented here; the choice is explicit via the
``reference_curve`` argument (default "harkins-jura"):

  * "harkins-jura" — t = sqrt(13.99 / (0.034 - log10(P/P0)))  [Angstrom]
      derives from oxidic surfaces; treat cautiously for carbonaceous or
      non-polar surfaces.
  * "halsey"        — t = 3.54 * (-5 / ln(P/P0))**(1/3)        [Angstrom]

On the two instrument files that carry their own t-plot values (`10.xls`,
`14H.xls`), Halsey reproduced the instrument's *total* area `a1` to 2.3-2.5 %
versus 8.6-12 % for Harkins-Jura. Harkins-Jura remains the default only for
backward compatibility; the choice is exposed and should be revisited per
sample. Separately, `S_external/S_BET` rises monotonically with falling BET C
across the five audit samples, crossing 1.0 near C ≈ 65 (g-OH 1.489 at C 9.3,
9(BC) 1.234 at 32.1, 13BgOH 1.176 at 34.5, 10 1.069 at 57.6, 14H 0.960 at
95.8). With n = 5 this is indicative, not proven — it is reported here as a
correlation, not asserted as causation.

A tabulated reference curve (e.g. graphitised carbon black, GCB) can be added
later as a data file: register it in ``REFERENCE_CURVES`` (a plain
name -> callable mapping) without touching the fitting code. The fitting only
sees the resulting t array.

References
----------
  * Lippens, B. C.; de Boer, J. H.  J. Catal. 1964, 3, 32-37.
  * de Boer, J. H.; Lippens, B. C.; Linsen, B. G.; Broekhoff, J. C. P.;
    van den Heuvel, A.; Osinga, T. J.  J. Colloid Interface Sci. 1966, 21,
    405-414.
  * Thommes, M. et al.  Pure Appl. Chem. 2015, 87 (9-10), 1051-1069,
    DOI 10.1515/pac-2014-1117 (IUPAC Technical Report).
  * Microtrac AppNote B-AD-009, "Structural investigation by t-plot method
    for nonporous, microporous, and mesoporous materials (basic edition)".
  * Microtrac AppNote B-AD-010 (worked activated-carbon-fibre example).

Author  : Hoda Jafari | github.com/Hj1308
License : MIT
"""

import argparse
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from bet_analysis import N2_TPLOT_SLOPE_FACTOR, N2_STP_TO_LIQUID

# ── Publication style (matches bet_analysis.py) ────────────────
plt.rcParams.update({
    "font.family"     : "serif",
    "font.serif"      : ["Times New Roman", "DejaVu Serif"],
    "font.size"       : 10,
    "axes.labelsize"  : 11,
    "axes.titlesize"  : 11,
    "figure.dpi"      : 150,
    "savefig.dpi"     : 300,
    "savefig.bbox"    : "tight",
    "xtick.direction" : "in",
    "ytick.direction" : "in",
    "xtick.top"       : True,
    "ytick.right"     : True,
})

C_MICRO = "#2166AC"   # blue  — matches BET tool palette
C_EXT   = "#D6604D"   # red-orange
C_FIT   = "#1A7A4A"   # green
C_TOTAL = "#7B3F9E"   # purple — line 1 (total)


# ══════════════════════════════════════════════════════════════
# REFERENCE THICKNESS EQUATIONS
# ══════════════════════════════════════════════════════════════

def harkins_jura_t(p_rel: np.ndarray) -> np.ndarray:
    """
    Statistical film thickness by Harkins-Jura equation.

    t (Å) = sqrt(13.99 / (0.034 - log10(P/P0)))

    Valid range: 0.08 < P/P0 < 0.60 (outside this, other equations preferred).
    Derives from oxidic surfaces; see module docstring for the caveat.

    Parameters
    ----------
    p_rel : array — relative pressure P/P0  (0 < p_rel < 1)

    Returns
    -------
    np.ndarray — film thickness in Angstrom (Å)
    """
    p_rel = np.clip(p_rel, 1e-9, 1 - 1e-9)
    return np.sqrt(13.99 / (0.034 - np.log10(p_rel)))


def halsey_t(p_rel: np.ndarray) -> np.ndarray:
    """
    Statistical film thickness by the Halsey equation.

    t (Å) = 3.54 * (-5 / ln(P/P0))**(1/3)

    Parameters
    ----------
    p_rel : array — relative pressure P/P0  (0 < p_rel < 1)

    Returns
    -------
    np.ndarray — film thickness in Angstrom (Å)
    """
    p_rel = np.clip(p_rel, 1e-9, 1 - 1e-9)
    return 3.54 * (-5.0 / np.log(p_rel)) ** (1.0 / 3.0)


# Name -> callable registry for reference t-curves. The fitting code below
# only consumes the resulting t array, so a tabulated curve (e.g. GCB from
# AppNote B-AD-010) can be added later as a data file: implement a callable
# that interpolates p/p0 -> t and register it here.
REFERENCE_CURVES = {
    "harkins-jura": harkins_jura_t,
    "halsey": halsey_t,
}


# The Harkins-Jura film thickness is stated valid for 0.08 < p/p0 < 0.60
# (see harkins_jura_t above). That range is a property of the *reference
# t-curve*, not of the sample, so it is enforced separately from the fit:
#   - below p/p0 = 0.08, micropore filling is still in progress, which is
#     exactly the regime line 1 (total surface area) models. p/p0 = 0.08
#     corresponds to t = 3.52 Å; 3.5 Å is the rounded floor.
#   - above p/p0 = 0.60, the t-curve is no longer a reliable film thickness.
#     6.5 Å is the conservative ceiling (p/p0 ≈ 0.5, safely inside the range).
# HJ_VALID_T_MAX therefore guards t-curve *validity*, not fit quality. The fit
# may additionally be capped LOWER, at the sample's capillary-condensation
# onset, which is read from the data by condensation_onset_t() — see fit_tplot.
HJ_VALID_T_MIN = 3.5
HJ_VALID_T_MAX = 6.5

# Each segment of the two-segment fit needs this many points to be meaningful.
MIN_SEGMENT_POINTS = 3

# B-AD-009: the mean pore diameter 2t is unreliable below 0.7 nm.
BEND_2T_MIN_NM = 0.7

# ── Capillary-condensation onset (data-derived fit ceiling) ─────
# The t-plot is only linear up to the onset of capillary condensation; past it
# the isotherm rises steeply and a linear t-plot is undefined. That onset is a
# property of the *sample* (its pore size), so it is detected from the data
# rather than hardcoded. It is looked for only in the mesopore pressure range.
_CONDENSATION_LOOK_P_MIN = 0.40
_CONDENSATION_LOOK_P_MAX = 0.95
_CONDENSATION_CURVATURE_FRACTION = 0.25

# ── Line-1 (micropore) bounds and sufficiency gate ──────────────
# Line 1 models micropore filling, which occurs below p/p0 ~ 0.08 and mostly
# below ~0.015 (Thommes et al. 2015 §6.1). It must therefore be allowed BELOW
# the Harkins-Jura validity floor (HJ_VALID_T_MIN) that bounds line 2. The
# line-1 floor is anchored at p/p0 = 0.005 (the low end of the primary filling
# range): that is t ~ 2.45 Å on the Harkins-Jura curve and t ~ 3.47 Å on the
# Halsey curve. The per-curve value is computed by line1_t_min().
LINE1_P_MIN = 0.005
LINE1_T_MIN = 2.4     # rounded Harkins-Jura t at LINE1_P_MIN; used as a UI floor

# Data-sufficiency gate (two layered). Line 1 needs enough adsorption points in
# the micropore region p/p0 < LINE1_P_MAX, AND at least one in the *primary*
# filling region p/p0 < LINE1_P_PRIMARY.
MIN_LINE1_POINTS = 3
LINE1_P_MAX = 0.08

# Thommes et al. (2015) §6.1 distinguishes primary micropore filling (very low
# p/p0) from the secondary filling of wider micropores over p/p0 ~ 0.01-0.15;
# Cychosz & Thommes (2018) §3 place micropore filling below p/p0 ~ 0.015.
# Points between 0.015 and 0.08 sample only the wider range and do not carry
# the steep primary-filling slope that line 1 must measure, so a count below
# 0.08 alone is not sufficient.
LINE1_P_PRIMARY = 0.015
MIN_LINE1_PRIMARY_POINTS = 1


def line1_t_min(reference_curve: str) -> float:
    """Line-1 floor thickness (Å) for a reference curve at p/p0 = LINE1_P_MIN."""
    return float(REFERENCE_CURVES[reference_curve](np.asarray([LINE1_P_MIN]))[0])


def condensation_onset_t(p, v, t) -> float:
    """Data-derived upper fit bound (Å), just below the capillary-condensation rise.

    ``HJ_VALID_T_MAX`` guards the Harkins-Jura *t-curve* validity
    (0.08 < p/p0 < 0.60), not the fit. Separately, the t-plot is only linear up
    to the onset of capillary condensation, which is a property of the *sample*
    (its pore size). That onset is detected here from the data rather than
    hardcoded: the condensation step is the dominant positive curvature of the
    isotherm in the mesopore range, so the onset is the first point, scanning up
    in p/p0, where ``d²V/d(log p/p0)²`` reaches
    ``_CONDENSATION_CURVATURE_FRACTION`` of its peak in that range.

    Returns the t of the last clean point before the rise, or ``HJ_VALID_T_MAX``
    when no condensation rise is present in the mesopore range.
    """
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(p)
    pp = p[order]
    vv = v[order]
    tt = t[order]

    x = np.log(np.clip(pp, 1e-12, None))
    curv = np.gradient(np.gradient(vv, x), x)

    look = (pp > _CONDENSATION_LOOK_P_MIN) & (pp <= _CONDENSATION_LOOK_P_MAX)
    if int(look.sum()) < 3:
        return float(HJ_VALID_T_MAX)
    peak = float(np.max(curv[look]))
    if not np.isfinite(peak) or peak <= 0.0:
        return float(HJ_VALID_T_MAX)

    thresh = _CONDENSATION_CURVATURE_FRACTION * peak
    rising = look & (curv >= thresh)
    idx = np.where(rising)[0]
    if len(idx) == 0:
        return float(HJ_VALID_T_MAX)
    i0 = int(idx[0])
    if i0 <= 0:
        return float(HJ_VALID_T_MAX)
    return float(min(tt[i0 - 1], HJ_VALID_T_MAX))


# ══════════════════════════════════════════════════════════════
# TWO-SEGMENT FIT (module-level, pure)
# ══════════════════════════════════════════════════════════════

def fit_two_segment(t, v, t_min: float, t_max: float,
                    split_t_min: float = HJ_VALID_T_MIN) -> dict:
    """
    Fit the two-segment t-plot construction of Lippens & de Boer.

    Line 1 (through the origin, low-t) gives the total surface area; line 2
    (free intercept, high-t) gives the external surface area (slope) and the
    micropore volume (intercept x Gurvich factor).

    Bend-point detection scans every split that leaves at least
    ``MIN_SEGMENT_POINTS`` in each segment and keeps line 2 at/above
    ``split_t_min``; for each, line 1 is fit through the origin
    (``slope = sum(t*v)/sum(t**2)``) and line 2 by ordinary least squares.

    The three physical-validity constraints of a t-plot decomposition
        * ``slope_1 > slope_2``    (micropore filling is the steeper region)
        * ``intercept_2 >= 0``
        * ``S_external <= S_total``
    are enforced *during* the scan: a split violating any of them is excluded,
    so a physically impossible decomposition (in particular ``S_external >
    S_total``) is never returned. ``2t >= 0.7 nm`` is a *reliability* statement
    (B-AD-009), not a validity gate, so it is flagged rather than used to reject
    a genuine bend. Among the valid splits the one with the lowest total SSE is
    selected. If no split satisfies the three validity constraints, the window
    admits no two-segment decomposition and ``None`` is returned (the caller
    should fall back to a single origin line — see :func:`fit_tplot_model`).

    Parameters
    ----------
    t, v : array-like — statistical thickness (Å) and adsorbed amount
        (cm³(STP)/g), same length.
    t_min, t_max : float — window bounds (Å); line 1 reaches down to ``t_min``,
        line 2 up to ``t_max``.
    split_t_min : float — the split (bend) must be at or above this thickness
        (line 2 never dips below it).

    Returns
    -------
    dict with the derived quantities, per-segment counts and constraint flags,
    or ``None`` if no physically valid two-segment decomposition exists in the
    window. Raises ValueError if the window holds fewer than
    ``2 * MIN_SEGMENT_POINTS`` points.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or t.shape != v.shape:
        raise ValueError("t and v must be 1-D arrays of equal length.")

    mask = (t >= t_min) & (t <= t_max)
    t = t[mask]
    v = v[mask]
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    n = len(t)

    if n < 2 * MIN_SEGMENT_POINTS:
        raise ValueError(
            f"two-segment t-plot fit needs at least {2 * MIN_SEGMENT_POINTS} "
            f"points in window ({t_min:.2f}-{t_max:.2f} Å) but only {n} are "
            f"available."
        )

    best = None
    for split in range(MIN_SEGMENT_POINTS, n - MIN_SEGMENT_POINTS + 1):
        if t[split] < split_t_min:
            continue  # line 2 would dip below the Harkins-Jura floor
        t1, v1 = t[:split], v[:split]
        t2, v2 = t[split:], v[split:]

        # Line 1: constrained through the origin.
        slope1 = float(np.dot(t1, v1) / np.dot(t1, t1))
        # Line 2: free intercept.
        reg = linregress(t2, v2)
        slope2 = float(reg.slope)
        intercept2 = float(reg.intercept)

        sse1 = float(np.sum((slope1 * t1 - v1) ** 2))
        sse2 = float(np.sum((slope2 * t2 + intercept2 - v2) ** 2))
        sse = sse1 + sse2

        S_total = slope1 * N2_TPLOT_SLOPE_FACTOR
        S_external = slope2 * N2_TPLOT_SLOPE_FACTOR
        v_micro_raw = intercept2 * N2_STP_TO_LIQUID

        denom = slope1 - slope2
        if denom > 1e-12:
            t_bend = intercept2 / denom
            two_t_nm = 2.0 * t_bend / 10.0
        else:
            t_bend = np.nan
            two_t_nm = np.nan

        ok_slope = slope1 > slope2
        ok_intercept = intercept2 >= 0
        ok_s = S_external <= S_total
        bend_defined = bool(denom > 1e-12)
        ok_bend = bool(bend_defined and two_t_nm >= BEND_2T_MIN_NM)

        # FIX B — the physically *impossible* constraints gate the fit: a split
        # with slope1 <= slope2, a negative intercept, S_external > S_total, or
        # parallel segments (no finite bend) is not a valid decomposition and is
        # excluded outright (never reported). 2t >= 0.7 nm is a *reliability*
        # statement (B-AD-009), not a validity gate, so it is carried through as
        # a flag rather than used to reject a genuine bend.
        if not (ok_slope and ok_intercept and ok_s and bend_defined):
            continue

        cand = {
            "split": split,
            "slope1": slope1, "slope2": slope2, "intercept2": intercept2,
            "sse": sse, "sse1": sse1, "sse2": sse2,
            "r2_1": 1.0 - sse1 / max(float(np.dot(v1, v1)), 1e-30),
            "r2_2": float(reg.rvalue ** 2),
            "S_total": S_total, "S_external": S_external,
            "v_micro_raw": v_micro_raw,
            "t_bend": t_bend, "two_t_nm": two_t_nm,
            "ok_bend": ok_bend,
        }
        if best is None or cand["sse"] < best["sse"]:
            best = cand

    if best is None:
        # No split satisfies the physical-validity constraints — the window
        # admits no two-segment decomposition (e.g. a straight line through the
        # origin or a convex t-plot). The caller should fall back to a single
        # line.
        return None

    slope1 = best["slope1"]
    slope2 = best["slope2"]
    intercept2 = best["intercept2"]
    split = best["split"]

    S_total = best["S_total"]
    S_external = best["S_external"]
    s_micro_raw = S_total - S_external
    s_micro = max(s_micro_raw, 0.0)

    v_micro_raw = best["v_micro_raw"]
    v_micro = max(v_micro_raw, 0.0)

    # The three physical-validity constraints were enforced during the scan;
    # ok_bend (2t >= 0.7 nm) is a *reliability* statement, not a validity gate,
    # so it is reported below rather than used to reject the decomposition.
    ok_slope = ok_intercept = ok_s = True
    ok_bend = best["ok_bend"]

    warnings_fired = []
    if not ok_bend:
        warnings_fired.append("2t < 0.7 nm")
        two_t_str = "undefined" if not np.isfinite(best["two_t_nm"]) else \
            f"{best['two_t_nm']:.3f} nm"
        warnings.warn(
            f"t-plot bend point gives 2t = {two_t_str}; B-AD-009 states the "
            "mean pore diameter is unreliable below 0.7 nm.",
            UserWarning, stacklevel=2,
        )

    # Low-confidence reasons: minimum-size segment, or an unreliable mean pore
    # diameter (the validity constraints are guaranteed by the scan).
    low_confidence_reasons = []
    if split < 4:
        low_confidence_reasons.append("fewer than 4 points in line-1 segment")
    if (n - split) < 4:
        low_confidence_reasons.append("fewer than 4 points in line-2 segment")
    if not ok_bend:
        low_confidence_reasons.append("mean pore diameter unreliable")
    low_confidence = bool(low_confidence_reasons)
    low_confidence_reason = "; ".join(low_confidence_reasons)

    return {
        "slope_1"        : round(slope1, 6),
        "slope_2"        : round(slope2, 6),
        "intercept_2"    : round(intercept2, 6),
        "S_total_m2g"    : round(S_total, 2),
        "S_external_m2g" : round(S_external, 2),
        "s_micro_raw"    : round(s_micro_raw, 2),
        "S_micro_m2g"    : round(s_micro, 2),
        "V_micro_raw_cm3g": round(v_micro_raw, 6),
        "V_micro_cm3g"   : round(v_micro, 5),
        "t_bend_A"       : round(best["t_bend"], 3) if np.isfinite(best["t_bend"]) else None,
        "2t_nm"          : round(best["two_t_nm"], 3) if np.isfinite(best["two_t_nm"]) else None,
        "R2_1"           : round(best["r2_1"], 5),
        "R2_2"           : round(best["r2_2"], 5),
        "n_points_1"     : split,
        "n_points_2"     : n - split,
        "n_points"       : n,
        "split_index"    : split,
        "sse"            : best["sse"],
        "t_range"        : (round(float(t_min), 2), round(float(t_max), 2)),
        "flags"          : {
            "slope_order_ok": ok_slope,
            "intercept_ok"  : ok_intercept,
            "s_order_ok"    : ok_s,
            "bend_ok"       : ok_bend,
        },
        "warnings"       : warnings_fired,
        "clamped"        : False,
        "low_confidence" : low_confidence,
        "low_confidence_reason": low_confidence_reason,
    }


def _aicc(n: int, sse: float, k: int) -> float:
    """Small-sample-corrected Akaike information criterion.

    ``AICc = n·ln(SSE/n) + 2k + 2k(k+1)/(n−k−1)``; lower is better. A perfect
    fit (SSE = 0) returns ``-inf``; an undefined value (too few points, or
    non-finite/negative SSE) returns ``+inf``.
    """
    if n - k - 1 <= 0:
        return float("inf")
    if not np.isfinite(sse) or sse < 0:
        return float("inf")
    if sse == 0.0:
        return float("-inf")
    return float(
        n * np.log(sse / n)
        + 2.0 * k
        + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    )


def fit_tplot_model(t, v, t_min: float, t_max: float,
                    split_t_min: float = HJ_VALID_T_MIN) -> dict:
    """Choose the best t-plot model: a single origin line vs a two-segment bend.

    FIX A — a bend is only reported when it is statistically justified. A single
    origin-constrained line (1 parameter) is always fitted first; the best
    *physically valid* two-segment fit (3 parameters: slope1, slope2, intercept2)
    is then compared with the small-sample-corrected AICc. Raw SSE is not used,
    because two segments always fit at least as well as one. AICc is chosen over
    an F-test because it penalises the extra parameters without an arbitrary
    p-value threshold, and its small-sample correction suits the short windows
    the t-plot uses.

    If the single line wins (or no valid two-segment fit exists), the result
    reports ``S_total == S_ext == slope*15.47``, ``S_micro = 0``,
    ``V_micro = 0`` and ``2t`` not applicable, with ``bend_detected`` False.

    Returns
    -------
    dict with ``model`` ("single_line" or "two_segment"), ``bend_detected`` and
    the full public result keys consumed by :meth:`TPlotAnalyser.fit_tplot`.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    mask = (t >= t_min) & (t <= t_max)
    t = t[mask]
    v = v[mask]
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    n = len(t)

    if n < 2 * MIN_SEGMENT_POINTS:
        raise ValueError(
            f"t-plot fit needs at least {2 * MIN_SEGMENT_POINTS} points in "
            f"window ({t_min:.2f}-{t_max:.2f} Å) but only {n} are available."
        )

    # Single line through the origin.
    slope0 = float(np.dot(t, v) / np.dot(t, t))
    sse0 = float(np.sum((slope0 * t - v) ** 2))
    r2_0 = 1.0 - sse0 / max(float(np.dot(v, v)), 1e-30)
    aicc0 = _aicc(n, sse0, 1)

    two = fit_two_segment(t, v, t_min, t_max, split_t_min)
    aicc_two = _aicc(n, two["sse"], 3) if two is not None else float("inf")

    if two is not None and aicc_two < aicc0:
        result = dict(two)
        result["model"] = "two_segment"
        result["bend_detected"] = True
        result["no_bend_reason"] = ""
        result["aicc"] = round(aicc_two, 3)
        result["aicc_single_line"] = round(aicc0, 3)
        result["single_slope"] = round(slope0, 6)
        return result

    S = slope0 * N2_TPLOT_SLOPE_FACTOR
    return {
        "model": "single_line",
        "bend_detected": False,
        "no_bend_reason": ("no micropore bend detected — a single straight line "
                           "through the origin fits best"),
        "aicc": round(aicc0, 3),
        "aicc_two_segment": None if two is None else round(aicc_two, 3),
        "single_slope": round(slope0, 6),
        "slope_1": round(slope0, 6),
        "slope_2": round(slope0, 6),
        "intercept_2": 0.0,
        "S_total_m2g": round(S, 2),
        "S_external_m2g": round(S, 2),
        "s_micro_raw": 0.0,
        "S_micro_m2g": 0.0,
        "V_micro_raw_cm3g": 0.0,
        "V_micro_cm3g": 0.0,
        "t_bend_A": None,
        "2t_nm": None,
        "R2_1": round(r2_0, 5),
        "R2_2": None,
        "n_points_1": n,
        "n_points_2": 0,
        "n_points": n,
        "split_index": None,
        "sse": sse0,
        "t_range": (round(float(t_min), 2), round(float(t_max), 2)),
        "flags": {},
        "warnings": [],
        "clamped": False,
        "low_confidence": False,
        "low_confidence_reason": "",
    }


def fit_line2_only(t, v, t_min: float, t_max: float) -> dict:
    """Fit line 2 alone (external surface area) over ``[t_min, t_max]``.

    Used when the two-segment fit is not possible (insufficient micropore-region
    points): the external surface area is still measurable from the multilayer
    region, so it is reported on its own, clearly labelled, without any
    micropore quantity.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    mask = (t >= t_min) & (t <= t_max)
    t2 = t[mask]
    v2 = v[mask]
    order = np.argsort(t2)
    t2, v2 = t2[order], v2[order]
    n = len(t2)
    if n < MIN_SEGMENT_POINTS:
        raise ValueError(
            f"line-2 fit needs at least {MIN_SEGMENT_POINTS} points in window "
            f"({t_min:.2f}-{t_max:.2f} Å) but only {n} are available."
        )
    reg = linregress(t2, v2)
    slope2 = float(reg.slope)
    intercept2 = float(reg.intercept)
    return {
        "slope_2"        : round(slope2, 6),
        "intercept_2"    : round(intercept2, 6),
        "S_external_m2g" : round(slope2 * N2_TPLOT_SLOPE_FACTOR, 2),
        "R2_2"           : round(float(reg.rvalue ** 2), 5),
        "n_points_2"     : n,
        "t_range"        : (round(float(t_min), 2), round(float(t_max), 2)),
    }


# ══════════════════════════════════════════════════════════════
# T-PLOT ANALYSER CLASS
# ══════════════════════════════════════════════════════════════

class TPlotAnalyser:
    """
    T-Plot analysis from N₂ physisorption data (two-segment construction).

    Parameters
    ----------
    pressure            : array-like — relative pressure (P/P0)
    volume_adsorbed     : array-like — volume adsorbed (cm³/g STP)
    s_bet               : float      — BET surface area (m²/g)
    total_pore_volume   : float      — total pore volume at P/P0 ≈ 0.99 (cm³/g)
    reference_curve     : str        — "harkins-jura" (default) or "halsey";
                                       see REFERENCE_CURVES.
    c_constant          : float|None — BET C constant, used to warn when the
                                       monolayer capacity is questionable
                                       (Thommes et al. 2015 §5.1.1, §6.2).
    """

    def __init__(self, pressure, volume_adsorbed, s_bet: float,
                 total_pore_volume: float, reference_curve: str = "harkins-jura",
                 c_constant: float = None):
        self.p    = np.array(pressure,        dtype=float)
        self.v    = np.array(volume_adsorbed, dtype=float)
        self.sbet = s_bet
        self.vtot = total_pore_volume
        self.c    = c_constant
        self.reference_curve = reference_curve
        if reference_curve not in REFERENCE_CURVES:
            raise ValueError(
                f"Unknown reference_curve {reference_curve!r}; choose from "
                f"{sorted(REFERENCE_CURVES)}."
            )
        self.t = REFERENCE_CURVES[reference_curve](self.p)

        if reference_curve == "harkins-jura" and c_constant is not None \
                and c_constant < 50:
            warnings.warn(
                "Harkins-Jura derives from oxidic surfaces and the BET C "
                "constant is below ~50, where IUPAC questions the monolayer "
                "capacity (Thommes et al. 2015 §5.1.1, §6.2). The t-plot is "
                "resting on a doubtful n_m for this sample; prefer a measured "
                "reference t-curve or the αs-plot.",
                UserWarning, stacklevel=2,
            )

    # ──────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────

    def fit_tplot(self, t_min: float = LINE1_T_MIN,
                  t_max: float = HJ_VALID_T_MAX) -> dict:
        """
        Fit the t-plot, choosing between a single origin line and a two-segment
        bend (Lippens & de Boer construction).

        Line 1 (origin) -> total surface area; line 2 (free intercept) ->
        external surface area (slope) and micropore volume (intercept). Line 1
        is allowed down to the per-curve ``line1_t_min`` so it can reach the
        micropore-filling region; line 2 is confined to
        ``[HJ_VALID_T_MIN, HJ_VALID_T_MAX]``, and the whole window is
        additionally capped at the data-derived capillary-condensation onset
        (:func:`condensation_onset_t`).

        A two-layer data-sufficiency gate runs first: micropore analysis is
        possible only when the adsorption data has at least ``MIN_LINE1_POINTS``
        points below p/p0 = ``LINE1_P_MAX`` AND at least
        ``MIN_LINE1_PRIMARY_POINTS`` point below the primary-filling bound
        ``LINE1_P_PRIMARY``. If either fails, the micropore quantities (V_micro,
        S_micro, S_total, t_bend, 2t) cannot be determined and are returned as
        ``None`` with ``micropore_analysis_possible`` False; only the external
        surface area (line 2) is reported in that case.

        When the gate passes, :func:`fit_tplot_model` decides between a single
        line (no bend) and a two-segment bend using AICc; a physically invalid
        two-segment fit is never returned.

        Returns the model dict plus ``reference_curve``, ``S_BET_m2g``, the
        sufficiency-gate keys and the ``S_ext_m2g`` / ``R2_tplot`` /
        ``intercept`` / ``slope`` compatibility keys.
        """
        t_min = max(float(t_min), line1_t_min(self.reference_curve))
        t_max = min(float(t_max), HJ_VALID_T_MAX)
        # FIX C — cap the fit window at the data-derived capillary-condensation
        # onset (a sample property), separately from the Harkins-Jura validity
        # ceiling applied above.
        t_max = min(t_max, condensation_onset_t(self.p, self.v, self.t))

        n_below = int((self.p < LINE1_P_MAX).sum())
        n_primary = int((self.p < LINE1_P_PRIMARY).sum())
        enough_below = n_below >= MIN_LINE1_POINTS
        enough_primary = n_primary >= MIN_LINE1_PRIMARY_POINTS
        gate_passed = enough_below and enough_primary

        if gate_passed:
            fit = fit_tplot_model(self.t, self.v, t_min, t_max,
                                  split_t_min=HJ_VALID_T_MIN)
            result = dict(fit)
        else:
            line2 = fit_line2_only(self.t, self.v, HJ_VALID_T_MIN, t_max)
            result = {
                "model": "line2_only",
                "bend_detected": False,
                "no_bend_reason": "",
                "slope_1": None,
                "slope_2": line2["slope_2"],
                "intercept_2": line2["intercept_2"],
                "S_total_m2g": None,
                "S_external_m2g": line2["S_external_m2g"],
                "s_micro_raw": None,
                "S_micro_m2g": None,
                "V_micro_raw_cm3g": None,
                "V_micro_cm3g": None,
                "t_bend_A": None,
                "2t_nm": None,
                "R2_1": None,
                "R2_2": line2["R2_2"],
                "n_points_1": 0,
                "n_points_2": line2["n_points_2"],
                "n_points": line2["n_points_2"],
                "split_index": None,
                "t_range": line2["t_range"],
                "flags": {},
                "warnings": [],
                "clamped": False,
                "low_confidence": False,
                "low_confidence_reason": "",
            }

        result["reference_curve"] = self.reference_curve
        result["S_BET_m2g"] = round(self.sbet, 2)
        result["micropore_analysis_possible"] = gate_passed
        result["n_points_below_pp008"] = n_below
        result["n_points_below_pp0015"] = n_primary

        if gate_passed:
            result["micropore_analysis_reason"] = ""
        else:
            failed = []
            if not enough_below:
                failed.append(
                    f"only {n_below} point(s) below p/p0 = 0.08 "
                    f"(need at least {MIN_LINE1_POINTS})"
                )
            if not enough_primary:
                failed.append(
                    f"only {n_primary} point(s) below p/p0 = 0.015 "
                    f"(need at least {MIN_LINE1_PRIMARY_POINTS})"
                )
            result["micropore_analysis_reason"] = (
                "micropore volume and surface area cannot be determined from "
                "this measurement (" + "; ".join(failed) + "). A t-plot "
                "micropore analysis needs at least one adsorption point below "
                "p/p0 ~ 0.015, ideally several lower still; check your "
                "instrument's low-pressure specification and measurement-range "
                "setting (Thommes et al. 2015 §6.1; Cychosz & Thommes 2018 §3). "
                "§6.1 also recommends argon at 87 K over nitrogen at 77 K where "
                "surface functional groups interact with the N2 quadrupole."
            )
        # Compatibility keys (Phase 1A naming) — S_ext is now the *external*
        # area from line 2, not the old single-line slope.
        result["S_ext_m2g"] = result["S_external_m2g"]
        result["R2_tplot"] = result["R2_2"]
        result["intercept"] = result["intercept_2"]
        result["slope"] = result["slope_2"]
        return result

    # ──────────────────────────────────────────────────────────
    # PORE DISTRIBUTION
    # ──────────────────────────────────────────────────────────

    def pore_distribution(self, v_micro: float) -> dict:
        """
        Calculate pore volume fractions.

        V_meso+macro = V_total - V_micro.  V_macro is not separated because it
        needs Hg porosimetry; the "meso" bucket below is really meso + macro.

        When the total pore volume is unavailable (``self.vtot`` is None — e.g.
        a plain isotherm that does not reach p/p0 ≈ 0.99), the meso/macro and
        total volumes are declined (None) rather than computed from a fake
        total.

        Returns
        -------
        dict with volumes (cm³/g) and % for micropore and meso+macro.
        """
        if self.vtot is None:
            return {
                "V_micro_cm3g": round(v_micro, 5),
                "V_meso_cm3g": None,
                "V_total_cm3g": None,
                "Micropore_%": None,
                "Meso_Macro_%": None,
                "V_total_reason": (
                    "total pore volume unavailable (isotherm does not reach "
                    "p/p0 ≈ 0.99, so the Gurvich total could not be derived)"
                ),
            }
        v_meso = max(self.vtot - v_micro, 0.0)
        total = v_micro + v_meso
        if total <= 0:
            return {"error": "Total pore volume is zero or negative."}
        return {
            "V_micro_cm3g"  : round(v_micro, 5),
            "V_meso_cm3g"   : round(v_meso,  5),
            "V_total_cm3g"  : round(total,  5),
            "Micropore_%"   : round(100 * v_micro / total, 1),
            "Meso_Macro_%"  : round(100 * v_meso  / total, 1),
        }

    # ──────────────────────────────────────────────────────────
    # FULL REPORT
    # ──────────────────────────────────────────────────────────

    def full_tplot_report(self, t_min: float = LINE1_T_MIN,
                          t_max: float = HJ_VALID_T_MAX) -> dict:
        """Run the two-segment fit and pore distribution — all together."""
        fit = self.fit_tplot(t_min, t_max)
        if fit["micropore_analysis_possible"]:
            dist = self.pore_distribution(fit["V_micro_cm3g"])
            return {**fit, **dist}
        # Micropore volume unknown -> pore distribution is not computable.
        no_dist = {
            "V_meso_cm3g": None,
            "V_total_cm3g": None,
            "Micropore_%": None,
            "Meso_Macro_%": None,
        }
        return {**fit, **no_dist}

    def print_report(self, sample_name: str = "Sample",
                     t_min: float = LINE1_T_MIN,
                     t_max: float = HJ_VALID_T_MAX):
        res = self.full_tplot_report(t_min, t_max)
        sep = "=" * 58
        print(f"\n{sep}")
        print(f"  T-Plot Report ({res['reference_curve']}) — {sample_name}")
        print(sep)
        if res["micropore_analysis_possible"]:
            if res.get("model") == "single_line":
                print(f"  Model          : single line through the origin — no micropore bend")
            else:
                print(f"  Model          : two-segment (micropore bend detected)")
            print(f"  Fit window     : {res['t_range'][0]}–{res['t_range'][1]} Å  "
                  f"({res['n_points_1']} + {res['n_points_2']} pts)")
            print(f"  R² (line 1/2)  : {res['R2_1']} / {res['R2_2']}")
        else:
            print(f"  Fit window     : {res['t_range'][0]}–{res['t_range'][1]} Å  "
                  f"({res['n_points_2']} pts, line 2 only)")
            print(f"  ⚠ Micropore analysis not possible: {res['micropore_analysis_reason']}")
        print(f"")
        print(f"  Surface Area")
        print(f"    S_BET        : {res['S_BET_m2g']:.2f}  m² g⁻¹  (monolayer)")
        if res["S_total_m2g"] is not None:
            print(f"    S_total      : {res['S_total_m2g']:.2f}  m² g⁻¹  (line 1)")
            print(f"    S_external   : {res['S_external_m2g']:.2f}  m² g⁻¹  (line 2)")
            print(f"    S_micro      : {res['S_micro_m2g']:.2f}  m² g⁻¹  (total − external)")
        else:
            print(f"    S_external   : {res['S_external_m2g']:.2f}  m² g⁻¹  (line 2 only)")
            print(f"    S_total      : not reported (micropore region undersampled)")
            print(f"    S_micro      : not reported (micropore region undersampled)")
        print(f"")
        if res["V_micro_cm3g"] is not None:
            print(f"  Pore Volumes")
            if res["V_meso_cm3g"] is None:
                print(f"    V_micro      : {res['V_micro_cm3g']:.5f}  cm³ g⁻¹")
                print(f"    V_meso+macro : not reported ({res.get('V_total_reason', 'total pore volume unavailable')})")
            else:
                print(f"    V_total      : {res['V_total_cm3g']:.5f}  cm³ g⁻¹")
                print(f"    V_micro      : {res['V_micro_cm3g']:.5f}  cm³ g⁻¹   ({res['Micropore_%']}%)")
                print(f"    V_meso+macro : {res['V_meso_cm3g']:.5f}  cm³ g⁻¹   ({res['Meso_Macro_%']}%)")
                print(f"      ↳ V_macro needs Hg porosimetry")
        else:
            print(f"  Pore Volumes   : not reported (micropore volume undetermined)")
        print(f"")
        if res["t_bend_A"] is not None:
            print(f"  Bend point")
            print(f"    t_bend       : {res['t_bend_A']:.3f} Å")
            print(f"    2t (diameter): {res['2t_nm']:.3f} nm")
        elif res.get("model") == "single_line" and res["micropore_analysis_possible"]:
            print(f"  Bend point     : none (no micropore bend detected)")
        else:
            print(f"  Bend point     : none (not fitted)")
        if res["warnings"]:
            print(f"")
            print(f"  Warnings       : {', '.join(res['warnings'])}")
        if res["low_confidence"]:
            print(f"  Low confidence : {res['low_confidence_reason']}")
        print(f"{sep}\n")

    # ──────────────────────────────────────────────────────────
    # PLOT
    # ──────────────────────────────────────────────────────────

    def plot_tplot(self, save_path: str = "tplot.png", sample_name: str = "Sample",
                   t_min: float = LINE1_T_MIN,
                   t_max: float = HJ_VALID_T_MAX) -> str:
        """
        2-panel T-Plot figure:
          [A] t-plot with the fitted lines + bend point (line 1/bend only when
              micropore analysis was possible)
          [B] Pore type distribution bar chart (omitted when V_micro unknown)
        """
        res  = self.full_tplot_report(t_min, t_max)
        t_lo, t_hi = res["t_range"]
        has_micropore = res["micropore_analysis_possible"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # ── [A] t-plot ─────────────────────────────────────────
        ax = axes[0]
        ax.scatter(self.t, self.v, color=C_MICRO, s=35, zorder=5,
                   label="Experimental data")

        order = np.argsort(self.t)
        t_s, v_s = self.t[order], self.v[order]

        single_line = has_micropore and res.get("model") == "single_line"

        if single_line:
            # One straight line through the origin — no bend to mark.
            ax.scatter(t_s, v_s, color=C_TOTAL, s=55, zorder=6, marker="o",
                       label="Line 1 (total)")
            t_line = np.linspace(0, self.t.max() * 1.02, 200)
            ax.plot(t_line, res["slope_1"] * t_line, "-", color=C_TOTAL, lw=1.8,
                    label=f"Surface area  S={res['S_total_m2g']:.1f} m²/g")
        elif has_micropore:
            # Highlight the two fitted segments
            split = res["n_points_1"]
            ax.scatter(t_s[:split], v_s[:split], color=C_TOTAL, s=55, zorder=6,
                       marker="o", label="Line 1 (total)")
            ax.scatter(t_s[split:], v_s[split:], color=C_EXT, s=55, zorder=6,
                       marker="s", label="Line 2 (external)")

            # Line 1: through the origin
            t_line1 = np.linspace(0, res["t_bend_A"] or self.t.min(), 100)
            ax.plot(t_line1, res["slope_1"] * t_line1, "-", color=C_TOTAL, lw=1.8,
                    label=f"Total surface area  S={res['S_total_m2g']:.1f} m²/g")
            # Line 2
            t_line2 = np.linspace((res["t_bend_A"] or self.t.min()),
                                  self.t.max() * 1.02, 200)
            ax.plot(t_line2, res["slope_2"] * t_line2 + res["intercept_2"], "-",
                    color=C_EXT, lw=1.8,
                    label=f"External surface area  S={res['S_external_m2g']:.1f} m²/g")
        else:
            t_line2 = np.linspace(t_lo, self.t.max() * 1.02, 200)
            ax.plot(t_line2, res["slope_2"] * t_line2 + res["intercept_2"], "-",
                    color=C_EXT, lw=1.8,
                    label=f"External surface area  S={res['S_external_m2g']:.1f} m²/g")

        # Bend point (only when line 1 was fitted)
        if has_micropore and res["t_bend_A"] is not None and np.isfinite(res["t_bend_A"]):
            v_bend = res["slope_1"] * res["t_bend_A"]
            ax.plot(res["t_bend_A"], v_bend, "o", color="k", ms=7, zorder=7)
            ax.annotate(f"2t = {res['2t_nm']:.2f} nm",
                        (res["t_bend_A"], v_bend),
                        textcoords="offset points", xytext=(8, -12),
                        fontsize=8.5)

        ax.set_xlabel("Statistical film thickness  t (Å)", fontsize=11)
        ax.set_ylabel("Volume adsorbed  (cm³ g⁻¹ STP)",   fontsize=11)
        ax.set_title(f"T-Plot ({res['reference_curve']})", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.grid(False)
        ax.set_xlim(0, self.t.max() * 1.05)

        # Annotation box
        if single_line:
            ann = (f"$S$ = {res['S_total_m2g']:.1f} m² g⁻¹\n"
                   "no micropore bend\n($S_{{micro}}$ = 0)")
        elif has_micropore:
            ann = (f"$S_{{total}}$ = {res['S_total_m2g']:.1f} m² g⁻¹\n"
                   f"$S_{{ext}}$ = {res['S_external_m2g']:.1f} m² g⁻¹\n"
                   f"$S_{{micro}}$ = {res['S_micro_m2g']:.1f} m² g⁻¹\n"
                   f"$V_{{micro}}$ = {res['V_micro_cm3g']:.4f} cm³ g⁻¹")
        else:
            ann = (f"$S_{{ext}}$ = {res['S_external_m2g']:.1f} m² g⁻¹\n"
                   "micropore analysis\nnot possible (undersampled)")
        ax.text(0.97, 0.05, ann, transform=ax.transAxes,
                va="bottom", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.7))

        # ── [B] Pore distribution bar ──────────────────────────
        ax2 = axes[1]
        if has_micropore:
            labels = ["Micropore", "Meso + Macro"]
            values = [res["Micropore_%"], res["Meso_Macro_%"]]
            colors = [C_MICRO, C_EXT]
            bars = ax2.bar(labels, values, color=colors, width=0.5,
                           edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                         fontweight="bold")
            ax2.set_ylabel("Pore Volume Fraction (%)", fontsize=11)
            ax2.set_title("Pore Type Distribution", fontsize=11, fontweight="bold")
            ax2.set_ylim(0, max(values) * 1.18)
            ax2.grid(axis="y", alpha=0.3)
            ax2.text(0.5, -0.14, "V_macro needs Hg porosimetry (folded into Meso+Macro)",
                     transform=ax2.transAxes, ha="center", fontsize=7.5, color="0.4")
        else:
            ax2.text(0.5, 0.5, "Micropore volume undetermined\n(insufficient points "
                     "below p/p0 = 0.08)", ha="center", va="center", fontsize=9,
                     color="0.4")
            ax2.set_xticks([])
            ax2.set_yticks([])

        fig.suptitle(f"T-Plot Analysis — {sample_name}",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path


# ══════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════

def _configure_console():
    """Emit UTF-8 so the report never crashes a cp1252 console (see bet_analysis)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main():
    _configure_console()
    parser = argparse.ArgumentParser(
        description="T-Plot Analysis (two-segment) from raw P/P0 + V data")
    parser.add_argument("--s-bet",  type=float, required=True,
                        help="BET surface area (m²/g)")
    parser.add_argument("--vtot",   type=float, required=True,
                        help="Total pore volume at P/P0=0.99 (cm³/g)")
    parser.add_argument("--sample", default="Sample",
                        help="Sample name for plot title")
    parser.add_argument("--reference-curve", default="harkins-jura",
                        choices=sorted(REFERENCE_CURVES))
    parser.add_argument("--t-min",  type=float, default=LINE1_T_MIN)
    parser.add_argument("--t-max",  type=float, default=HJ_VALID_T_MAX)
    args = parser.parse_args()

    print("\n  ⚠  Standalone mode: using built-in demo data.")
    print("     For real data, import TPlotAnalyser from bet_analysis workflow.\n")

    # Demo data — typical mesoporous silica
    p = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.50,0.70,0.90,0.99])
    v = np.array([85., 102.,115.,126.,135.,143.,150.,172.,200.,280.,520.])

    tp = TPlotAnalyser(p, v, s_bet=args.s_bet, total_pore_volume=args.vtot,
                       reference_curve=args.reference_curve)
    tp.print_report(sample_name=args.sample, t_min=args.t_min, t_max=args.t_max)
    tp.plot_tplot(save_path=f"{args.sample}_tplot.png", sample_name=args.sample,
                  t_min=args.t_min, t_max=args.t_max)
    print(f"  Plot saved → {args.sample}_tplot.png")


if __name__ == "__main__":
    main()
