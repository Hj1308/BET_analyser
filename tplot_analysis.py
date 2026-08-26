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
# (see harkins_jura_t above). The t-plot fit window must not cross that
# boundary:
#   - below p/p0 = 0.08, micropore filling is still in progress, which is
#     exactly the regime line 1 (total surface area) models. p/p0 = 0.08
#     corresponds to t = 3.52 Å; 3.5 Å is the rounded floor.
#   - above p/p0 = 0.60, the thickness grows rapidly and the isotherm enters
#     the capillary-condensation region, where a linear t-plot is undefined.
#     6.5 Å is the conservative ceiling (p/p0 ≈ 0.5, safely inside the range).
HJ_VALID_T_MIN = 3.5
HJ_VALID_T_MAX = 6.5

# Each segment of the two-segment fit needs this many points to be meaningful.
MIN_SEGMENT_POINTS = 3

# B-AD-009: the mean pore diameter 2t is unreliable below 0.7 nm.
BEND_2T_MIN_NM = 0.7


# ══════════════════════════════════════════════════════════════
# TWO-SEGMENT FIT (module-level, pure)
# ══════════════════════════════════════════════════════════════

def fit_two_segment(t, v, t_min: float, t_max: float) -> dict:
    """
    Fit the two-segment t-plot construction of Lippens & de Boer.

    Line 1 (through the origin, low-t) gives the total surface area; line 2
    (free intercept, high-t) gives the external surface area (slope) and the
    micropore volume (intercept x Gurvich factor).

    Bend-point detection scans every split that leaves at least
    ``MIN_SEGMENT_POINTS`` in each segment; for each, line 1 is fit through the
    origin (``slope = sum(t*v)/sum(t**2)``) and line 2 by ordinary least
    squares. The split is chosen to minimise the total sum of squared residuals
    (SSE) of the two lines — the maximum-likelihood choice. The physical
    constraints below are then checked on that best-fit split and reported as
    warnings/flags, *not* used to select the split: in particular ``2t >= 0.7 nm``
    is a reliability statement about the result, and using it (or any constraint
    count) to relocate the bend would hide a genuinely small mean pore diameter
    — exactly the silent tuning this review exists to prevent.

    Both segments must lie inside ``[t_min, t_max]``; the caller is expected to
    keep that inside ``[HJ_VALID_T_MIN, HJ_VALID_T_MAX]``.

    Physical constraints each raise a ``UserWarning`` and set a flag in the
    returned dict (never a silent clamp):
      * ``slope_1 > slope_2``        (micropore filling is the steeper region)
      * ``intercept_2 >= 0``
      * ``S_external <= S_total``
      * ``2t >= 0.7 nm``             (mean pore diameter reliable)

    Parameters
    ----------
    t, v : array-like — statistical thickness (Å) and adsorbed amount
        (cm³(STP)/g), same length.
    t_min, t_max : float — window bounds (Å); only points within are used.

    Returns
    -------
    dict with the derived quantities, per-segment counts, constraint flags and
    any emitted warnings. Raises ValueError if fewer than
    ``2 * MIN_SEGMENT_POINTS`` points fall in the window.
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
        if abs(denom) > 1e-12:
            t_bend = intercept2 / denom
        else:
            t_bend = np.nan
        two_t_nm = 2.0 * t_bend / 10.0 if np.isfinite(t_bend) else np.nan

        ok_slope = slope1 > slope2
        ok_intercept = intercept2 >= 0
        ok_s = S_external <= S_total
        ok_bend = bool(np.isfinite(two_t_nm) and two_t_nm >= BEND_2T_MIN_NM)

        n_violations = (int(not ok_slope) + int(not ok_intercept)
                        + int(not ok_s) + int(not ok_bend))

        cand = {
            "split": split,
            "slope1": slope1, "slope2": slope2, "intercept2": intercept2,
            "sse": sse, "sse1": sse1, "sse2": sse2,
            "r2_1": 1.0 - sse1 / max(float(np.dot(v1, v1)), 1e-30),
            "r2_2": float(reg.rvalue ** 2),
            "S_total": S_total, "S_external": S_external,
            "v_micro_raw": v_micro_raw,
            "t_bend": t_bend, "two_t_nm": two_t_nm,
            "ok_slope": ok_slope, "ok_intercept": ok_intercept,
            "ok_s": ok_s, "ok_bend": ok_bend,
            "n_violations": n_violations,
        }
        if best is None or cand["sse"] < best["sse"]:
            best = cand

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

    ok_slope = best["ok_slope"]
    ok_intercept = best["ok_intercept"]
    ok_s = best["ok_s"]
    ok_bend = best["ok_bend"]

    warnings_fired = []
    if not ok_slope:
        warnings_fired.append("slope_1 <= slope_2")
        warnings.warn(
            "t-plot line 1 (total surface area) is not steeper than line 2 "
            "(external); micropore filling should be the steeper region. The "
            "sample may be non-porous or the reference t-curve may not match "
            "its surface chemistry.",
            UserWarning, stacklevel=2,
        )
    if not ok_intercept:
        warnings_fired.append("intercept_2 < 0")
        warnings.warn(
            f"t-plot line 2 intercept is negative ({intercept2:.4f} cm³/g STP); "
            "V_micro was clamped to 0. A negative intercept usually means the "
            "reference t-curve does not match the sample's surface chemistry, "
            "or the split lies inside the micropore-filling region.",
            UserWarning, stacklevel=2,
        )
    if not ok_s:
        warnings_fired.append("S_external > S_total")
        warnings.warn(
            f"t-plot external surface area ({S_external:.2f} m²/g) exceeds the "
            f"total ({S_total:.2f} m²/g); S_micro was clamped to 0. The "
            "reference t-curve likely does not match the sample's surface.",
            UserWarning, stacklevel=2,
        )
    if not ok_bend:
        warnings_fired.append("2t < 0.7 nm")
        two_t_str = "undefined" if not np.isfinite(best["two_t_nm"]) else \
            f"{best['two_t_nm']:.3f} nm"
        warnings.warn(
            f"t-plot bend point gives 2t = {two_t_str}; B-AD-009 states the "
            "mean pore diameter is unreliable below 0.7 nm.",
            UserWarning, stacklevel=2,
        )

    # Low-confidence reasons: minimum-size segment, violated constraints, or a
    # bend point that is not meaningful (line 1 no steeper than line 2).
    low_confidence_reasons = []
    if split < 4:
        low_confidence_reasons.append("fewer than 4 points in line-1 segment")
    if (n - split) < 4:
        low_confidence_reasons.append("fewer than 4 points in line-2 segment")
    if not ok_slope:
        low_confidence_reasons.append("no steeper micropore-filling segment")
    if not ok_intercept:
        low_confidence_reasons.append("negative line-2 intercept")
    if not ok_s:
        low_confidence_reasons.append("external area exceeds total")
    if not ok_bend:
        low_confidence_reasons.append("mean pore diameter unreliable")
    low_confidence = bool(low_confidence_reasons)
    low_confidence_reason = "; ".join(low_confidence_reasons)

    clamped = (v_micro_raw < 0.0) or (s_micro_raw < 0.0)

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
        "t_range"        : (round(float(t_min), 2), round(float(t_max), 2)),
        "flags"          : {
            "slope_order_ok": ok_slope,
            "intercept_ok"  : ok_intercept,
            "s_order_ok"    : ok_s,
            "bend_ok"       : ok_bend,
        },
        "warnings"       : warnings_fired,
        "clamped"        : clamped,
        "low_confidence" : low_confidence,
        "low_confidence_reason": low_confidence_reason,
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

    def fit_tplot(self, t_min: float = HJ_VALID_T_MIN,
                  t_max: float = HJ_VALID_T_MAX) -> dict:
        """
        Fit the two-segment t-plot (Lippens & de Boer construction).

        Line 1 (origin) -> total surface area; line 2 (free intercept) ->
        external surface area (slope) and micropore volume (intercept). Both
        segments are kept inside ``[HJ_VALID_T_MIN, HJ_VALID_T_MAX]``.

        Returns the dict from :func:`fit_two_segment` plus ``reference_curve``,
        ``S_BET_m2g`` and the ``S_ext_m2g`` / ``R2_tplot`` / ``intercept`` /
        ``slope`` compatibility keys.
        """
        t_min = max(float(t_min), HJ_VALID_T_MIN)
        t_max = min(float(t_max), HJ_VALID_T_MAX)

        fit = fit_two_segment(self.t, self.v, t_min, t_max)

        result = dict(fit)
        result["reference_curve"] = self.reference_curve
        result["S_BET_m2g"] = round(self.sbet, 2)
        # Compatibility keys (Phase 1A naming) — S_ext is now the *external*
        # area from line 2, not the old single-line slope.
        result["S_ext_m2g"] = fit["S_external_m2g"]
        result["R2_tplot"] = fit["R2_2"]
        result["intercept"] = fit["intercept_2"]
        result["slope"] = fit["slope_2"]
        return result

    # ──────────────────────────────────────────────────────────
    # PORE DISTRIBUTION
    # ──────────────────────────────────────────────────────────

    def pore_distribution(self, v_micro: float) -> dict:
        """
        Calculate pore volume fractions.

        V_meso+macro = V_total - V_micro.  V_macro is not separated because it
        needs Hg porosimetry; the "meso" bucket below is really meso + macro.

        Returns
        -------
        dict with volumes (cm³/g) and % for micropore and meso+macro.
        """
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

    def full_tplot_report(self, t_min: float = HJ_VALID_T_MIN,
                          t_max: float = HJ_VALID_T_MAX) -> dict:
        """Run the two-segment fit and pore distribution — all together."""
        fit  = self.fit_tplot(t_min, t_max)
        dist = self.pore_distribution(fit["V_micro_cm3g"])
        return {**fit, **dist}

    # ──────────────────────────────────────────────────────────
    # PRINT REPORT
    # ──────────────────────────────────────────────────────────

    def print_report(self, sample_name: str = "Sample",
                     t_min: float = HJ_VALID_T_MIN,
                     t_max: float = HJ_VALID_T_MAX):
        res = self.full_tplot_report(t_min, t_max)
        sep = "=" * 58
        print(f"\n{sep}")
        print(f"  T-Plot Report ({res['reference_curve']}) — {sample_name}")
        print(sep)
        print(f"  Fit window     : {res['t_range'][0]}–{res['t_range'][1]} Å  "
              f"({res['n_points_1']} + {res['n_points_2']} pts)")
        print(f"  R² (line 1/2)  : {res['R2_1']} / {res['R2_2']}")
        print(f"")
        print(f"  Surface Area")
        print(f"    S_BET        : {res['S_BET_m2g']:.2f}  m² g⁻¹  (monolayer)")
        print(f"    S_total      : {res['S_total_m2g']:.2f}  m² g⁻¹  (line 1)")
        print(f"    S_external   : {res['S_external_m2g']:.2f}  m² g⁻¹  (line 2)")
        print(f"    S_micro      : {res['S_micro_m2g']:.2f}  m² g⁻¹  (total − external)")
        print(f"")
        print(f"  Pore Volumes")
        print(f"    V_total      : {res['V_total_cm3g']:.5f}  cm³ g⁻¹")
        print(f"    V_micro      : {res['V_micro_cm3g']:.5f}  cm³ g⁻¹   ({res['Micropore_%']}%)")
        print(f"    V_meso+macro : {res['V_meso_cm3g']:.5f}  cm³ g⁻¹   ({res['Meso_Macro_%']}%)")
        print(f"      ↳ V_macro needs Hg porosimetry")
        print(f"")
        if res["t_bend_A"] is not None:
            print(f"  Bend point")
            print(f"    t_bend       : {res['t_bend_A']:.3f} Å")
            print(f"    2t (diameter): {res['2t_nm']:.3f} nm")
        else:
            print(f"  Bend point     : none (no meaningful split)")
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
                   t_min: float = HJ_VALID_T_MIN,
                   t_max: float = HJ_VALID_T_MAX) -> str:
        """
        2-panel T-Plot figure:
          [A] t-plot with the two fitted lines + bend point
          [B] Pore type distribution bar chart
        """
        res  = self.full_tplot_report(t_min, t_max)
        t_lo, t_hi = res["t_range"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # ── [A] t-plot ─────────────────────────────────────────
        ax = axes[0]
        ax.scatter(self.t, self.v, color=C_MICRO, s=35, zorder=5,
                   label="Experimental data")

        # Highlight the two fitted segments
        order = np.argsort(self.t)
        t_s, v_s = self.t[order], self.v[order]
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

        # Bend point
        if res["t_bend_A"] is not None and np.isfinite(res["t_bend_A"]):
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
        ann = (f"$S_{{total}}$ = {res['S_total_m2g']:.1f} m² g⁻¹\n"
               f"$S_{{ext}}$ = {res['S_external_m2g']:.1f} m² g⁻¹\n"
               f"$S_{{micro}}$ = {res['S_micro_m2g']:.1f} m² g⁻¹\n"
               f"$V_{{micro}}$ = {res['V_micro_cm3g']:.4f} cm³ g⁻¹")
        ax.text(0.97, 0.05, ann, transform=ax.transAxes,
                va="bottom", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.7))

        # ── [B] Pore distribution bar ──────────────────────────
        ax2   = axes[1]
        labels = ["Micropore", "Meso + Macro"]
        values = [res["Micropore_%"], res["Meso_Macro_%"]]
        colors = [C_MICRO, C_EXT]
        bars   = ax2.bar(labels, values, color=colors, width=0.5,
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

        fig.suptitle(f"T-Plot Analysis — {sample_name}",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path


# ══════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
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
    parser.add_argument("--t-min",  type=float, default=HJ_VALID_T_MIN)
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
