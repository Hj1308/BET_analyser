"""
BET Analysis Tool — Publication-Quality Figures
================================================
Reads instrument XLS/XLSX output and computes:
  - Isotherm type classification  (IUPAC Type I–VI, including I(a)/I(b))
  - Hysteresis type classification (IUPAC H1–H4)
  - BET plot with regression verification
  - Rouquerol auto BET range selection (IUPAC 2015 / ISO 9277)
  - BJH differential pore size distribution
  - Cumulative pore volume
  - BET vs BJH surface area comparison

Usage:
    python bet_analysis.py --file C3N4.xls --sample "C3N4" --rouquerol

Author  : Hoda Jafari | github.com/Hj1308
License : MIT
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress
from scipy.interpolate import interp1d
from tabulate import tabulate

from rouquerol import (
    select_bet_range,
    diagnose_instrument_range,
    format_rouquerol_report,
)
from langmuir import fit_langmuir_window, format_langmuir_report

# np.trapz was removed in NumPy 2.0 and renamed np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ══════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS — N₂ at 77 K
# ══════════════════════════════════════════════════════════════

N2_BET_FACTOR         = 4.353   # m²/g per cm³(STP)/g  [σ_N2=0.162 nm², NA, Vmolar]
N2_TPLOT_SLOPE_FACTOR = 15.47   # m²/g per cm³/(g·Å)   [Harkins-Jura conversion]
# Gurvich rule: V_liquid(cm3/g) = V_STP(cm3/g) x N2_STP_TO_LIQUID
#   = (M / V_molar) / rho = (28.013 / 22413.96) / 0.808
# Ref: Gurvich (1915); Microtrac AppNote "The Adsorption Isotherm", eq. 1
N2_STP_TO_LIQUID      = 1.5468e-3   # cm3(liquid N2) per cm3(STP), 77 K
N2_CAVITATION_NM      = 3.4     # forced closure diameter (nm) for N₂ at 77 K


# ══════════════════════════════════════════════════════════════
# IUPAC VALIDITY THRESHOLDS — Thommes et al. (2015)
# ══════════════════════════════════════════════════════════════

# §5.1.1 — BET C constant and Point B.
BET_C_NOT_APPLICABLE      = 2.0    # C < 2  : Type III/V — BET not applicable
BET_C_POINT_B_QUESTIONABLE = 50.0  # C < 50 : Point B not a single point; n_m doubtful
BET_C_KNEE_SHARP          = 80.0   # C >= 80: sharp knee, Point B well defined

# §7.2 / §9 — BJH (Kelvin-equation) underestimates narrow mesopores by 20-30%.
BJH_NARROW_MESOPORE_NM    = 10.0   # peak diameter (nm) below which BJH is unreliable


# ══════════════════════════════════════════════════════════════
# MATPLOTLIB — publication settings
# ══════════════════════════════════════════════════════════════

def setup_plot_style():
    """Apply publication-quality matplotlib settings. Call once before plotting."""
    plt.rcParams.update({
        "font.family"        : "serif",
        "font.serif"         : ["Times New Roman", "DejaVu Serif"],
        "font.size"          : 10,
        "axes.labelsize"     : 11,
        "axes.titlesize"     : 11,
        "xtick.labelsize"    : 9,
        "ytick.labelsize"    : 9,
        "legend.fontsize"    : 9,
        "legend.framealpha"  : 0.9,
        "legend.edgecolor"   : "0.7",
        "figure.dpi"         : 150,
        "savefig.dpi"        : 300,
        "savefig.bbox"       : "tight",
        "lines.linewidth"    : 1.5,
        "axes.linewidth"     : 0.8,
        "xtick.major.width"  : 0.8,
        "ytick.major.width"  : 0.8,
        "xtick.minor.width"  : 0.5,
        "ytick.minor.width"  : 0.5,
        "xtick.major.size"   : 4,
        "ytick.major.size"   : 4,
        "xtick.minor.size"   : 2,
        "ytick.minor.size"   : 2,
        "xtick.direction"    : "in",
        "ytick.direction"    : "in",
        "xtick.top"          : True,
        "ytick.right"        : True,
        "axes.grid"          : False,
    })

# Color palette (colorblind-safe)
C_ADS   = "#2166AC"   # blue
C_DES   = "#D6604D"   # red-orange
C_BET   = "#1A7A4A"   # green
C_BJH   = "#7B3F9E"   # purple
C_CUM   = "#C07028"   # amber
C_SHADE = "#AACCE8"   # light blue fill


# ══════════════════════════════════════════════════════════════
# 1. DATA READING
# ══════════════════════════════════════════════════════════════

def _load_sheets(filepath: str) -> tuple:
    """
    Load all sheets from XLS or XLSX.

    - .xls  : reads via the xlrd API directly (xls_reader module), bypassing
              the pandas >= 2.0 engine guard that rejects xlrd 1.2.x.
    - .xlsx : standard pandas path with openpyxl.

    Returns (sheet_names, raw_dict) where raw_dict maps sheet name to a
    header=None DataFrame.
    """
    if str(filepath).lower().endswith(".xls"):
        from xls_reader import read_xls_sheets
        return read_xls_sheets(filepath)
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    raw = {sh: pd.read_excel(filepath, sheet_name=sh,
                             engine="openpyxl", header=None)
           for sh in xl.sheet_names}
    return xl.sheet_names, raw


def read_bet_xls(filepath: str) -> dict:
    """
    Parse the XLS/XLSX file produced by the BET instrument.
    Returns a dict with keys: ads, des, bet_pts, bjh, summary

    Raises
    ------
    ValueError
        If expected sheet names or row labels are not found in the file.
    """
    sheet_names, raw = _load_sheets(filepath)

    required_sheets = {"AdsDes", "BET", "BJH", "Summary"}
    missing = required_sheets - set(sheet_names)
    if missing:
        raise ValueError(
            f"Missing required sheet(s) in XLS file: {missing}. "
            f"Found sheets: {sheet_names}"
        )

    # ── Adsorption / Desorption isotherm ──────────────────────
    df = raw["AdsDes"]

    ads_rows = df[df.iloc[:, 0] == "ADS"].index
    if len(ads_rows) == 0:
        raise ValueError(
            "Label 'ADS' not found in sheet 'AdsDes'. "
            "Check instrument XLS format — expected label in column 0."
        )
    ads_start = ads_rows[0] + 1

    des_rows = df[df.iloc[:, 0] == "DES"].index
    if len(des_rows) == 0:
        raise ValueError(
            "Label 'DES' not found in sheet 'AdsDes'. "
            "Check instrument XLS format — expected label in column 0."
        )
    des_start = des_rows[0] + 1

    def _extract(start, end_label):
        rows = []
        for i in range(start, len(df)):
            if end_label is not None and df.iloc[i, 0] == end_label:
                break
            row = df.iloc[i]
            try:
                pp0 = float(row[5])
                va  = float(row[6])
                rows.append((pp0, va))
            except (ValueError, TypeError):
                break
        return np.array(rows)

    ads = _extract(ads_start, "DES")
    des = _extract(des_start, None)

    if len(ads) == 0:
        raise ValueError("No adsorption data points could be parsed from sheet 'AdsDes'.")

    # ── BET plot points ───────────────────────────────────────
    df_b = raw["BET"]
    no_rows_b = df_b[df_b.iloc[:, 0] == "No"].index
    if len(no_rows_b) == 0:
        raise ValueError("Label 'No' not found in BET sheet. Cannot locate data table.")
    no_row = no_rows_b[0] + 1

    bet_pts = []
    for i in range(no_row, len(df_b)):
        try:
            pp0 = float(df_b.iloc[i, 1])
            y   = float(df_b.iloc[i, 2])
            bet_pts.append((pp0, y))
        except (ValueError, TypeError):
            break
    bet_pts = np.array(bet_pts)

    start_pt_rows = df_b[df_b.iloc[:, 0] == "Starting point"]
    end_pt_rows   = df_b[df_b.iloc[:, 0] == "End point"]
    if start_pt_rows.empty or end_pt_rows.empty:
        raise ValueError("'Starting point' or 'End point' labels not found in BET sheet.")
    start_pt = int(start_pt_rows.iloc[0, 3])
    end_pt   = int(end_pt_rows.iloc[0, 3])

    # ── BJH pore size distribution ────────────────────────────
    df_j = raw["BJH"]
    no_rows_j = df_j[df_j.iloc[:, 0] == "No"].index
    if len(no_rows_j) == 0:
        raise ValueError("Label 'No' not found in BJH sheet. Cannot locate data table.")
    no_row_j = no_rows_j[0] + 1

    bjh_rows = []
    for i in range(no_row_j, len(df_j)):
        try:
            rp     = float(df_j.iloc[i, 2])
            dVpdrp = float(df_j.iloc[i, 3])
            SVp    = float(df_j.iloc[i, 4])
            Sap    = float(df_j.iloc[i, 5])
            bjh_rows.append((rp, dVpdrp, SVp, Sap))
        except (ValueError, TypeError):
            break
    bjh = np.array(bjh_rows)  # cols: rp(nm), dVp/drp, cum_Vp, cum_Sap

    # ── Summary values ────────────────────────────────────────
    df_s = raw["Summary"]
    def _get(label):
        mask = df_s.iloc[:, 0] == label
        if not mask.any():
            return np.nan
        row = df_s.loc[mask].iloc[0]
        for col in [3, 2]:
            try:
                v = float(row.iloc[col])
                if not np.isnan(v):
                    return v
            except (ValueError, TypeError):
                pass
        return np.nan

    summary = {
        "Vm"              : _get("Vm"),
        "S_BET"           : _get("as,BET"),
        "C"               : _get("C"),
        "Vp_total"        : _get("Total pore volume(p/p0=0.990)"),
        "dp_avg"          : _get("Average pore diameter"),
        "rp_peak_BJH"     : _get("rp,peak(Area)"),
        "S_BJH"           : _get("ap"),
        "Vp_BJH"          : _get("Vp"),
        "start_pt"        : start_pt,
        "end_pt"          : end_pt,
    }
    return dict(ads=ads, des=des, bet_pts=bet_pts, bjh=bjh, summary=summary)


# ══════════════════════════════════════════════════════════════
# 2. ISOTHERM CLASSIFICATION
# ══════════════════════════════════════════════════════════════

# Normalised loop-area threshold above which an adsorption/desorption branch
# pair is treated as a mesopore capillary-condensation loop (the Type IV/V
# signature). This is a heuristic, not a physical law. Measured normalised
# areas on the audit samples:
#   - 13BgOH.xls : 0.0125   (below -> no loop)
#   - 14H.xls    : 0.0215   (~8% above -> loop, uncertain)
#   - 9.xls      : 0.0221   (~10% above -> loop, uncertain)
#   - g-OH.xls   : 0.0459   (well above -> loop)
#   - 10.xls     : 0.0518   (well above -> loop)
# Synthetic fixtures: TypeIV_H1 = 0.0334, TypeV_H2 = 0.0317 (both above).
# 14H and 9 sit only ~8-10% above the threshold, so classifications whose loop
# area is near 0.02 are uncertain. Exposed as a keyword argument on
# classify_isotherm so it can be overridden.
HYSTERESIS_AREA_THRESHOLD = 0.02

# Minimum number of desorption-branch points required before a loop can be
# defined at all; a 1-2 point branch cannot close a loop.
MIN_HYSTERESIS_POINTS = 3


def hysteresis_loop(ads: np.ndarray, des: np.ndarray) -> dict:
    """Interpolate both branches onto a common p/p0 grid and compute the loop.

    The positive part of (desorption − adsorption) is integrated over the
    p/p0 overlap of the two branches and normalised by the maximum adsorbed
    amount. This is the single place the normalised loop area is computed;
    both :func:`classify_isotherm` and :func:`classify_hysteresis` call it.

    A desorption branch with fewer than ``MIN_HYSTERESIS_POINTS`` points, or
    branches that do not overlap in p/p0, cannot define a loop; in both cases
    ``norm_area`` is ``0.0`` and the interpolation arrays are ``None``.

    Returns
    -------
    dict with keys ``norm_area``, and — when a loop is defined — ``p_grid``,
    ``Va_a_g``, ``Va_d_g``, ``hyst``, ``p_lo``, ``p_hi``.
    """
    empty = {
        "norm_area": 0.0,
        "p_grid": None, "Va_a_g": None, "Va_d_g": None, "hyst": None,
        "p_lo": None, "p_hi": None,
    }
    if len(des) < MIN_HYSTERESIS_POINTS:
        return empty

    pp0_a, Va_a = ads[:, 0], ads[:, 1]
    pp0_d, Va_d = des[:, 0], des[:, 1]

    sort_d = np.argsort(pp0_d)
    pp0_d, Va_d = pp0_d[sort_d], Va_d[sort_d]

    p_lo = max(pp0_a.min(), pp0_d.min())
    p_hi = min(pp0_a.max(), pp0_d.max())
    if p_hi <= p_lo:
        return empty

    p_grid = np.linspace(p_lo, p_hi, 200)
    f_ads = interp1d(pp0_a, Va_a, bounds_error=False, fill_value="extrapolate")
    f_des = interp1d(pp0_d, Va_d, bounds_error=False, fill_value="extrapolate")

    Va_a_g = f_ads(p_grid)
    Va_d_g = f_des(p_grid)
    hyst = np.clip(Va_d_g - Va_a_g, 0, None)

    hyst_area = float(_trapezoid(hyst, p_grid))
    norm_area = hyst_area / (Va_a.max() + 1e-9)

    return {
        "norm_area": norm_area,
        "p_grid": p_grid, "Va_a_g": Va_a_g, "Va_d_g": Va_d_g, "hyst": hyst,
        "p_lo": p_lo, "p_hi": p_hi,
    }


def classify_isotherm(ads: np.ndarray, des: np.ndarray,
                      hysteresis_threshold: float = HYSTERESIS_AREA_THRESHOLD) -> dict:
    """
    IUPAC 2015 physisorption isotherm classification.
    Ref: Thommes et al., Pure Appl. Chem. 87, 1051–1069 (2015).

    Strategy:
      Step 1 — detect a capillary-condensation loop via loop area (→ Type IV/V)
      Step 2 — examine low-p/p0 concavity (IV vs V)
      Step 3 — no condensation loop: shape analysis (I, II, III, VI)
      Step 4 — Type I sub-classification: I(a) vs I(b)

    ``hysteresis_threshold`` is the minimum normalised loop area
    (:func:`hysteresis_loop`) for a branch pair to count as a
    *capillary-condensation* loop (the Type IV/V signature per Thommes et al.
    2015 §4.2), **not** the presence of any hysteresis at all — a small loop
    (e.g. an H3 loop sitting on a Type II adsorption branch, §4.3.2) falls
    below this threshold and is reported by :func:`classify_hysteresis`
    instead.
    """
    pp0_a, Va_a = ads[:, 0], ads[:, 1]
    has_condensation_loop = \
        hysteresis_loop(ads, des)["norm_area"] >= hysteresis_threshold

    # -- Concavity at low relative pressure ----------------------
    low_mask = pp0_a < 0.35
    if low_mask.sum() > 2:
        x_l = pp0_a[low_mask]
        y_l = Va_a[low_mask]
        d2  = np.gradient(np.gradient(y_l, x_l), x_l)
        concave_low = float(d2.mean()) < 0
        concave_low_measured = True
    else:
        # Fewer than 3 points below p/p0 = 0.35: concavity is defaulted, not
        # measured. Mark it so the classification can refuse to guess.
        concave_low = True
        concave_low_measured = False

    # -- Plateau check at high p/p0 ------------------------------
    high_mask = pp0_a > 0.75
    if high_mask.sum() > 2:
        Va_high   = Va_a[high_mask]
        variation = (Va_high.max() - Va_high.min()) / Va_high.mean()
        has_plateau = variation < 0.25
    else:
        has_plateau = False

    # -- Initial steep rise (micropores) -------------------------
    very_low_mask = pp0_a < 0.1
    if very_low_mask.sum() > 1:
        slope_init = (Va_a[very_low_mask][-1] - Va_a[very_low_mask][0]) / \
                     (pp0_a[very_low_mask][-1] - pp0_a[very_low_mask][0] + 1e-9)
        steep_init = slope_init > 100
    else:
        steep_init = False

    # -- Stepped isotherm check ----------------------------------
    dVa = np.diff(Va_a)
    pp0_mid = 0.5 * (pp0_a[:-1] + pp0_a[1:])
    peaks = np.where((dVa > dVa.mean() + 2 * dVa.std()) &
                     (pp0_mid > 0.1) & (pp0_mid < 0.9))[0]
    is_stepped = len(peaks) >= 2

    # -- Ultra-narrow micropore check (Type I(a) vs I(b)) -------
    ultra_low_mask = pp0_a < 0.01
    if ultra_low_mask.sum() > 1 and steep_init:
        frac_ultra = Va_a[ultra_low_mask].max() / (Va_a.max() + 1e-9)
        is_type_Ia = frac_ultra > 0.5
    else:
        is_type_Ia = False

    # -- Classification ------------------------------------------
    # TODO: Type IV(b) is currently unreachable. Per Thommes et al. (2015) §4.2,
    # a mesoporous adsorbent with pores below a critical width (~4 nm for N2 in
    # cylindrical pores at 77 K) gives a completely *reversible* Type IVb
    # isotherm with no hysteresis loop. Our classifier can only send such a
    # sample to I/II/III/VI. Discriminating IV(a)/IV(b) needs pore-size input
    # and is deferred to a later phase; every Type IV produced here is
    # hysteresis-bearing, hence the "Type IV(a)" label below.
    if is_stepped and not has_condensation_loop:
        iso_type = "Type VI"
        explanation = ("Stepped isotherm. Multilayer adsorption on a "
                       "uniform non-porous surface.")
    elif has_condensation_loop:
        if concave_low:
            iso_type = "Type IV(a)"
            explanation = ("Hysteresis loop present + concave at low p/p₀. "
                           "Characteristic of mesoporous materials. "
                           "Monolayer–multilayer adsorption followed by "
                           "capillary condensation in mesopores.")
        else:
            iso_type = "Type V"
            explanation = ("Hysteresis loop present + convex at low p/p₀. "
                           "Weak adsorbate–adsorbent interactions combined "
                           "with mesoporosity.")
    else:
        # No hysteresis loop. Low-p/p0 concavity is what separates Type II
        # (concave — strong adsorbate–adsorbent interaction) from Type III
        # (convex — weak interaction); ``has_plateau`` must not be able to
        # send a concave isotherm to Type III.
        if steep_init and has_plateau:
            if is_type_Ia:
                iso_type = "Type I(a)"
                explanation = ("Very steep rise at p/p₀ < 0.01 — indicates "
                               "ultra-micropores (< 1 nm). Typical of activated "
                               "carbons or zeolites with narrow pore size distribution.")
            else:
                iso_type = "Type I(b)"
                explanation = ("Steep rise extending to p/p₀ ~ 0.1 — indicates "
                               "micropores in range 1–2.5 nm plus possibly narrow "
                               "mesopores. Common in MOFs and hierarchical carbons.")
        elif concave_low and concave_low_measured:
            # Type II = unrestricted monolayer–multilayer adsorption: uptake
            # rises without limit as p/p0 -> 1, so a plateau is not required
            # (a genuine Type II has none). ``concave_low and has_plateau``
            # (without ``steep_init``) also lands here — that combination is
            # still a strong-interaction multilayer isotherm, not Type III.
            iso_type = "Type II"
            explanation = ("Concave at low p/p₀ — strong adsorbate–adsorbent "
                           "interaction. Unrestricted monolayer–multilayer "
                           "adsorption whose thickness increases without limit "
                           "as p/p0 → 1. Characteristic of nonporous or "
                           "macroporous adsorbents; the same adsorption-branch "
                           "shape is also carried by non-rigid plate-like "
                           "aggregates, where it is accompanied by an H3 "
                           "hysteresis loop (Thommes et al. 2015 §4.2, §4.3.2).")
        elif not concave_low:
            iso_type = "Type III"
            explanation = ("Convex throughout. Weak adsorbate–adsorbent "
                           "interactions, multilayer adsorption.")
        else:
            # concave_low was defaulted to True because fewer than 3 points
            # lie below p/p0 = 0.35 — the concavity is a guess, not a
            # measurement, so we decline to classify.
            iso_type = "Unclassified"
            explanation = ("Concavity at low p/p₀ could not be measured "
                           "(fewer than 3 points below p/p0 = 0.35), so the "
                           "isotherm cannot be classified confidently.")

    return {"type": iso_type, "explanation": explanation,
            "has_hysteresis": has_condensation_loop,
            "has_condensation_loop": has_condensation_loop,
            "concave_low": concave_low,
            "has_plateau": has_plateau}


# ══════════════════════════════════════════════════════════════
# 3. HYSTERESIS CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def classify_hysteresis(ads: np.ndarray, des: np.ndarray) -> dict:
    """
    IUPAC 2015 hysteresis classification (H1–H4).
    Ref: Thommes et al., Pure Appl. Chem. 87, 1051–1069 (2015).

    Scoring approach — each feature votes for a type:
      H1 : steep + parallel branches, narrow loop  → uniform cylinders
      H2 : gentle ads, steep des (triangular loop) → ink-bottle pores
      H3 : no plateau, non-rigid slit-shaped       → plate aggregates
      H4 : nearly flat + narrow loop               → slit + micropores

    The ``score_share`` field is the winning type's share of the total score
    (a heuristic for how decisively one type wins), not a probability. On a
    score tie all tied types are returned joined by "/" with a "low"
    share label rather than being broken by dict insertion order.
    """
    loop = hysteresis_loop(ads, des)
    if loop["p_grid"] is None:
        return {"type": "None", "explanation": "No hysteresis detected.",
                "scores": {}, "features": {}}

    pp0_a, Va_a = ads[:, 0], ads[:, 1]

    p_grid = loop["p_grid"]
    Va_a_g = loop["Va_a_g"]
    Va_d_g = loop["Va_d_g"]
    hyst   = loop["hyst"]
    p_lo   = loop["p_lo"]

    # ── Feature 1: hysteresis area (normalised) ────────────────
    norm_area = loop["norm_area"]

    # ── Feature 2: slope ratio ─────────────────────────────────
    ads_slopes  = np.abs(np.gradient(Va_a_g, p_grid))
    des_slopes  = np.abs(np.gradient(Va_d_g, p_grid))
    mid = (p_grid > 0.3) & (p_grid < 0.95)
    ratio_max   = float(des_slopes[mid].max() /
                        (ads_slopes[mid].max() + 1e-9))
    ratio_mean  = float(des_slopes[mid].mean() /
                        (ads_slopes[mid].mean() + 1e-9))

    # ── Feature 3: loop shape ──────────────────────────────────
    peak_idx   = np.argmax(hyst)
    peak_pos   = p_grid[peak_idx]
    is_left_skewed = peak_pos < 0.65

    # ── Feature 4: plateau on adsorption branch ────────────────
    high_ads_mask = pp0_a > 0.75
    if high_ads_mask.sum() > 2:
        Va_hi  = Va_a[high_ads_mask]
        plateau_variation = (Va_hi.max() - Va_hi.min()) / (Va_hi.mean() + 1e-9)
        has_plateau = plateau_variation < 0.35
    else:
        has_plateau = False

    # ── Feature 5: flatness at low p/p0 ───────────────────────
    low_ads_mask = pp0_a < 0.5
    if low_ads_mask.sum() > 3:
        Va_lo = Va_a[low_ads_mask]
        pp0_lo = pp0_a[low_ads_mask]
        flat_slope = (Va_lo[-1] - Va_lo[0]) / (pp0_lo[-1] - pp0_lo[0] + 1e-9)
        is_flat_low = flat_slope < 30
    else:
        is_flat_low = False

    # ── Feature 6: closure point ───────────────────────────────
    hyst_open = hyst > hyst.max() * 0.05
    if hyst_open.any():
        closure_p = float(p_grid[hyst_open][0])
    else:
        closure_p = p_lo
    forced_closure = closure_p < 0.45

    # ── Scoring ───────────────────────────────────────────────
    scores = {"H1": 0, "H2": 0, "H3": 0, "H4": 0}

    if ratio_mean < 2.0 and norm_area < 0.15 and has_plateau:
        scores["H1"] += 3
    if ratio_max < 2.5:
        scores["H1"] += 1

    if ratio_max > 2.0:
        scores["H2"] += 3
    if is_left_skewed:
        scores["H2"] += 2
    if has_plateau:
        scores["H2"] += 1

    if not has_plateau:
        scores["H3"] += 3
    if norm_area > 0.20:
        scores["H3"] += 2
    if not is_left_skewed:
        scores["H3"] += 1

    if is_flat_low and norm_area < 0.12:
        scores["H4"] += 3
    if not has_plateau and norm_area < 0.18:
        scores["H4"] += 1

    # Deterministic tie-break. ``max(scores, key=scores.get)`` resolves a tie
    # by dict insertion order (a Python implementation detail, undocumented).
    # Instead, on a tie we report every tied type joined by "/" (e.g.
    # "H3/H4") with a "low" score-share label, so the ambiguity is explicit
    # rather than silently resolved.
    max_score = max(scores.values())
    tied = [k for k in ("H1", "H2", "H3", "H4") if scores[k] == max_score]
    best = tied[0] if len(tied) == 1 else "/".join(tied)

    # share of the total score carried by the winning type(s) — a heuristic
    # measure of how decisively one type wins, NOT a probability. Kept as a
    # number but labelled as a score share in reports and the UI.
    score_share = max_score / (sum(scores.values()) + 1e-9)

    explanations = {
        "H1": ("Narrow, symmetric loop. Both adsorption and desorption "
               "branches are steep and nearly parallel. Associated with "
               "uniform, open-ended cylindrical mesopores."),
        "H2": ("Triangular loop with steeper desorption branch. "
               "Indicates ink-bottle (narrow neck) pores or pore-blocking "
               "and cavitation effects. Common in disordered mesoporous "
               "materials."),
        "H3": ("Loop does not show limiting adsorption near p/p₀ → 1. "
               "Associated with non-rigid aggregates of plate-like "
               "particles forming slit-shaped pores (e.g. layered "
               "materials such as C₃N₄)."),
        "H4": ("Narrow loop, nearly horizontal and parallel branches. "
               "Often found in microporous solids containing mesopores "
               "and narrow slit-shaped pores."),
    }

    if len(tied) == 1:
        explanation = explanations[best]
        score_share_label = ("high" if score_share > 0.55 else
                             "moderate" if score_share > 0.40 else "low")
    else:
        explanation = ("Score tie between " + " and ".join(tied) + ". "
                       "The measured features are consistent with more than "
                       "one loop type; the ambiguity is reported instead of "
                       "forcing a single label.")
        score_share_label = "low"

    features = {
        "hysteresis_area_norm" : round(norm_area, 4),
        "slope_ratio_max"      : round(ratio_max, 3),
        "slope_ratio_mean"     : round(ratio_mean, 3),
        "peak_position_p/p0"   : round(float(peak_pos), 3),
        "has_plateau_ads"      : has_plateau,
        "flat_at_low_pp0"      : is_flat_low,
        "closure_point_p/p0"   : round(closure_p, 3),
        "forced_closure_N2"    : forced_closure,
    }

    return {"type": best,
            "explanation": explanation,
            "score_share": score_share_label,
            "score_share_pct": round(score_share * 100, 1),
            "scores": scores,
            "features": features}


# ══════════════════════════════════════════════════════════════
# 4. BET PLOT VERIFICATION
# ══════════════════════════════════════════════════════════════

def verify_bet(bet_pts: np.ndarray, summary: dict,
               ads: np.ndarray = None) -> dict:
    """
    Re-fit the BET linearisation using the same point range
    as the instrument and compare results.

    BET linear form:
        1/[Va(p0/p - 1)] = (C-1)/(Vm·C) · (p/p0) + 1/(Vm·C)
        → Vm = 1/(slope + intercept)
        → C  = 1 + slope/intercept
        → S_BET = Vm × N2_BET_FACTOR  (m²/g, N₂ σ=0.162 nm² at 77 K)

    Notes
    -----
    - Va and Vm must be in cm³(STP)/g.
    - Valid BET range: 0.05 ≤ p/p₀ ≤ 0.35 (IUPAC 2015).
    - C constant must be positive; negative C indicates invalid range.
    - If `ads` (the adsorption branch, columns p/p0 and Va) is given,
      a Rouquerol auto range selection is run and attached to the result.

    Warns
    -----
    UserWarning
        If C constant is negative (selected p/p₀ range is outside
        the valid BET region — adjust start_pt/end_pt).
    """
    s   = summary["start_pt"]
    e   = summary["end_pt"] + 1
    pts = bet_pts[s:e]

    x, y = pts[:, 0], pts[:, 1]
    slope, intercept, r, *_ = linregress(x, y)
    R2   = r ** 2
    Vm   = 1.0 / (slope + intercept)
    C    = 1.0 + slope / intercept
    S_BET = Vm * N2_BET_FACTOR

    # ── IUPAC 2015 validity check: C must be positive ──────────
    if C < 0:
        warnings.warn(
            f"BET C constant is negative (C = {C:.2f}). "
            "The selected p/p₀ range is outside the valid BET region. "
            "Per IUPAC 2015, adjust start_pt/end_pt so that all selected "
            "points lie within 0.05 ≤ p/p₀ ≤ 0.35 and Va(p₀/p - 1) "
            "increases monotonically with p/p₀. "
            "Results from this fit should not be reported.",
            UserWarning,
            stacklevel=2,
        )

    # ── Additional consistency check ───────────────────────────
    if not np.all(np.diff(y) > 0):
        warnings.warn(
            "BET linearisation y-values are not strictly monotonically "
            "increasing over the selected range. Consider revising the "
            "point selection (start_pt/end_pt).",
            UserWarning,
            stacklevel=2,
        )

    # ── Rouquerol auto range (optional) ───────────────────────
    rouquerol_result = None
    if ads is not None:
        try:
            rouquerol_result = select_bet_range(ads[:, 0], ads[:, 1])
        except Exception:
            rouquerol_result = None

    return dict(
        x=x, y=y, slope=slope, intercept=intercept,
        R2=R2, Vm=Vm, C=C,
        S_BET_calc=S_BET,
        S_BET_instrument=summary["S_BET"],
        C_valid=(C > 0),
        all_pts=bet_pts,
        rouquerol_result=rouquerol_result,
    )


# ══════════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_all(data: dict, iso_cls: dict, hyst_cls: dict,
             bet_res: dict, sample_name: str, save: bool = True,
             show: bool = True):
    """
    4-panel publication figure:
      [A] N₂ Adsorption–Desorption Isotherm
      [B] BET Plot
      [C] BJH Differential Pore Size Distribution (adsorption branch)
      [D] Cumulative Pore Volume + BET vs BJH comparison
    """
    setup_plot_style()

    ads = data["ads"];  des = data["des"]
    bjh = data["bjh"];  s   = data["summary"]

    fig = plt.figure(figsize=(7.2, 6.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.42, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c])
            for r, c in [(0,0),(0,1),(1,0),(1,1)]]

    # ── [A] Isotherm ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(ads[:, 0], ads[:, 1], "o-", color=C_ADS,
            ms=4, lw=1.4, label="Adsorption")
    if len(des):
        sort_d = np.argsort(des[:, 0])[::-1]
        ax.plot(des[sort_d, 0], des[sort_d, 1], "s--",
                color=C_DES, ms=4, lw=1.4, label="Desorption")
        # Shade only the hysteresis loop: the p/p0 interval where both branches
        # exist, bounded by the two interpolated branches (not a single polygon
        # with straight closing edges, which produced a full-width wedge).
        p_lo = max(ads[:, 0].min(), des[:, 0].min())
        p_hi = min(ads[:, 0].max(), des[:, 0].max())
        if p_hi > p_lo:
            p_grid = np.linspace(p_lo, p_hi, 200)
            s_a = np.argsort(ads[:, 0])
            s_d = np.argsort(des[:, 0])
            Va_a_g = np.interp(p_grid, ads[s_a, 0], ads[s_a, 1])
            Va_d_g = np.interp(p_grid, des[s_d, 0], des[s_d, 1])
            ax.fill_between(p_grid, Va_a_g, Va_d_g, alpha=0.10,
                            color=C_ADS, linewidth=0)

    ax.set_xlabel(r"Relative Pressure ($p/p_0$)")
    ax.set_ylabel(r"Volume Adsorbed (cm$^3$ g$^{-1}$ STP)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    iso_label  = iso_cls["type"]
    hyst_label = hyst_cls["type"] if hyst_cls["type"] != "None" else ""
    ax_ann = iso_label + (f" / {hyst_label}" if hyst_label else "")
    ax.text(0.03, 0.78, ax_ann, transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="0.7", lw=0.7, alpha=0.9))
    _label_panel(ax, "A")

    # ── [B] BET Plot ──────────────────────────────────────────
    ax = axes[1]
    ax.scatter(bet_res["all_pts"][:, 0], bet_res["all_pts"][:, 1],
               color="0.75", s=20, zorder=2, label="Unused points")
    ax.scatter(bet_res["x"], bet_res["y"],
               color=C_BET, s=30, zorder=4, label="Fitted points")
    x_fit = np.linspace(bet_res["x"].min(), bet_res["x"].max(), 200)
    y_fit = bet_res["slope"] * x_fit + bet_res["intercept"]
    ax.plot(x_fit, y_fit, "-", color=C_BET, lw=1.6, zorder=3)

    ax.set_xlabel(r"$p/p_0$")
    ax.set_ylabel(r"$\frac{1}{V_\mathrm{a}(p_0/p - 1)}$  (g cm$^{-3}$)",
                  labelpad=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    c_flag = "" if bet_res["C_valid"] else "  ⚠ C<0"
    txt = (f"$S_{{BET}}$ = {s['S_BET']:.2f} m² g⁻¹\n"
           f"$C$ = {s['C']:.1f}{c_flag}\n"
           f"$R^2$ = {bet_res['R2']:.5f}")
    ax.text(0.05, 0.94, txt, transform=ax.transAxes,
            va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="0.7", lw=0.7, alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)
    _label_panel(ax, "B")

    # ── [C] BJH Differential PSD (adsorption branch) ─────────
    # IUPAC note: adsorption BJH avoids the ~3.4 nm N₂ cavitation
    # artefact that appears in desorption BJH at 77 K (p/p₀ ≈ 0.42).
    ax = axes[2]
    # Instrument headers verified as radius ("rp/nm") and per-radius
    # differential ("dVp/drp"), so rp*2 = diameter and dV/dd = dV/dr / 2.
    rp    = bjh[:, 0] * 2          # radius (nm) -> diameter (nm)
    dVdd  = bjh[:, 1] / 2.0        # dVp/drp -> dVp/ddp

    # Upper x-limit from the data, independent of the row order the instrument
    # used (BELSORP writes large→small): sort a local copy by diameter, then find
    # the smallest diameter where V_below(d) — the pore volume contained in pores
    # of diameter <= d — reaches 99 % of the total.
    order = np.argsort(rp)
    rp_s  = rp[order]
    cv_s  = bjh[order, 2]
    total = float(cv_s.max())
    # V_below rises with diameter. Ascending instruments store it directly; a
    # descending (BELSORP) column stores the complement (volume in pores >= d),
    # so flip it. The 99 % test below is identical in both directions.
    if cv_s[-1] < cv_s[0]:             # accumulated from the large-diameter end
        v_below = total - cv_s
        decreasing = True
    else:                              # accumulated from the small-diameter end
        v_below = cv_s
        decreasing = False
    x_max = float(rp_s[-1])
    if total > 0:
        idx = np.where(v_below >= 0.99 * total)[0]
        if len(idx):
            x_max = float(rp_s[idx[0]])
    x_max = max(x_max, 5.0)

    ax.plot(rp, dVdd, "-", color=C_BJH, lw=1.5)
    ax.fill_between(rp, dVdd, alpha=0.15, color=C_BJH)

    peak_idx = np.argmax(dVdd)
    ax.axvline(rp[peak_idx], ls="--", lw=0.9, color=C_BJH, alpha=0.7)
    ax.text(rp[peak_idx] + 0.5, dVdd[peak_idx] * 0.95,
            f"{rp[peak_idx]:.1f} nm", fontsize=8, color=C_BJH)

    ax.set_xlabel(r"Pore Diameter (nm)")
    ax.set_ylabel(r"d$V_p$/d$d_p$  (cm$^3$ g$^{-1}$ nm$^{-1}$)")
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.axvline(N2_CAVITATION_NM, ls=":", lw=0.8, color="0.6", alpha=0.7)
    ax.text(N2_CAVITATION_NM, ax.get_ylim()[1] * 0.9,
            "cavitation\n(~3.4 nm)", fontsize=6.5, color="0.5",
            va="top", ha="left")
    _label_panel(ax, "C")

    # ── [D] Cumulative Pore Volume ────────────────────────────
    ax  = axes[3]
    ax2 = ax.twinx()

    cum_Vp  = bjh[:, 2]
    cum_Sap = bjh[:, 3]

    ax.plot(rp, cum_Vp, "-", color=C_CUM, lw=1.5,
            label=r"$V_p$ cumulative")
    ax2.plot(rp, cum_Sap, "--", color=C_BJH, lw=1.5,
             label=r"$S_{ap}$ cumulative")

    ax2.axhline(s["S_BET"], ls=":", lw=1.0, color=C_BET,
                label=f"$S_{{BET}}$ = {s['S_BET']:.1f} m² g⁻¹")
    ax2.axhline(s["S_BJH"], ls=":", lw=1.0, color=C_BJH,
                label=f"$S_{{BJH}}$ = {s['S_BJH']:.1f} m² g⁻¹")

    ax.set_xlabel(r"Pore Diameter (nm)")
    vp_dir = " (from large d)" if decreasing else ""
    ax.set_ylabel(r"Cum. Pore Volume" + vp_dir + r" (cm$^3$ g$^{-1}$)",
                  color=C_CUM)
    ax2.set_ylabel(r"Cum. Surface Area" + vp_dir + r" (m$^2$ g$^{-1}$)",
                   color=C_BJH)
    ax.tick_params(axis="y", colors=C_CUM)
    ax2.tick_params(axis="y", colors=C_BJH)
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0)

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2,
              fontsize=7.5, loc="lower right")
    _label_panel(ax, "D")

    fig.suptitle(f"BET/BJH Analysis — {sample_name}",
                 fontsize=12, y=1.01, fontweight="bold")

    plt.tight_layout()

    if save:
        out = f"{sample_name.replace(' ', '_')}_BET_analysis.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\n  Figure saved → {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _label_panel(ax, letter):
    ax.text(-0.13, 1.03, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# ══════════════════════════════════════════════════════════════
# 6. SUMMARY REPORT
# ══════════════════════════════════════════════════════════════

def validity_warnings(s: dict, iso_cls: dict) -> list:
    """IUPAC-validity warnings for the reported quantities (Thommes et al. 2015).

    Report-text only: returns strings for display and never alters a computed
    value. ``s`` is the instrument ``summary`` dict from :func:`read_bet_xls`;
    ``iso_cls`` is the result of :func:`classify_isotherm`.
    """
    notes = []
    C = s.get("C", np.nan)
    iso_type = iso_cls.get("type", "")

    # §5.1.1 — BET C constant and Point B.
    if np.isfinite(C):
        if C < BET_C_NOT_APPLICABLE:
            notes.append("BET C < 2 — the isotherm is Type III/V and the BET "
                         "method is not applicable (Thommes et al. 2015 §5.1.1).")
        elif C < BET_C_POINT_B_QUESTIONABLE:
            notes.append("BET C < 50 — Point B cannot be identified as a single "
                         "point and the interpretation of n_m is questionable "
                         "(Thommes et al. 2015 §5.1.1).")
        elif C >= BET_C_KNEE_SHARP:
            notes.append("BET C ≥ 80 — the knee is sharp and Point B is well "
                         "defined (Thommes et al. 2015 §5.1.1).")

    # §5.2.2 / §5.1.1 — Type I BET area is an apparent area.
    if iso_type in ("Type I(a)", "Type I(b)"):
        notes.append("Type I isotherm — the BET area is an apparent surface "
                     "area (an adsorbent 'fingerprint'), not a realistic "
                     "probe-accessible area (Thommes et al. 2015 §5.2.2, §5.1.1).")

    # §7.2 / §9 — BJH underestimates narrow mesopores.
    rp_peak = s.get("rp_peak_BJH", np.nan)
    if np.isfinite(rp_peak):
        peak_diam = rp_peak * 2.0
        if peak_diam < BJH_NARROW_MESOPORE_NM:
            notes.append(f"BJH peak diameter {peak_diam:.1f} nm is below 10 nm — "
                         "Kelvin-equation (BJH) procedures underestimate narrow "
                         "mesopore size by ~20-30% (Thommes et al. 2015 §7.2, §9).")

    # §7.1 — Gurvich total pore volume needs a near-horizontal high-p/p0 region.
    if not iso_cls.get("has_plateau", False):
        notes.append("The isotherm does not approach a plateau near p/p0 = 1 — "
                     "the Gurvich-rule total pore volume is not valid for this "
                     "(composite Type IV + Type II) isotherm (Thommes et al. 2015 §7.1).")

    return notes


def print_report(data: dict, iso_cls: dict, hyst_cls: dict,
                 bet_res: dict, sample_name: str):
    s = data["summary"]
    h = hyst_cls

    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  BET Analysis Report — {sample_name}")
    print(sep)

    rows = [
        ["BET Surface Area",    f"{s['S_BET']:.3f}",  "m² g⁻¹"],
        ["Vm (monolayer cap.)",  f"{s['Vm']:.4f}",     "cm³(STP) g⁻¹"],
        ["BET C constant",       f"{s['C']:.2f}",      "—"],
        ["Total Pore Volume",    f"{s['Vp_total']:.4f}","cm³ g⁻¹"],
        ["Average Pore Diameter",f"{s['dp_avg']:.3f}",  "nm"],
        ["BJH Surface Area",     f"{s['S_BJH']:.3f}",  "m² g⁻¹"],
        ["BJH Peak Diameter",    f"{s['rp_peak_BJH']*2:.2f}", "nm"],
    ]
    print(tabulate(rows, headers=["Parameter", "Value", "Unit"],
                   tablefmt="simple"))

    if not bet_res["C_valid"]:
        print("\n  ⚠  WARNING: BET C constant is NEGATIVE.")
        print("     The selected p/p₀ range is outside the valid BET region.")
        print("     Per IUPAC 2015, revise start_pt/end_pt (0.05 ≤ p/p₀ ≤ 0.35).")

    notes = validity_warnings(s, iso_cls)
    if notes:
        print("\n  Validity notes (IUPAC 2015)")
        for n in notes:
            print(f"    ⚠  {n}")

    print(f"\n  BET Regression  (points {s['start_pt']}–{s['end_pt']})")
    print(f"    Slope     : {bet_res['slope']:.6f}")
    print(f"    Intercept : {bet_res['intercept']:.6f}")
    print(f"    R²        : {bet_res['R2']:.6f}")
    print(f"    Vm (calc) : {bet_res['Vm']:.4f} cm³(STP) g⁻¹")
    print(f"    C (calc)  : {bet_res['C']:.2f}  {'✓ valid' if bet_res['C_valid'] else '✗ invalid — see warning above'}")

    # ── Rouquerol report ──────────────────────────────────────
    if bet_res.get("rouquerol_result") is not None:
        print(f"\n{format_rouquerol_report(bet_res['rouquerol_result'], sample_name)}")

    print(f"\n  Isotherm Classification")
    print(f"    Type        : {iso_cls['type']}")
    print(f"    Explanation : {iso_cls['explanation']}")

    if h["type"] != "None":
        print(f"\n  Hysteresis Classification")
        print(f"    Type         : {h['type']}")
        print(f"    Score share  : {h['score_share']} ({h['score_share_pct']:.0f}% of total score)")
        print(f"    Explanation  : {h['explanation']}")
        print(f"\n  Scoring:")
        for k, v in sorted(h["scores"].items(), key=lambda x: -x[1]):
            bar = "█" * v + "░" * (8 - v)
            print(f"    {k}  {bar}  {v}")
        print(f"\n  Feature Analysis:")
        feat_rows = [[k, str(v)] for k, v in h["features"].items()]
        print(tabulate(feat_rows, headers=["Feature", "Value"],
                       tablefmt="simple"))

    no_condensation_types = ("Type I(a)", "Type I(b)", "Type II", "Type III",
                             "Type VI")
    if iso_cls["type"] in no_condensation_types and h["type"] != "None":
        print(f"\n  Note: a {h['type']} hysteresis loop together with a "
              f"{iso_cls['type']} isotherm is an expected combination — an H3 "
              "loop sits on a Type II adsorption branch by definition "
              "(Thommes et al. 2015 §4.3.2).")

    ratio = s["S_BET"] / s["S_BJH"] if s["S_BJH"] else float("nan")
    print(f"\n  BET vs BJH Comparison")
    print(f"    S_BET  : {s['S_BET']:.2f} m² g⁻¹")
    print(f"    S_BJH  : {s['S_BJH']:.2f} m² g⁻¹  (adsorption branch)")
    print(f"    Ratio  : {ratio:.3f}")
    if ratio > 1.15:
        print("    Note   : S_BET > S_BJH — micropore contribution likely.")
    elif ratio < 0.85:
        print("    Note   : S_BJH > S_BET — check BJH model assumptions.")
    else:
        print("    Note   : Good agreement between BET and BJH methods.")
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════

def _configure_console():
    """Make stdout emit UTF-8 so the report never crashes a cp1252 console.

    ``errors="replace"`` degrades an unrenderable glyph to ``?`` rather than
    raising ``UnicodeEncodeError``. ``reconfigure`` is unavailable on very old
    Pythons, so fall back to a no-op there (the caller is on Python >= 3.10).
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main():
    _configure_console()
    parser = argparse.ArgumentParser(
        description="BET/BJH Analysis Tool — publication-quality figures")
    parser.add_argument("--file",   required=True,
                        help="Path to BET instrument XLS/XLSX file")
    parser.add_argument("--sample", default="Sample",
                        help="Sample name for plot title and file name")
    parser.add_argument("--no-show", action="store_true",
                        help="Suppress the interactive figure window "
                             "(the PNG is still saved)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the PNG file")
    parser.add_argument("--rouquerol", action="store_true",
                        help="Run Rouquerol auto BET range selection")
    parser.add_argument("--langmuir", action="store_true",
                        help="Run Langmuir surface-area analysis on the "
                             "adsorption branch (default window 0.05-0.30)")
    args = parser.parse_args()

    print(f"\n  Reading: {args.file}")
    data = read_bet_xls(args.file)

    iso_cls  = classify_isotherm(data["ads"], data["des"])
    hyst_cls = classify_hysteresis(data["ads"], data["des"])
    bet_res  = verify_bet(data["bet_pts"], data["summary"],
                          ads=data["ads"] if args.rouquerol else None)

    print_report(data, iso_cls, hyst_cls, bet_res, args.sample)

    if args.langmuir:
        _print_langmuir(data, iso_cls, args.sample)

    plot_all(data, iso_cls, hyst_cls, bet_res, args.sample,
             save=not args.no_save, show=not args.no_show)


def _print_langmuir(data: dict, iso_cls: dict, sample_name: str):
    """Run and print the Langmuir report for the CLI's ``--langmuir`` flag.

    Uses the conservative 0.05-0.30 window; if it holds fewer than
    ``MIN_LANGMUIR_POINTS`` points the fit raises ValueError, which is reported
    as a note rather than a traceback.
    """
    p_ads = data["ads"][:, 0]
    n_ads = data["ads"][:, 1]

    lo = max(0.05, float(p_ads.min()))
    hi = min(0.30, float(p_ads.max()))
    mask = (p_ads >= lo - 1e-9) & (p_ads <= hi + 1e-9)
    p_sel = p_ads[mask]
    n_sel = n_ads[mask]

    try:
        result = fit_langmuir_window(
            p_sel, n_sel,
            has_hysteresis=iso_cls["has_hysteresis"],
            has_plateau=iso_cls["has_plateau"],
            S_BET=data["summary"]["S_BET"],
        )
    except ValueError as e:
        print(f"\n  Langmuir: skipped — {e}")
        return

    print(f"\n{format_langmuir_report(result, sample_name)}")


if __name__ == "__main__":
    main()
