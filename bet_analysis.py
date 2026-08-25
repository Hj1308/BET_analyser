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

def classify_isotherm(ads: np.ndarray, des: np.ndarray) -> dict:
    """
    IUPAC 2015 physisorption isotherm classification.
    Ref: Thommes et al., Pure Appl. Chem. 87, 1051–1069 (2015).

    Strategy:
      Step 1 — detect hysteresis (→ Type IV or V)
      Step 2 — examine low-p/p0 concavity (IV vs V)
      Step 3 — no hysteresis: shape analysis (I, II, III, VI)
      Step 4 — Type I sub-classification: I(a) vs I(b)
    """
    pp0_a, Va_a = ads[:, 0], ads[:, 1]
    has_hyst    = len(des) > 0

    # -- Concavity at low relative pressure ----------------------
    low_mask = pp0_a < 0.35
    if low_mask.sum() > 2:
        x_l = pp0_a[low_mask]
        y_l = Va_a[low_mask]
        d2  = np.gradient(np.gradient(y_l, x_l), x_l)
        concave_low = float(d2.mean()) < 0
    else:
        concave_low = True

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
    if is_stepped and not has_hyst:
        iso_type = "Type VI"
        explanation = ("Stepped isotherm. Multilayer adsorption on a "
                       "uniform non-porous surface.")
    elif has_hyst:
        if concave_low:
            iso_type = "Type IV"
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
        elif concave_low and has_plateau:
            iso_type = "Type II"
            explanation = ("S-shaped (sigmoid) isotherm. Non-porous or "
                           "macroporous material. Unrestricted mono- to "
                           "multilayer adsorption.")
        else:
            iso_type = "Type III"
            explanation = ("Convex throughout. Weak adsorbate–adsorbent "
                           "interactions, multilayer adsorption.")

    return {"type": iso_type, "explanation": explanation,
            "has_hysteresis": has_hyst, "concave_low": concave_low,
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
    """
    if len(des) == 0:
        return {"type": "None", "explanation": "No hysteresis detected.",
                "scores": {}, "features": {}}

    pp0_a, Va_a = ads[:, 0], ads[:, 1]
    pp0_d, Va_d = des[:, 0], des[:, 1]

    sort_d = np.argsort(pp0_d)
    pp0_d, Va_d = pp0_d[sort_d], Va_d[sort_d]

    p_lo = max(pp0_a.min(), pp0_d.min())
    p_hi = min(pp0_a.max(), pp0_d.max())
    p_grid = np.linspace(p_lo, p_hi, 200)

    f_ads = interp1d(pp0_a, Va_a, bounds_error=False, fill_value="extrapolate")
    f_des = interp1d(pp0_d, Va_d, bounds_error=False, fill_value="extrapolate")

    Va_a_g = f_ads(p_grid)
    Va_d_g = f_des(p_grid)
    hyst   = np.clip(Va_d_g - Va_a_g, 0, None)

    # ── Feature 1: hysteresis area (normalised) ────────────────
    hyst_area = float(_trapezoid(hyst, p_grid))
    norm_area = hyst_area / (Va_a.max() + 1e-9)

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
    if is_flat_low:
        scores["H4"] += 2
    if not has_plateau and norm_area < 0.18:
        scores["H4"] += 1

    best = max(scores, key=scores.get)
    confidence = scores[best] / (sum(scores.values()) + 1e-9)

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

    conf_label = ("high" if confidence > 0.55 else
                  "moderate" if confidence > 0.40 else "low")

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
            "explanation": explanations[best],
            "confidence": conf_label,
            "confidence_pct": round(confidence * 100, 1),
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
             bet_res: dict, sample_name: str, save: bool = True):
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
        pp0_d_s, Va_d_s = des[sort_d, 0], des[sort_d, 1]
        pp0_all = np.concatenate([ads[:, 0], pp0_d_s[::-1]])
        Va_all  = np.concatenate([ads[:, 1], Va_d_s[::-1]])
        ax.fill(pp0_all, Va_all, alpha=0.10, color=C_ADS)

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
    ax.text(0.03, 0.96, ax_ann, transform=ax.transAxes,
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
    rp   = bjh[:, 0] * 2   # rp (nm) → diameter (nm); confirm instrument outputs rp not dp
    dVdr = bjh[:, 1]

    ax.plot(rp, dVdr, "-", color=C_BJH, lw=1.5)
    ax.fill_between(rp, dVdr, alpha=0.15, color=C_BJH)

    peak_idx = np.argmax(dVdr)
    ax.axvline(rp[peak_idx], ls="--", lw=0.9, color=C_BJH, alpha=0.7)
    ax.text(rp[peak_idx] + 0.5, dVdr[peak_idx] * 0.95,
            f"{rp[peak_idx]:.1f} nm", fontsize=8, color=C_BJH)

    ax.set_xlabel(r"Pore Diameter (nm)")
    ax.set_ylabel(r"d$V_p$/d$r_p$  (cm$^3$ g$^{-1}$ nm$^{-1}$)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.axvline(N2_CAVITATION_NM, ls=":", lw=0.8, color="0.6", alpha=0.7)
    ax.text(N2_CAVITATION_NM + 0.2,
            ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] > 0 else 0.001,
            "cavitation\n(~3.4 nm)", fontsize=6.5, color="0.5", va="bottom")
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
    ax.set_ylabel(r"Cum. Pore Volume (cm$^3$ g$^{-1}$)", color=C_CUM)
    ax2.set_ylabel(r"Cum. Surface Area (m$^2$ g$^{-1}$)", color=C_BJH)
    ax.tick_params(axis="y", colors=C_CUM)
    ax2.tick_params(axis="y", colors=C_BJH)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2,
              fontsize=7.5, loc="lower right")
    _label_panel(ax, "D")

    fig.suptitle(f"BET/BJH Analysis — {sample_name}",
                 fontsize=12, y=1.01, fontweight="bold")

    if save:
        out = f"{sample_name.replace(' ', '_')}_BET_analysis.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\n  Figure saved → {out}")

    plt.tight_layout()
    plt.show()


def _label_panel(ax, letter):
    ax.text(-0.13, 1.03, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# ══════════════════════════════════════════════════════════════
# 6. SUMMARY REPORT
# ══════════════════════════════════════════════════════════════

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
                   tablefmt="rounded_outline"))

    if not bet_res["C_valid"]:
        print("\n  ⚠  WARNING: BET C constant is NEGATIVE.")
        print("     The selected p/p₀ range is outside the valid BET region.")
        print("     Per IUPAC 2015, revise start_pt/end_pt (0.05 ≤ p/p₀ ≤ 0.35).")

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
        print(f"    Type        : {h['type']}")
        print(f"    Confidence  : {h['confidence']} ({h['confidence_pct']:.0f}%)")
        print(f"    Explanation : {h['explanation']}")
        print(f"\n  Scoring:")
        for k, v in sorted(h["scores"].items(), key=lambda x: -x[1]):
            bar = "█" * v + "░" * (8 - v)
            print(f"    {k}  {bar}  {v}")
        print(f"\n  Feature Analysis:")
        feat_rows = [[k, str(v)] for k, v in h["features"].items()]
        print(tabulate(feat_rows, headers=["Feature", "Value"],
                       tablefmt="simple"))

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

def main():
    parser = argparse.ArgumentParser(
        description="BET/BJH Analysis Tool — publication-quality figures")
    parser.add_argument("--file",   required=True,
                        help="Path to BET instrument XLS/XLSX file")
    parser.add_argument("--sample", default="Sample",
                        help="Sample name for plot title and file name")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figure without displaying")
    parser.add_argument("--rouquerol", action="store_true",
                        help="Run Rouquerol auto BET range selection")
    args = parser.parse_args()

    print(f"\n  Reading: {args.file}")
    data = read_bet_xls(args.file)

    iso_cls  = classify_isotherm(data["ads"], data["des"])
    hyst_cls = classify_hysteresis(data["ads"], data["des"])
    bet_res  = verify_bet(data["bet_pts"], data["summary"],
                          ads=data["ads"] if args.rouquerol else None)

    print_report(data, iso_cls, hyst_cls, bet_res, args.sample)
    plot_all(data, iso_cls, hyst_cls, bet_res, args.sample,
             save=not args.no_show)


if __name__ == "__main__":
    main()
