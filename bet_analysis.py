"""
BET Analysis Tool — Publication-Quality Figures
================================================
Reads instrument XLS output and computes:
  - Isotherm type classification  (IUPAC Type I–VI, including I(a)/I(b))
  - Hysteresis type classification (IUPAC H1–H4)
  - BET plot with regression verification
  - BJH differential pore size distribution
  - Cumulative pore volume
  - BET vs BJH surface area comparison

Usage:
    python bet_analysis.py --file C3N4.xls --sample "C3N4"

Author  : Hoda Jafari | github.com/Hj1308
License : MIT
"""

import argparse
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress
from scipy.interpolate import interp1d
from tabulate import tabulate


# ══════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS — N₂ at 77 K
# ══════════════════════════════════════════════════════════

N2_BET_FACTOR         = 4.353
N2_TPLOT_SLOPE_FACTOR = 15.47
N2_LIQUID_FACTOR      = 1547.0
N2_CAVITATION_NM      = 3.4


# ══════════════════════════════════════════════════════════
# MATPLOTLIB — publication settings
# ══════════════════════════════════════════════════════════

def setup_plot_style():
    """Apply publication-quality matplotlib settings."""
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

C_ADS   = "#2166AC"
C_DES   = "#D6604D"
C_BET   = "#1A7A4A"
C_BJH   = "#7B3F9E"
C_CUM   = "#C07028"
C_SHADE = "#AACCE8"


# ══════════════════════════════════════════════════════════
# 1. DATA READING
# ══════════════════════════════════════════════════════════

def _get_excel_engine(filepath: str) -> str:
    """Return the correct pandas Excel engine based on file extension."""
    return "xlrd" if str(filepath).lower().endswith(".xls") else "openpyxl"


def read_bet_xls(filepath: str) -> dict:
    """
    Parse the XLS/XLSX file produced by the BET instrument.
    Automatically selects xlrd for legacy .xls or openpyxl for .xlsx.
    Returns a dict with keys: ads, des, bet_pts, bjh, summary
    """
    engine = _get_excel_engine(filepath)
    xl = pd.ExcelFile(filepath, engine=engine)
    required_sheets = {"AdsDes", "BET", "BJH", "Summary"}
    missing = required_sheets - set(xl.sheet_names)
    if missing:
        raise ValueError(
            f"Missing required sheet(s) in XLS file: {missing}. "
            f"Found sheets: {xl.sheet_names}"
        )

    raw = {sh: pd.read_excel(filepath, sheet_name=sh,
                             engine=engine, header=None)
           for sh in xl.sheet_names}

    # ── Adsorption / Desorption isotherm ─────────────────────
    df = raw["AdsDes"]
    ads_rows = df[df.iloc[:, 0] == "ADS"].index
    if len(ads_rows) == 0:
        raise ValueError("Label 'ADS' not found in sheet 'AdsDes'.")
    ads_start = ads_rows[0] + 1

    des_rows = df[df.iloc[:, 0] == "DES"].index
    if len(des_rows) == 0:
        raise ValueError("Label 'DES' not found in sheet 'AdsDes'.")
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

    # ── BET plot points ──────────────────────────────────
    df_b = raw["BET"]
    no_rows_b = df_b[df_b.iloc[:, 0] == "No"].index
    if len(no_rows_b) == 0:
        raise ValueError("Label 'No' not found in BET sheet.")
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

    # ── BJH pore size distribution ─────────────────────────
    df_j = raw["BJH"]
    no_rows_j = df_j[df_j.iloc[:, 0] == "No"].index
    if len(no_rows_j) == 0:
        raise ValueError("Label 'No' not found in BJH sheet.")
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
    bjh = np.array(bjh_rows)

    # ── Summary values ───────────────────────────────────
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
        "Vm"          : _get("Vm"),
        "S_BET"       : _get("as,BET"),
        "C"           : _get("C"),
        "Vp_total"    : _get("Total pore volume(p/p0=0.990)"),
        "dp_avg"      : _get("Average pore diameter"),
        "rp_peak_BJH" : _get("rp,peak(Area)"),
        "S_BJH"       : _get("ap"),
        "Vp_BJH"      : _get("Vp"),
        "start_pt"    : start_pt,
        "end_pt"      : end_pt,
    }
    return dict(ads=ads, des=des, bet_pts=bet_pts, bjh=bjh, summary=summary)


# ══════════════════════════════════════════════════════════
# 2. ISOTHERM CLASSIFICATION
# ══════════════════════════════════════════════════════════

def classify_isotherm(ads: np.ndarray, des: np.ndarray) -> dict:
    pp0_a, Va_a = ads[:, 0], ads[:, 1]
    has_hyst    = len(des) > 0

    low_mask = pp0_a < 0.35
    if low_mask.sum() > 2:
        x_l = pp0_a[low_mask]; y_l = Va_a[low_mask]
        d2  = np.gradient(np.gradient(y_l, x_l), x_l)
        concave_low = float(d2.mean()) < 0
    else:
        concave_low = True

    high_mask = pp0_a > 0.75
    if high_mask.sum() > 2:
        Va_high = Va_a[high_mask]
        variation = (Va_high.max() - Va_high.min()) / Va_high.mean()
        has_plateau = variation < 0.25
    else:
        has_plateau = False

    very_low_mask = pp0_a < 0.1
    if very_low_mask.sum() > 1:
        slope_init = ((Va_a[very_low_mask][-1] - Va_a[very_low_mask][0]) /
                      (pp0_a[very_low_mask][-1] - pp0_a[very_low_mask][0] + 1e-9))
        steep_init = slope_init > 100
    else:
        steep_init = False

    dVa = np.diff(Va_a)
    pp0_mid = 0.5 * (pp0_a[:-1] + pp0_a[1:])
    peaks = np.where((dVa > dVa.mean() + 2 * dVa.std()) &
                     (pp0_mid > 0.1) & (pp0_mid < 0.9))[0]
    is_stepped = len(peaks) >= 2

    ultra_low_mask = pp0_a < 0.01
    if ultra_low_mask.sum() > 1 and steep_init:
        frac_ultra = Va_a[ultra_low_mask].max() / (Va_a.max() + 1e-9)
        is_type_Ia = frac_ultra > 0.5
    else:
        is_type_Ia = False

    if is_stepped and not has_hyst:
        iso_type = "Type VI"
        explanation = "Stepped isotherm. Multilayer adsorption on a uniform non-porous surface."
    elif has_hyst:
        if concave_low:
            iso_type = "Type IV"
            explanation = ("Hysteresis loop present + concave at low p/p\u2080. "
                           "Characteristic of mesoporous materials.")
        else:
            iso_type = "Type V"
            explanation = ("Hysteresis loop present + convex at low p/p\u2080. "
                           "Weak adsorbate\u2013adsorbent interactions combined with mesoporosity.")
    else:
        if steep_init and has_plateau:
            if is_type_Ia:
                iso_type = "Type I(a)"
                explanation = "Very steep rise at p/p\u2080 < 0.01 \u2014 ultra-micropores (< 1 nm)."
            else:
                iso_type = "Type I(b)"
                explanation = "Steep rise to p/p\u2080 ~ 0.1 \u2014 micropores 1\u20132.5 nm."
        elif concave_low and has_plateau:
            iso_type = "Type II"
            explanation = "S-shaped isotherm. Non-porous or macroporous material."
        else:
            iso_type = "Type III"
            explanation = "Convex throughout. Weak adsorbate\u2013adsorbent interactions."

    return {"type": iso_type, "explanation": explanation,
            "has_hysteresis": has_hyst, "concave_low": concave_low,
            "has_plateau": has_plateau}


# ══════════════════════════════════════════════════════════
# 3. HYSTERESIS CLASSIFICATION
# ══════════════════════════════════════════════════════════

def classify_hysteresis(ads: np.ndarray, des: np.ndarray) -> dict:
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

    hyst_area = float(np.trapz(hyst, p_grid))
    norm_area = hyst_area / (Va_a.max() + 1e-9)

    ads_slopes = np.abs(np.gradient(Va_a_g, p_grid))
    des_slopes = np.abs(np.gradient(Va_d_g, p_grid))
    mid = (p_grid > 0.3) & (p_grid < 0.95)
    ratio_max  = float(des_slopes[mid].max() / (ads_slopes[mid].max() + 1e-9))
    ratio_mean = float(des_slopes[mid].mean() / (ads_slopes[mid].mean() + 1e-9))

    peak_idx = np.argmax(hyst)
    peak_pos = p_grid[peak_idx]
    is_left_skewed = peak_pos < 0.65

    high_ads_mask = pp0_a > 0.75
    if high_ads_mask.sum() > 2:
        Va_hi = Va_a[high_ads_mask]
        plateau_variation = (Va_hi.max() - Va_hi.min()) / (Va_hi.mean() + 1e-9)
        has_plateau = plateau_variation < 0.35
    else:
        has_plateau = False

    low_ads_mask = pp0_a < 0.5
    if low_ads_mask.sum() > 3:
        Va_lo  = Va_a[low_ads_mask]
        pp0_lo = pp0_a[low_ads_mask]
        flat_slope = (Va_lo[-1] - Va_lo[0]) / (pp0_lo[-1] - pp0_lo[0] + 1e-9)
        is_flat_low = flat_slope < 30
    else:
        is_flat_low = False

    hyst_open = hyst > hyst.max() * 0.05
    closure_p = float(p_grid[hyst_open][0]) if hyst_open.any() else p_lo
    forced_closure = closure_p < 0.45

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
        "H1": ("Narrow, symmetric loop. Both branches are steep and nearly parallel. "
               "Associated with uniform, open-ended cylindrical mesopores."),
        "H2": ("Triangular loop with steeper desorption branch. "
               "Indicates ink-bottle pores or pore-blocking/cavitation effects."),
        "H3": ("Loop does not show limiting adsorption near p/p\u2080 \u2192 1. "
               "Associated with non-rigid aggregates of plate-like particles (e.g. C\u2083N\u2084)."),
        "H4": ("Narrow loop, nearly horizontal and parallel branches. "
               "Found in microporous solids containing narrow slit-shaped pores."),
    }

    conf_label = "high" if confidence > 0.55 else "moderate" if confidence > 0.40 else "low"
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

    return {"type": best, "explanation": explanations[best],
            "confidence": conf_label, "confidence_pct": round(confidence * 100, 1),
            "scores": scores, "features": features}


# ══════════════════════════════════════════════════════════
# 4. BET PLOT VERIFICATION
# ══════════════════════════════════════════════════════════

def verify_bet(bet_pts: np.ndarray, summary: dict) -> dict:
    s_idx = summary["start_pt"]
    e_idx = summary["end_pt"] + 1
    pts   = bet_pts[s_idx:e_idx]

    x, y = pts[:, 0], pts[:, 1]
    slope, intercept, r, *_ = linregress(x, y)
    R2    = r ** 2
    Vm    = 1.0 / (slope + intercept)
    C     = 1.0 + slope / intercept
    S_BET = Vm * N2_BET_FACTOR

    if C < 0:
        warnings.warn(
            f"BET C constant is negative (C = {C:.2f}). "
            "Selected p/p\u2080 range is outside the valid BET region (IUPAC 2015).",
            UserWarning, stacklevel=2,
        )
    if not np.all(np.diff(y) > 0):
        warnings.warn(
            "BET linearisation y-values are not strictly monotonically increasing.",
            UserWarning, stacklevel=2,
        )

    return dict(
        x=x, y=y, slope=slope, intercept=intercept,
        R2=R2, Vm=Vm, C=C,
        S_BET_calc=S_BET,
        S_BET_instrument=summary["S_BET"],
        C_valid=(C > 0),
        all_pts=bet_pts,
    )


# ══════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════

def plot_all(data: dict, iso_cls: dict, hyst_cls: dict,
             bet_res: dict, sample_name: str, save: bool = True):
    setup_plot_style()

    ads = data["ads"]; des = data["des"]
    bjh = data["bjh"]; s   = data["summary"]

    fig = plt.figure(figsize=(7.2, 6.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]

    ax = axes[0]
    ax.plot(ads[:, 0], ads[:, 1], "o-", color=C_ADS, ms=4, lw=1.4, label="Adsorption")
    if len(des):
        sort_d = np.argsort(des[:, 0])[::-1]
        ax.plot(des[sort_d, 0], des[sort_d, 1], "s--", color=C_DES,
                ms=4, lw=1.4, label="Desorption")
        pp0_all = np.concatenate([ads[:, 0], des[sort_d, 0][::-1]])
        Va_all  = np.concatenate([ads[:, 1], des[sort_d, 1][::-1]])
        ax.fill(pp0_all, Va_all, alpha=0.10, color=C_ADS)
    ax.set_xlabel(r"Relative Pressure ($p/p_0$)")
    ax.set_ylabel(r"Volume Adsorbed (cm$^3$ g$^{-1}$ STP)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    iso_label  = iso_cls["type"]
    hyst_label = hyst_cls["type"] if hyst_cls["type"] != "None" else ""
    ax_ann = iso_label + (f" / {hyst_label}" if hyst_label else "")
    ax.text(0.03, 0.96, ax_ann, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.7))
    _label_panel(ax, "A")

    ax = axes[1]
    ax.scatter(bet_res["all_pts"][:, 0], bet_res["all_pts"][:, 1],
               color="0.75", s=20, zorder=2, label="Unused points")
    ax.scatter(bet_res["x"], bet_res["y"], color=C_BET, s=30, zorder=4, label="Fitted points")
    x_fit = np.linspace(bet_res["x"].min(), bet_res["x"].max(), 200)
    ax.plot(x_fit, bet_res["slope"] * x_fit + bet_res["intercept"],
            "-", color=C_BET, lw=1.6, zorder=3)
    ax.set_xlabel(r"$p/p_0$")
    ax.set_ylabel(r"$\frac{1}{V_\mathrm{a}(p_0/p - 1)}$  (g cm$^{-3}$)", labelpad=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    c_flag = "" if bet_res["C_valid"] else "  \u26a0 C<0"
    txt = (f"$S_{{BET}}$ = {s['S_BET']:.2f} m\u00b2 g\u207b\u00b9\n"
           f"$C$ = {s['C']:.1f}{c_flag}\n"
           f"$R^2$ = {bet_res['R2']:.5f}")
    ax.text(0.05, 0.94, txt, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.7))
    ax.legend(loc="lower right", fontsize=8)
    _label_panel(ax, "B")

    ax = axes[2]
    rp   = bjh[:, 0] * 2
    dVdr = bjh[:, 1]
    ax.plot(rp, dVdr, "-", color=C_BJH, lw=1.5)
    ax.fill_between(rp, dVdr, alpha=0.15, color=C_BJH)
    peak_idx = np.argmax(dVdr)
    ax.axvline(rp[peak_idx], ls="--", lw=0.9, color=C_BJH, alpha=0.7)
    ax.text(rp[peak_idx] + 0.5, dVdr[peak_idx] * 0.95,
            f"{rp[peak_idx]:.1f} nm", fontsize=8, color=C_BJH)
    ax.set_xlabel(r"Pore Diameter (nm)")
    ax.set_ylabel(r"d$V_p$/d$r_p$  (cm$^3$ g$^{-1}$ nm$^{-1}$)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.axvline(N2_CAVITATION_NM, ls=":", lw=0.8, color="0.6", alpha=0.7)
    _label_panel(ax, "C")

    ax  = axes[3]
    ax2 = ax.twinx()
    cum_Vp  = bjh[:, 2]
    cum_Sap = bjh[:, 3]
    ax.plot(rp, cum_Vp,  "-",  color=C_CUM, lw=1.5, label=r"$V_p$ cumulative")
    ax2.plot(rp, cum_Sap, "--", color=C_BJH, lw=1.5, label=r"$S_{ap}$ cumulative")
    ax2.axhline(s["S_BET"], ls=":", lw=1.0, color=C_BET,
                label=f"$S_{{BET}}$ = {s['S_BET']:.1f} m\u00b2 g\u207b\u00b9")
    ax2.axhline(s["S_BJH"], ls=":", lw=1.0, color=C_BJH,
                label=f"$S_{{BJH}}$ = {s['S_BJH']:.1f} m\u00b2 g\u207b\u00b9")
    ax.set_xlabel(r"Pore Diameter (nm)")
    ax.set_ylabel(r"Cum. Pore Volume (cm$^3$ g$^{-1}$)", color=C_CUM)
    ax2.set_ylabel(r"Cum. Surface Area (m$^2$ g$^{-1}$)", color=C_BJH)
    ax.tick_params(axis="y", colors=C_CUM)
    ax2.tick_params(axis="y", colors=C_BJH)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7.5, loc="lower right")
    _label_panel(ax, "D")

    fig.suptitle(f"BET/BJH Analysis \u2014 {sample_name}",
                 fontsize=12, y=1.01, fontweight="bold")
    if save:
        out = f"{sample_name.replace(' ', '_')}_BET_analysis.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\n  Figure saved \u2192 {out}")
    plt.tight_layout()
    plt.show()


def _label_panel(ax, letter):
    ax.text(-0.13, 1.03, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# ══════════════════════════════════════════════════════════
# 6. SUMMARY REPORT
# ══════════════════════════════════════════════════════════

def print_report(data: dict, iso_cls: dict, hyst_cls: dict,
                 bet_res: dict, sample_name: str):
    s = data["summary"]; h = hyst_cls
    sep = "=" * 60
    print(f"\n{sep}\n  BET Analysis Report \u2014 {sample_name}\n{sep}")
    rows = [
        ["BET Surface Area",     f"{s['S_BET']:.3f}",         "m\u00b2 g\u207b\u00b9"],
        ["Vm (monolayer cap.)",   f"{s['Vm']:.4f}",            "cm\u00b3(STP) g\u207b\u00b9"],
        ["BET C constant",        f"{s['C']:.2f}",             "\u2014"],
        ["Total Pore Volume",     f"{s['Vp_total']:.4f}",      "cm\u00b3 g\u207b\u00b9"],
        ["Average Pore Diameter", f"{s['dp_avg']:.3f}",        "nm"],
        ["BJH Surface Area",      f"{s['S_BJH']:.3f}",         "m\u00b2 g\u207b\u00b9"],
        ["BJH Peak Diameter",     f"{s['rp_peak_BJH']*2:.2f}", "nm"],
    ]
    print(tabulate(rows, headers=["Parameter", "Value", "Unit"], tablefmt="rounded_outline"))
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BET/BJH Analysis Tool")
    parser.add_argument("--file",   required=True, help="Path to BET instrument XLS/XLSX file")
    parser.add_argument("--sample", default="Sample", help="Sample name")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    data     = read_bet_xls(args.file)
    iso_cls  = classify_isotherm(data["ads"], data["des"])
    hyst_cls = classify_hysteresis(data["ads"], data["des"])
    bet_res  = verify_bet(data["bet_pts"], data["summary"])
    print_report(data, iso_cls, hyst_cls, bet_res, args.sample)
    plot_all(data, iso_cls, hyst_cls, bet_res, args.sample, save=not args.no_show)


if __name__ == "__main__":
    main()
