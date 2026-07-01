"""
BET Analysis Tool — Publication-Quality Figures
================================================
Reads instrument XLS/XLSX output and computes:
  - Isotherm type classification  (IUPAC Type I–VI, including I(a)/I(b))
  - Hysteresis type classification (IUPAC H1–H4)
  - BET plot with regression verification
  - BJH differential pore size distribution
  - Cumulative pore volume
  - BET vs BJH surface area comparison

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


N2_BET_FACTOR         = 4.353
N2_TPLOT_SLOPE_FACTOR = 15.47
N2_LIQUID_FACTOR      = 1547.0
N2_CAVITATION_NM      = 3.4


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.framealpha": 0.9, "legend.edgecolor": "0.7",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "lines.linewidth": 1.5, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.grid": False,
    })

C_ADS = "#2166AC"; C_DES = "#D6604D"; C_BET = "#1A7A4A"
C_BJH = "#7B3F9E"; C_CUM = "#C07028"; C_SHADE = "#AACCE8"


# ══════════════════════════════════════════════════════════
# 1. DATA READING
# ══════════════════════════════════════════════════════════

def _load_sheets(filepath: str) -> tuple:
    """
    Load all sheets from XLS or XLSX.
    - .xls  : uses xlrd API directly (bypasses pandas engine check)
    - .xlsx : uses openpyxl via pandas (standard path)
    Returns (sheet_names, raw_dict)
    """
    if str(filepath).lower().endswith(".xls"):
        from xls_reader import read_xls_sheets
        return read_xls_sheets(filepath)
    else:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
        raw = {sh: pd.read_excel(filepath, sheet_name=sh,
                                 engine="openpyxl", header=None)
               for sh in xl.sheet_names}
        return xl.sheet_names, raw


def read_bet_xls(filepath: str) -> dict:
    """
    Parse the XLS/XLSX file produced by the BET instrument.
    Returns a dict with keys: ads, des, bet_pts, bjh, summary
    """
    sheet_names, raw = _load_sheets(filepath)

    required_sheets = {"AdsDes", "BET", "BJH", "Summary"}
    missing = required_sheets - set(sheet_names)
    if missing:
        raise ValueError(
            f"Missing required sheet(s): {missing}. Found: {sheet_names}")

    # ── Adsorption / Desorption ───────────────────────────
    df = raw["AdsDes"]
    ads_rows = df[df.iloc[:, 0] == "ADS"].index
    if len(ads_rows) == 0:
        raise ValueError("Label 'ADS' not found in sheet 'AdsDes'.")
    des_rows = df[df.iloc[:, 0] == "DES"].index
    if len(des_rows) == 0:
        raise ValueError("Label 'DES' not found in sheet 'AdsDes'.")

    def _extract(start, end_label):
        rows = []
        for i in range(start, len(df)):
            if end_label is not None and df.iloc[i, 0] == end_label:
                break
            try:
                rows.append((float(df.iloc[i, 5]), float(df.iloc[i, 6])))
            except (ValueError, TypeError):
                break
        return np.array(rows)

    ads = _extract(ads_rows[0] + 1, "DES")
    des = _extract(des_rows[0] + 1, None)
    if len(ads) == 0:
        raise ValueError("No adsorption data points found in sheet 'AdsDes'.")

    # ── BET ─────────────────────────────────────────────
    df_b = raw["BET"]
    no_rows_b = df_b[df_b.iloc[:, 0] == "No"].index
    if len(no_rows_b) == 0:
        raise ValueError("Label 'No' not found in BET sheet.")
    bet_pts = []
    for i in range(no_rows_b[0] + 1, len(df_b)):
        try:
            bet_pts.append((float(df_b.iloc[i, 1]), float(df_b.iloc[i, 2])))
        except (ValueError, TypeError):
            break
    bet_pts = np.array(bet_pts)

    sp = df_b[df_b.iloc[:, 0] == "Starting point"]
    ep = df_b[df_b.iloc[:, 0] == "End point"]
    if sp.empty or ep.empty:
        raise ValueError("'Starting point'/'End point' not found in BET sheet.")
    start_pt = int(sp.iloc[0, 3])
    end_pt   = int(ep.iloc[0, 3])

    # ── BJH ─────────────────────────────────────────────
    df_j = raw["BJH"]
    no_rows_j = df_j[df_j.iloc[:, 0] == "No"].index
    if len(no_rows_j) == 0:
        raise ValueError("Label 'No' not found in BJH sheet.")
    bjh_rows = []
    for i in range(no_rows_j[0] + 1, len(df_j)):
        try:
            bjh_rows.append((
                float(df_j.iloc[i, 2]), float(df_j.iloc[i, 3]),
                float(df_j.iloc[i, 4]), float(df_j.iloc[i, 5])
            ))
        except (ValueError, TypeError):
            break
    bjh = np.array(bjh_rows)

    # ── Summary ───────────────────────────────────────
    df_s = raw["Summary"]
    def _get(label):
        mask = df_s.iloc[:, 0] == label
        if not mask.any(): return np.nan
        row = df_s.loc[mask].iloc[0]
        for col in [3, 2]:
            try:
                v = float(row.iloc[col])
                if not np.isnan(v): return v
            except (ValueError, TypeError): pass
        return np.nan

    summary = {
        "Vm": _get("Vm"), "S_BET": _get("as,BET"), "C": _get("C"),
        "Vp_total": _get("Total pore volume(p/p0=0.990)"),
        "dp_avg": _get("Average pore diameter"),
        "rp_peak_BJH": _get("rp,peak(Area)"),
        "S_BJH": _get("ap"), "Vp_BJH": _get("Vp"),
        "start_pt": start_pt, "end_pt": end_pt,
    }
    return dict(ads=ads, des=des, bet_pts=bet_pts, bjh=bjh, summary=summary)


# ══════════════════════════════════════════════════════════
# 2. ISOTHERM CLASSIFICATION
# ══════════════════════════════════════════════════════════

def classify_isotherm(ads, des):
    pp0_a, Va_a = ads[:, 0], ads[:, 1]
    has_hyst = len(des) > 0

    low_mask = pp0_a < 0.35
    if low_mask.sum() > 2:
        x_l, y_l = pp0_a[low_mask], Va_a[low_mask]
        d2 = np.gradient(np.gradient(y_l, x_l), x_l)
        concave_low = float(d2.mean()) < 0
    else:
        concave_low = True

    high_mask = pp0_a > 0.75
    has_plateau = False
    if high_mask.sum() > 2:
        Va_hi = Va_a[high_mask]
        has_plateau = (Va_hi.max() - Va_hi.min()) / Va_hi.mean() < 0.25

    very_low_mask = pp0_a < 0.1
    steep_init = False
    if very_low_mask.sum() > 1:
        sl = ((Va_a[very_low_mask][-1] - Va_a[very_low_mask][0]) /
              (pp0_a[very_low_mask][-1] - pp0_a[very_low_mask][0] + 1e-9))
        steep_init = sl > 100

    dVa = np.diff(Va_a)
    pp0_mid = 0.5 * (pp0_a[:-1] + pp0_a[1:])
    is_stepped = len(np.where((dVa > dVa.mean() + 2*dVa.std()) &
                               (pp0_mid > 0.1) & (pp0_mid < 0.9))[0]) >= 2

    is_type_Ia = False
    ultra_low = pp0_a < 0.01
    if ultra_low.sum() > 1 and steep_init:
        is_type_Ia = Va_a[ultra_low].max() / (Va_a.max() + 1e-9) > 0.5

    if is_stepped and not has_hyst:
        t, ex = "Type VI", "Stepped isotherm. Multilayer adsorption on uniform non-porous surface."
    elif has_hyst:
        if concave_low:
            t, ex = "Type IV", "Hysteresis + concave at low p/p\u2080. Mesoporous material."
        else:
            t, ex = "Type V", "Hysteresis + convex at low p/p\u2080. Weak adsorbate-adsorbent interactions."
    else:
        if steep_init and has_plateau:
            if is_type_Ia:
                t, ex = "Type I(a)", "Ultra-micropores < 1 nm."
            else:
                t, ex = "Type I(b)", "Micropores 1\u20132.5 nm."
        elif concave_low and has_plateau:
            t, ex = "Type II", "Non-porous or macroporous material."
        else:
            t, ex = "Type III", "Weak adsorbate\u2013adsorbent interactions."

    return {"type": t, "explanation": ex, "has_hysteresis": has_hyst,
            "concave_low": concave_low, "has_plateau": has_plateau}


# ══════════════════════════════════════════════════════════
# 3. HYSTERESIS CLASSIFICATION
# ══════════════════════════════════════════════════════════

def classify_hysteresis(ads, des):
    if len(des) == 0:
        return {"type": "None", "explanation": "No hysteresis.",
                "scores": {}, "features": {}, "confidence": "n/a", "confidence_pct": 0}

    pp0_a, Va_a = ads[:, 0], ads[:, 1]
    sort_d = np.argsort(des[:, 0])
    pp0_d, Va_d = des[sort_d, 0], des[sort_d, 1]

    p_lo = max(pp0_a.min(), pp0_d.min())
    p_hi = min(pp0_a.max(), pp0_d.max())
    p_grid = np.linspace(p_lo, p_hi, 200)
    f_ads = interp1d(pp0_a, Va_a, bounds_error=False, fill_value="extrapolate")
    f_des = interp1d(pp0_d, Va_d, bounds_error=False, fill_value="extrapolate")
    Va_a_g = f_ads(p_grid); Va_d_g = f_des(p_grid)
    hyst = np.clip(Va_d_g - Va_a_g, 0, None)

    norm_area = float(np.trapz(hyst, p_grid)) / (Va_a.max() + 1e-9)
    ads_sl = np.abs(np.gradient(Va_a_g, p_grid))
    des_sl = np.abs(np.gradient(Va_d_g, p_grid))
    mid = (p_grid > 0.3) & (p_grid < 0.95)
    ratio_max  = float(des_sl[mid].max() / (ads_sl[mid].max() + 1e-9))
    ratio_mean = float(des_sl[mid].mean() / (ads_sl[mid].mean() + 1e-9))
    peak_pos = p_grid[np.argmax(hyst)]
    is_left_skewed = peak_pos < 0.65

    high_ads = pp0_a > 0.75
    has_plateau = False
    if high_ads.sum() > 2:
        Va_hi = Va_a[high_ads]
        has_plateau = (Va_hi.max() - Va_hi.min()) / (Va_hi.mean() + 1e-9) < 0.35

    low_ads = pp0_a < 0.5
    is_flat_low = False
    if low_ads.sum() > 3:
        fl = ((Va_a[low_ads][-1] - Va_a[low_ads][0]) /
              (pp0_a[low_ads][-1] - pp0_a[low_ads][0] + 1e-9))
        is_flat_low = fl < 30

    hyst_open = hyst > hyst.max() * 0.05
    closure_p = float(p_grid[hyst_open][0]) if hyst_open.any() else p_lo

    sc = {"H1": 0, "H2": 0, "H3": 0, "H4": 0}
    if ratio_mean < 2.0 and norm_area < 0.15 and has_plateau: sc["H1"] += 3
    if ratio_max < 2.5: sc["H1"] += 1
    if ratio_max > 2.0: sc["H2"] += 3
    if is_left_skewed:  sc["H2"] += 2
    if has_plateau:     sc["H2"] += 1
    if not has_plateau: sc["H3"] += 3
    if norm_area > 0.20: sc["H3"] += 2
    if not is_left_skewed: sc["H3"] += 1
    if is_flat_low and norm_area < 0.12: sc["H4"] += 3
    if is_flat_low: sc["H4"] += 2
    if not has_plateau and norm_area < 0.18: sc["H4"] += 1

    best = max(sc, key=sc.get)
    conf = sc[best] / (sum(sc.values()) + 1e-9)
    expl = {
        "H1": "Narrow symmetric loop. Uniform cylindrical mesopores.",
        "H2": "Triangular loop, steep desorption. Ink-bottle / pore-blocking.",
        "H3": "No plateau near p/p\u2080\u21921. Plate-like aggregates / slit pores.",
        "H4": "Narrow, flat loop. Microporous + narrow slit pores.",
    }
    return {
        "type": best, "explanation": expl[best],
        "confidence": "high" if conf > 0.55 else "moderate" if conf > 0.40 else "low",
        "confidence_pct": round(conf * 100, 1), "scores": sc,
        "features": {
            "hysteresis_area_norm": round(norm_area, 4),
            "slope_ratio_max": round(ratio_max, 3),
            "slope_ratio_mean": round(ratio_mean, 3),
            "peak_position_p/p0": round(float(peak_pos), 3),
            "has_plateau_ads": has_plateau, "flat_at_low_pp0": is_flat_low,
            "closure_point_p/p0": round(closure_p, 3),
            "forced_closure_N2": closure_p < 0.45,
        },
    }


# ══════════════════════════════════════════════════════════
# 4. BET VERIFICATION
# ══════════════════════════════════════════════════════════

def verify_bet(bet_pts, summary):
    pts = bet_pts[summary["start_pt"]: summary["end_pt"] + 1]
    x, y = pts[:, 0], pts[:, 1]
    slope, intercept, r, *_ = linregress(x, y)
    R2 = r ** 2; Vm = 1.0 / (slope + intercept)
    C = 1.0 + slope / intercept
    if C < 0:
        warnings.warn(f"BET C < 0 (C={C:.2f}). Range outside valid BET region.",
                      UserWarning, stacklevel=2)
    if not np.all(np.diff(y) > 0):
        warnings.warn("BET y not monotonically increasing.", UserWarning, stacklevel=2)
    return dict(x=x, y=y, slope=slope, intercept=intercept, R2=R2, Vm=Vm, C=C,
                S_BET_calc=Vm * N2_BET_FACTOR,
                S_BET_instrument=summary["S_BET"],
                C_valid=(C > 0), all_pts=bet_pts)


# ══════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════

def plot_all(data, iso_cls, hyst_cls, bet_res, sample_name, save=True):
    setup_plot_style()
    ads = data["ads"]; des = data["des"]
    bjh = data["bjh"]; s   = data["summary"]
    fig = plt.figure(figsize=(7.2, 6.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]

    # A — Isotherm
    ax = axes[0]
    ax.plot(ads[:,0], ads[:,1], "o-", color=C_ADS, ms=4, lw=1.4, label="Adsorption")
    if len(des):
        sd = np.argsort(des[:,0])[::-1]
        ax.plot(des[sd,0], des[sd,1], "s--", color=C_DES, ms=4, lw=1.4, label="Desorption")
        ax.fill(np.concatenate([ads[:,0], des[sd,0][::-1]]),
                np.concatenate([ads[:,1], des[sd,1][::-1]]), alpha=0.10, color=C_ADS)
    ax.set_xlabel(r"Relative Pressure ($p/p_0$)")
    ax.set_ylabel(r"Volume Adsorbed (cm$^3$ g$^{-1}$ STP)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    hl = hyst_cls["type"]
    ax.text(0.03, 0.96, iso_cls["type"] + (f" / {hl}" if hl != "None" else ""),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.7))
    _label_panel(ax, "A")

    # B — BET
    ax = axes[1]
    ax.scatter(bet_res["all_pts"][:,0], bet_res["all_pts"][:,1],
               color="0.75", s=20, zorder=2, label="Unused")
    ax.scatter(bet_res["x"], bet_res["y"], color=C_BET, s=30, zorder=4, label="Fitted")
    xf = np.linspace(bet_res["x"].min(), bet_res["x"].max(), 200)
    ax.plot(xf, bet_res["slope"]*xf + bet_res["intercept"], "-", color=C_BET, lw=1.6)
    ax.set_xlabel(r"$p/p_0$")
    ax.set_ylabel(r"$\frac{1}{V_\mathrm{a}(p_0/p-1)}$ (g cm$^{-3}$)", labelpad=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    cf = "" if bet_res["C_valid"] else "  \u26a0 C<0"
    ax.text(0.05, 0.94,
            f"$S_{{BET}}$={s['S_BET']:.2f} m\u00b2/g\n$C$={s['C']:.1f}{cf}\n$R^2$={bet_res['R2']:.5f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.7))
    ax.legend(loc="lower right", fontsize=8)
    _label_panel(ax, "B")

    # C — BJH PSD
    ax = axes[2]
    rp = bjh[:,0]*2; dVdr = bjh[:,1]
    ax.plot(rp, dVdr, "-", color=C_BJH, lw=1.5)
    ax.fill_between(rp, dVdr, alpha=0.15, color=C_BJH)
    pk = np.argmax(dVdr)
    ax.axvline(rp[pk], ls="--", lw=0.9, color=C_BJH, alpha=0.7)
    ax.text(rp[pk]+0.5, dVdr[pk]*0.95, f"{rp[pk]:.1f} nm", fontsize=8, color=C_BJH)
    ax.axvline(N2_CAVITATION_NM, ls=":", lw=0.8, color="0.6", alpha=0.7)
    ax.set_xlabel("Pore Diameter (nm)"); ax.set_ylabel(r"d$V_p$/d$r_p$ (cm$^3$ g$^{-1}$ nm$^{-1}$)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    _label_panel(ax, "C")

    # D — Cumulative
    ax = axes[3]; ax2 = ax.twinx()
    ax.plot(rp, bjh[:,2], "-", color=C_CUM, lw=1.5, label=r"$V_p$ cum")
    ax2.plot(rp, bjh[:,3], "--", color=C_BJH, lw=1.5, label=r"$S_{ap}$ cum")
    ax2.axhline(s["S_BET"], ls=":", lw=1.0, color=C_BET,
                label=f"$S_{{BET}}$={s['S_BET']:.1f}")
    ax2.axhline(s["S_BJH"], ls=":", lw=1.0, color=C_BJH,
                label=f"$S_{{BJH}}$={s['S_BJH']:.1f}")
    ax.set_xlabel("Pore Diameter (nm)")
    ax.set_ylabel(r"Cum. Pore Vol. (cm$^3$ g$^{-1}$)", color=C_CUM)
    ax2.set_ylabel(r"Cum. Surface Area (m$^2$ g$^{-1}$)", color=C_BJH)
    ax.tick_params(axis="y", colors=C_CUM); ax2.tick_params(axis="y", colors=C_BJH)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    l1, b1 = ax.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, b1+b2, fontsize=7.5, loc="lower right")
    _label_panel(ax, "D")

    fig.suptitle(f"BET/BJH Analysis \u2014 {sample_name}",
                 fontsize=12, y=1.01, fontweight="bold")
    if save:
        out = f"{sample_name.replace(' ','_')}_BET_analysis.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Figure saved \u2192 {out}")
    plt.tight_layout(); plt.show()


def _label_panel(ax, letter):
    ax.text(-0.13, 1.03, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# ══════════════════════════════════════════════════════════
# 6. REPORT
# ══════════════════════════════════════════════════════════

def print_report(data, iso_cls, hyst_cls, bet_res, sample_name):
    s = data["summary"]; h = hyst_cls; sep = "="*60
    print(f"\n{sep}\n  BET Analysis \u2014 {sample_name}\n{sep}")
    rows = [
        ["BET Surface Area",    f"{s['S_BET']:.3f}",        "m\u00b2/g"],
        ["Vm",                   f"{s['Vm']:.4f}",           "cm\u00b3(STP)/g"],
        ["C",                    f"{s['C']:.2f}",            "\u2014"],
        ["Total Pore Volume",   f"{s['Vp_total']:.4f}",     "cm\u00b3/g"],
        ["Avg Pore Diameter",   f"{s['dp_avg']:.3f}",       "nm"],
        ["BJH Surface Area",    f"{s['S_BJH']:.3f}",        "m\u00b2/g"],
        ["BJH Peak Diameter",   f"{s['rp_peak_BJH']*2:.2f}","nm"],
    ]
    print(tabulate(rows, headers=["Parameter","Value","Unit"], tablefmt="rounded_outline"))
    print(f"  Isotherm: {iso_cls['type']} \u2014 {iso_cls['explanation']}")
    if h["type"] != "None":
        print(f"  Hysteresis: {h['type']} ({h['confidence']}) \u2014 {h['explanation']}")
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BET/BJH Analysis Tool")
    parser.add_argument("--file",    required=True)
    parser.add_argument("--sample",  default="Sample")
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
