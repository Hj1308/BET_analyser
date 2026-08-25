"""
tplot_analysis.py — T-Plot Analysis (Harkins-Jura)
====================================================
Determines:
  - Micropore volume  (V_micro, cm³/g)
  - External surface area  (S_ext, m²/g)  — meso + macro
  - Pore type distribution: Micropore / Mesopore / Macropore (%)

Harkins-Jura statistical film thickness:
    t = sqrt(13.99 / (0.034 - log10(P/P0)))   [Angstrom]

In the t-plot:
  • Linear through origin  → no micropores, all surface is external
  • Positive intercept     → micropore volume present
  • Slope ∝ external (mesopore) surface area

Usage:
    Standalone:
        python tplot_analysis.py --file C3N4.xls --sample "C3N4" --s-bet 95.3 --vtot 0.38

    As module alongside bet_analysis.py:
        from tplot_analysis import TPlotAnalyser
        tp = TPlotAnalyser(pressure, volume_adsorbed, s_bet=95.3, total_pore_volume=0.38)
        report = tp.full_tplot_report(t_min=3.5, t_max=5.0)
        tp.plot_tplot(save_path="tplot.png", t_min=3.5, t_max=5.0)

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


# ══════════════════════════════════════════════════════════════
# HARKINS-JURA THICKNESS EQUATION
# ══════════════════════════════════════════════════════════════

def harkins_jura_t(p_rel: np.ndarray) -> np.ndarray:
    """
    Statistical film thickness by Harkins-Jura equation.

    t (Å) = sqrt(13.99 / (0.034 - log10(P/P0)))

    Valid range: 0.08 < P/P0 < 0.60 (outside this, other equations preferred)

    Parameters
    ----------
    p_rel : array — relative pressure P/P0  (0 < p_rel < 1)

    Returns
    -------
    np.ndarray — film thickness in Angstrom (Å)
    """
    p_rel = np.clip(p_rel, 1e-9, 1 - 1e-9)
    return np.sqrt(13.99 / (0.034 - np.log10(p_rel)))


# ══════════════════════════════════════════════════════════════
# T-PLOT ANALYSER CLASS
# ══════════════════════════════════════════════════════════════

class TPlotAnalyser:
    """
    T-Plot analysis from N₂ physisorption data.

    Parameters
    ----------
    pressure            : array-like — relative pressure (P/P0)
    volume_adsorbed     : array-like — volume adsorbed (cm³/g STP)
    s_bet               : float      — BET surface area (m²/g)
    total_pore_volume   : float      — total pore volume at P/P0 ≈ 0.99 (cm³/g)
    """

    def __init__(self, pressure, volume_adsorbed, s_bet: float, total_pore_volume: float):
        self.p    = np.array(pressure,        dtype=float)
        self.v    = np.array(volume_adsorbed, dtype=float)
        self.sbet = s_bet
        self.vtot = total_pore_volume
        self.t    = harkins_jura_t(self.p)

    # ──────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────

    def fit_tplot(self, t_min: float = 3.5, t_max: float = 5.0) -> dict:
        """
        Fit the linear region of the t-plot.

        Default range 3.5–5.0 Å is the standard IUPAC/BET linear region
        (p/p₀ ≈ 0.08–0.30). If fewer than 3 points fall in range, the
        range is auto-expanded; if it still has fewer than 3 points a
        ValueError is raised (a 2-point fit has R² = 1.0 by construction).

        Conversion factors:
            S_ext (m²/g)    = slope × N2_TPLOT_SLOPE_FACTOR
            V_micro (cm³/g) = intercept × N2_STP_TO_LIQUID
              (Gurvich rule: V_liquid = V_STP × (M/V_molar)/rho ≈ 1.5468e-3)

        Returns
        -------
        dict with:
            S_ext_m2g     — external (mesopore) surface area  (m²/g)
            V_micro_cm3g  — micropore volume  (cm³/g)
            R2_tplot      — R² of linear fit
            slope         — raw regression slope
            intercept     — raw regression intercept
            t_range       — (t_min, t_max) actually used
            n_points      — number of points fitted
            low_confidence— True when exactly 3 points were fitted

        Raises
        ------
        ValueError
            If fewer than 3 points are available even after auto-expansion.
        """
        mask = (self.t >= t_min) & (self.t <= t_max)
        if mask.sum() < 3:
            t_min = float(self.t.min()) + 0.2
            t_max = float(self.t.max()) - 0.2
            mask  = (self.t >= t_min) & (self.t <= t_max)

        n_points = int(mask.sum())
        if n_points < 3:
            raise ValueError(
                f"t-plot fit needs at least 3 points in window "
                f"({t_min:.2f}–{t_max:.2f} Å) but only {n_points} are available."
            )

        slope, intercept, r, *_ = linregress(self.t[mask], self.v[mask])
        s_ext   = slope * N2_TPLOT_SLOPE_FACTOR
        v_micro_raw = intercept * N2_STP_TO_LIQUID
        v_micro = max(v_micro_raw, 0.0)
        clamped = v_micro_raw < 0.0
        if clamped:
            warnings.warn(
                f"t-plot intercept is negative ({intercept:.4f} cm³/g STP); "
                "V_micro was clamped to 0. A negative intercept usually means "
                "the reference t-curve does not match the sample's surface "
                "chemistry, or the fitted window spans the micropore-filling "
                "region.",
                UserWarning,
                stacklevel=2,
            )

        return {
            "S_ext_m2g"     : round(s_ext,   2),
            "V_micro_cm3g"  : round(v_micro, 5),
            "R2_tplot"      : round(r ** 2,  5),
            "slope"         : round(slope,   5),
            "intercept"     : round(intercept, 5),
            "intercept_raw" : round(intercept, 5),
            "t_range"       : (round(t_min, 2), round(t_max, 2)),
            "n_points"      : n_points,
            "low_confidence": n_points == 3,
            "clamped"       : clamped,
        }

    # ──────────────────────────────────────────────────────────
    # PORE DISTRIBUTION
    # ──────────────────────────────────────────────────────────

    def pore_distribution(self, v_micro: float) -> dict:
        """
        Calculate pore volume fractions.

        V_meso  = V_total - V_micro
        V_macro = 0 by default
                  (accurate V_macro requires Hg porosimetry)

        Returns
        -------
        dict with volumes (cm³/g) and % for micropore, mesopore, macropore
        """
        v_meso  = max(self.vtot - v_micro, 0.0)
        v_macro = 0.0
        total   = v_micro + v_meso + v_macro
        if total <= 0:
            return {"error": "Total pore volume is zero or negative."}
        return {
            "V_micro_cm3g" : round(v_micro,       5),
            "V_meso_cm3g"  : round(v_meso,        5),
            "V_macro_cm3g" : round(v_macro,       5),
            "V_total_cm3g" : round(total,         5),
            "Micropore_%"  : round(100 * v_micro / total, 1),
            "Mesopore_%"   : round(100 * v_meso  / total, 1),
            "Macropore_%"  : round(100 * v_macro / total, 1),
        }

    # ──────────────────────────────────────────────────────────
    # MICROPORE SURFACE AREA
    # ──────────────────────────────────────────────────────────

    def micropore_surface_area(self, s_ext: float) -> dict:
        """
        Micropore surface area from t-plot:
            S_micro = S_BET - S_ext

        Returns
        -------
        dict: S_micro_m2g, S_ext_m2g, S_BET_m2g, s_micro_raw, clamped
        """
        s_micro_raw = self.sbet - s_ext
        s_micro = max(s_micro_raw, 0.0)
        clamped = s_micro_raw < 0.0
        if clamped:
            warnings.warn(
                f"t-plot external surface area ({s_ext:.2f} m²/g) exceeds S_BET "
                f"({self.sbet:.2f} m²/g); S_micro was clamped to 0. This usually "
                "means the reference t-curve does not match the sample's surface "
                "chemistry, or the fitted window spans the micropore-filling region.",
                UserWarning,
                stacklevel=2,
            )
        return {
            "S_BET_m2g"   : round(self.sbet, 2),
            "S_ext_m2g"   : round(s_ext,     2),
            "S_micro_m2g" : round(s_micro,   2),
            "s_micro_raw" : round(s_micro_raw, 2),
            "clamped"     : clamped,
        }

    # ──────────────────────────────────────────────────────────
    # FULL REPORT
    # ──────────────────────────────────────────────────────────

    def full_tplot_report(self, t_min: float = 3.5, t_max: float = 5.0) -> dict:
        """Run fit, pore distribution, and surface area — all together."""
        fit  = self.fit_tplot(t_min, t_max)
        dist = self.pore_distribution(fit["V_micro_cm3g"])
        sa   = self.micropore_surface_area(fit["S_ext_m2g"])
        result = {**fit, **dist, **sa}
        result["clamped"] = bool(fit.get("clamped")) or bool(sa.get("clamped"))
        return result

    # ──────────────────────────────────────────────────────────
    # PRINT REPORT
    # ──────────────────────────────────────────────────────────

    def print_report(self, sample_name: str = "Sample",
                     t_min: float = 3.5, t_max: float = 5.0):
        res = self.full_tplot_report(t_min, t_max)
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  T-Plot Report (Harkins-Jura) — {sample_name}")
        print(sep)
        print(f"  Fit range      : {res['t_range'][0]}–{res['t_range'][1]} Å  ({res['n_points']} pts)")
        print(f"  R²             : {res['R2_tplot']}")
        print(f"")
        print(f"  Surface Area")
        print(f"    S_BET        : {res['S_BET_m2g']:.2f}  m² g⁻¹")
        print(f"    S_ext        : {res['S_ext_m2g']:.2f}  m² g⁻¹  (meso + macro)")
        print(f"    S_micro      : {res['S_micro_m2g']:.2f}  m² g⁻¹")
        print(f"")
        print(f"  Pore Volumes")
        print(f"    V_total      : {res['V_total_cm3g']:.5f}  cm³ g⁻¹")
        print(f"    V_micro      : {res['V_micro_cm3g']:.5f}  cm³ g⁻¹   ({res['Micropore_%']}%)")
        print(f"    V_meso       : {res['V_meso_cm3g']:.5f}  cm³ g⁻¹   ({res['Mesopore_%']}%)")
        print(f"    V_macro      : {res['V_macro_cm3g']:.5f}  cm³ g⁻¹   ({res['Macropore_%']}%)")
        print(f"      ↳ Note: V_macro from Hg porosimetry for accuracy")
        print(f"{sep}\n")

    # ──────────────────────────────────────────────────────────
    # PLOT
    # ──────────────────────────────────────────────────────────

    def plot_tplot(self, save_path: str = "tplot.png", sample_name: str = "Sample",
                   t_min: float = 3.5, t_max: float = 5.0) -> str:
        """
        2-panel T-Plot figure:
          [A] t-plot with linear fit + region highlight
          [B] Pore type distribution bar chart
        """
        res  = self.full_tplot_report(t_min, t_max)
        t_lo, t_hi = res["t_range"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # ── [A] t-plot ─────────────────────────────────────────
        ax = axes[0]
        ax.scatter(self.t, self.v, color=C_MICRO, s=35, zorder=5,
                   label="Experimental data")

        # Highlight fit region
        mask = (self.t >= t_lo) & (self.t <= t_hi)
        ax.scatter(self.t[mask], self.v[mask], color=C_FIT, s=50,
                   zorder=6, label=f"Fitted ({t_lo}–{t_hi} Å)")

        # Regression line extended across full range
        t_line  = np.linspace(self.t.min() * 0.95, self.t.max() * 1.02, 300)
        v_line  = res["slope"] * t_line + res["intercept"]
        ax.plot(t_line, v_line, "-", color=C_FIT, lw=1.8,
                label=f"Linear fit  R²={res['R2_tplot']}")

        # Mark intercept (micropore volume indicator)
        if abs(res["intercept"]) > 0.1:
            ax.axhline(res["intercept"], ls=":", lw=0.9, color=C_EXT, alpha=0.7)
            ax.text(self.t.min(), res["intercept"] + self.v.max()*0.02,
                    f"intercept={res['intercept']:.2f}\n→ V_micro={res['V_micro_cm3g']:.4f} cm³/g",
                    fontsize=7.5, color=C_EXT)

        ax.set_xlabel("Statistical film thickness  t (Å)", fontsize=11)
        ax.set_ylabel("Volume adsorbed  (cm³ g⁻¹ STP)",   fontsize=11)
        ax.set_title("T-Plot (Harkins-Jura)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(False)
        ax.set_xlim(self.t.min() * 0.92, self.t.max() * 1.05)

        # Annotation box
        ann = (f"$S_{{ext}}$ = {res['S_ext_m2g']:.1f} m² g⁻¹\n"
               f"$S_{{micro}}$ = {res['S_micro_m2g']:.1f} m² g⁻¹\n"
               f"$V_{{micro}}$ = {res['V_micro_cm3g']:.4f} cm³ g⁻¹")
        ax.text(0.97, 0.05, ann, transform=ax.transAxes,
                va="bottom", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.7))

        # ── [B] Pore distribution bar ──────────────────────────
        ax2   = axes[1]
        labels = ["Micropore", "Mesopore", "Macropore"]
        values = [res["Micropore_%"], res["Mesopore_%"], res["Macropore_%"]]
        colors = [C_MICRO, C_EXT, "#888888"]
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
        description="T-Plot Analysis (Harkins-Jura) from raw P/P0 + V data")
    parser.add_argument("--s-bet",  type=float, required=True,
                        help="BET surface area (m²/g)")
    parser.add_argument("--vtot",   type=float, required=True,
                        help="Total pore volume at P/P0=0.99 (cm³/g)")
    parser.add_argument("--sample", default="Sample",
                        help="Sample name for plot title")
    parser.add_argument("--t-min",  type=float, default=3.5)
    parser.add_argument("--t-max",  type=float, default=5.0)
    args = parser.parse_args()

    print("\n  ⚠  Standalone mode: using built-in demo data.")
    print("     For real data, import TPlotAnalyser from bet_analysis workflow.\n")

    # Demo data — typical mesoporous silica
    p = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.50,0.70,0.90,0.99])
    v = np.array([85., 102.,115.,126.,135.,143.,150.,172.,200.,280.,520.])

    tp = TPlotAnalyser(p, v, s_bet=args.s_bet, total_pore_volume=args.vtot)
    tp.print_report(sample_name=args.sample, t_min=args.t_min, t_max=args.t_max)
    tp.plot_tplot(save_path=f"{args.sample}_tplot.png", sample_name=args.sample,
                  t_min=args.t_min, t_max=args.t_max)
    print(f"  Plot saved → {args.sample}_tplot.png")


if __name__ == "__main__":
    main()
