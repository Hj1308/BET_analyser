"""
BET Analyser — Streamlit Web Application
=========================================
Upload your BET instrument XLS file (or use the CSV template)
and get publication-quality BET/BJH analysis instantly.

Author  : Hoda Jafari | github.com/Hj1308
License : MIT
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from bet_analysis import (
    read_bet_xls,
    classify_isotherm,
    classify_hysteresis,
    verify_bet,
    plot_all,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BET Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #6c757d; margin-bottom: 2px; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #212529; }
    .metric-unit  { font-size: 0.72rem; color: #6c757d; }
    .tag-valid   { background:#d4edda; color:#155724; border-radius:4px;
                   padding:2px 8px; font-size:0.8rem; font-weight:600; }
    .tag-warning { background:#fff3cd; color:#856404; border-radius:4px;
                   padding:2px 8px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CSV TEMPLATE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _make_csv_template() -> bytes:
    """Generate a multi-section CSV template for manual data entry."""
    lines = [
        "# BET Analyser — Manual Input Template",
        "# Instructions:",
        "#   1. Fill in each section below.",
        "#   2. Do NOT change the section headers (lines starting with []).",
        "#   3. Delete these comment lines before uploading.",
        "#   4. Save as CSV (comma-separated).",
        "",
        "[ISOTHERM]",
        "pp0_ads,Va_ads_cm3g",
        "0.050,5.10",
        "0.100,7.20",
        "0.150,8.80",
        "0.200,10.50",
        "0.250,12.10",
        "0.300,14.00",
        "0.400,17.50",
        "0.500,21.00",
        "0.600,26.00",
        "0.700,32.00",
        "0.800,42.00",
        "0.900,58.00",
        "0.950,68.00",
        "",
        "pp0_des,Va_des_cm3g",
        "0.950,68.00",
        "0.900,62.00",
        "0.800,48.00",
        "0.700,36.00",
        "0.600,28.00",
        "0.500,22.00",
        "0.400,18.00",
        "0.300,14.00",
        "",
        "[BET_POINTS]",
        "pp0,y_bet",
        "0.050,0.0095",
        "0.100,0.0132",
        "0.150,0.0168",
        "0.200,0.0205",
        "0.250,0.0241",
        "0.300,0.0278",
        "",
        "[SUMMARY]",
        "parameter,value",
        "S_BET,95.30",
        "Vm,21.90",
        "C,120.50",
        "Vp_total,0.380",
        "dp_avg,16.00",
        "S_BJH,88.00",
        "Vp_BJH,0.370",
        "rp_peak_BJH,4.00",
        "",
        "[BJH]",
        "rp_nm,dVp_drp,cum_Vp,cum_Sap",
        "1.50,0.0010,0.0010,0.50",
        "2.00,0.0050,0.0060,2.00",
        "2.50,0.0120,0.0180,4.50",
        "3.00,0.0200,0.0380,8.00",
        "4.00,0.0180,0.0560,11.00",
        "5.00,0.0100,0.0660,13.00",
        "7.00,0.0060,0.0720,14.00",
        "10.00,0.0040,0.0760,14.50",
    ]
    return "\n".join(lines).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# CSV PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_csv_template(file_bytes: bytes) -> dict:
    text = file_bytes.decode("utf-8")
    lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
    sections = {}
    current = None
    buf = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current and buf:
                sections[current] = "\n".join(buf)
            current = stripped[1:-1]
            buf = []
        elif stripped:
            buf.append(stripped)
    if current and buf:
        sections[current] = "\n".join(buf)

    required = {"ISOTHERM", "BET_POINTS", "SUMMARY", "BJH"}
    missing = required - set(sections)
    if missing:
        raise ValueError(f"Missing sections: {missing}. Download a fresh template.")

    def _read_section(key):
        return pd.read_csv(io.StringIO(sections[key]))

    df_iso = _read_section("ISOTHERM")
    ads_cols = [c for c in df_iso.columns if "ads" in c.lower()]
    des_cols = [c for c in df_iso.columns if "des" in c.lower()]
    ads = df_iso[ads_cols].dropna().values.astype(float)
    des = df_iso[des_cols].dropna().values.astype(float) if des_cols else np.array([])

    df_bet = _read_section("BET_POINTS")
    bet_pts = df_bet.values.astype(float)

    df_sum = _read_section("SUMMARY")
    df_sum.columns = ["parameter", "value"]
    s_dict = dict(zip(df_sum["parameter"].str.strip(),
                      pd.to_numeric(df_sum["value"], errors="coerce")))

    required_keys = ["S_BET", "Vm", "C", "Vp_total", "dp_avg",
                     "S_BJH", "Vp_BJH", "rp_peak_BJH"]
    missing_keys = [k for k in required_keys if k not in s_dict]
    if missing_keys:
        raise ValueError(f"Missing summary parameters: {missing_keys}.")

    summary = {
        "Vm": float(s_dict["Vm"]),
        "S_BET": float(s_dict["S_BET"]),
        "C": float(s_dict["C"]),
        "Vp_total": float(s_dict["Vp_total"]),
        "dp_avg": float(s_dict["dp_avg"]),
        "rp_peak_BJH": float(s_dict["rp_peak_BJH"]),
        "S_BJH": float(s_dict["S_BJH"]),
        "Vp_BJH": float(s_dict["Vp_BJH"]),
        "start_pt": 0,
        "end_pt": len(bet_pts) - 1,
    }

    df_bjh = _read_section("BJH")
    bjh = df_bjh.values.astype(float)
    return dict(ads=ads, des=des, bet_pts=bet_pts, bjh=bjh, summary=summary)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔬 BET Analyser")
    st.caption("Publication-quality BET/BJH + T-Plot analysis")
    st.divider()

    st.subheader("📁 Input")
    input_mode = st.radio(
        "File format",
        ["Instrument XLS", "Manual CSV"],
        help=(
            "**Instrument XLS** — direct output from Belsorp, ASAP, "
            "Quantachrome, or compatible instruments.\n\n"
            "**Manual CSV** — fill in the downloadable template below."
        ),
    )

    if input_mode == "Manual CSV":
        st.info("📥 Download the template, fill in your data, then upload it here.")
        st.download_button(
            label="⬇ Download CSV Template",
            data=_make_csv_template(),
            file_name="BET_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader(
        "Upload your file",
        type=["xls", "xlsx", "csv"],
        help="XLS/XLSX from instrument, or filled CSV template.",
    )

    st.divider()
    sample_name = st.text_input("Sample name", value="Sample")

    st.divider()
    st.subheader("⚙️ Options")
    show_tplot    = st.checkbox("Show T-Plot analysis", value=True)
    show_features = st.checkbox("Show hysteresis feature table", value=True)

    st.divider()
    st.markdown(
        "**DOI:** [10.5281/zenodo.21104234](https://doi.org/10.5281/zenodo.21104234)  \n"
        "MIT License · [GitHub](https://github.com/Hj1308/BET_analyser)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔬 BET / BJH Analyser")
st.caption("Publication-quality physisorption analysis · IUPAC 2015 compliant")

if uploaded is None:
    st.info("👈 Upload a file from the sidebar to start the analysis.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📁 Instrument XLS")
        st.markdown(
            "Direct output from **Belsorp**, ASAP, Quantachrome, "
            "or any compatible BET instrument. No reformatting needed."
        )
    with c2:
        st.markdown("### 📋 Manual CSV")
        st.markdown(
            "No instrument XLS? Download the **CSV template** "
            "from the sidebar, fill in your data, and upload it here."
        )
    with c3:
        st.markdown("### 📊 What you get")
        st.markdown(
            "- IUPAC isotherm + hysteresis classification\n"
            "- BET regression with R² and C-constant check\n"
            "- BJH differential PSD\n"
            "- Cumulative pore volume vs surface area\n"
            "- T-Plot micropore analysis\n"
            "- Downloadable 300 dpi figure + CSV report"
        )
    st.stop()


# ── Load & parse ──────────────────────────────────────────────────────────────
with st.spinner("Reading file…"):
    try:
        file_bytes = uploaded.read()
        ext = Path(uploaded.name).suffix.lower()
        if ext == ".csv":
            data = _parse_csv_template(file_bytes)
        else:
            tmp = Path(f"/tmp/{uploaded.name}")
            tmp.write_bytes(file_bytes)
            data = read_bet_xls(str(tmp))
    except Exception as e:
        st.error(f"**File read error:** {e}")
        st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
with st.spinner("Running analysis…"):
    iso_cls  = classify_isotherm(data["ads"], data["des"])
    hyst_cls = classify_hysteresis(data["ads"], data["des"])
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        bet_res = verify_bet(data["bet_pts"], data["summary"])

for w in caught_warnings:
    st.warning(f"⚠ {w.message}")

s = data["summary"]

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_bet, tab_bjh, tab_tplot, tab_download = st.tabs([
    "📊 Overview", "📈 BET", "🔵 BJH / PSD", "🔬 T-Plot", "📥 Download"
])

# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader(f"Results — {sample_name}")
    cols = st.columns(4)
    kpi = [
        (f"{s['S_BET']:.2f}",   "m² g⁻¹",  "BET Surface Area"),
        (f"{s['Vp_total']:.4f}", "cm³ g⁻¹", "Total Pore Volume"),
        (f"{s['dp_avg']:.1f}",   "nm",       "Avg Pore Diameter"),
        (f"{s['C']:.1f}",        "—",        "BET C Constant"),
    ]
    for col, (val, unit, label) in zip(cols, kpi):
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-unit">{unit}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Isotherm type**")
        st.success(f"**{iso_cls['type']}** — {iso_cls['explanation']}")
    with c2:
        if hyst_cls["type"] != "None":
            conf = hyst_cls["confidence"]
            fn = st.success if conf == "high" else st.warning if conf == "moderate" else st.error
            st.markdown("**Hysteresis type**")
            fn(
                f"**{hyst_cls['type']}** — {hyst_cls['explanation']}  \n"
                f"Confidence: {conf} ({hyst_cls['confidence_pct']:.0f}%)"
            )
        else:
            st.markdown("**Hysteresis**")
            st.info("No hysteresis detected.")

    st.divider()
    st.markdown("**Summary Table**")
    df_out = pd.DataFrame([
        ["BET Surface Area",      f"{s['S_BET']:.3f}",         "m² g⁻¹"],
        ["Vm (monolayer cap.)",    f"{s['Vm']:.4f}",            "cm³(STP) g⁻¹"],
        ["BET C constant",         f"{s['C']:.2f}",             "—"],
        ["Total Pore Volume",      f"{s['Vp_total']:.4f}",      "cm³ g⁻¹"],
        ["Average Pore Diameter",  f"{s['dp_avg']:.3f}",        "nm"],
        ["BJH Surface Area",       f"{s['S_BJH']:.3f}",         "m² g⁻¹"],
        ["BJH Peak Pore Diameter", f"{s['rp_peak_BJH']*2:.2f}", "nm"],
    ], columns=["Parameter", "Value", "Unit"])
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    tag = (
        '<span class="tag-valid">✓ C constant valid</span>'
        if bet_res["C_valid"] else
        '<span class="tag-warning">⚠ C constant negative — check p/p₀ range</span>'
    )
    st.markdown(tag, unsafe_allow_html=True)


# ── TAB 2: BET PLOT ──────────────────────────────────────────────────────────────
with tab_bet:
    st.subheader("BET Plot & Regression")
    from bet_analysis import setup_plot_style, C_BET

    col_stats, col_fig = st.columns([1, 2])
    with col_stats:
        st.markdown("**Regression Details**")
        st.table(pd.DataFrame({
            "Parameter": ["Slope", "Intercept", "R²", "Vm (calc)", "C (calc)"],
            "Value": [
                f"{bet_res['slope']:.6f}",
                f"{bet_res['intercept']:.6f}",
                f"{bet_res['R2']:.6f}",
                f"{bet_res['Vm']:.4f}",
                f"{bet_res['C']:.2f}",
            ],
        }))
        st.markdown(
            f"Points used: **{s['start_pt']}** → **{s['end_pt']}** "
            f"({s['end_pt'] - s['start_pt'] + 1} points)"
        )
    with col_fig:
        setup_plot_style()
        fig_bet, ax = plt.subplots(figsize=(5, 3.8))
        ax.scatter(bet_res["all_pts"][:, 0], bet_res["all_pts"][:, 1],
                   color="0.75", s=22, zorder=2, label="Unused")
        ax.scatter(bet_res["x"], bet_res["y"],
                   color=C_BET, s=32, zorder=4, label="Fitted")
        x_fit = np.linspace(bet_res["x"].min(), bet_res["x"].max(), 200)
        ax.plot(x_fit, bet_res["slope"] * x_fit + bet_res["intercept"],
                "-", color=C_BET, lw=1.6)
        ax.set_xlabel(r"$p/p_0$")
        ax.set_ylabel(r"$1/[V_a(p_0/p-1)]$ (g cm$^{-3}$)")
        ax.legend(fontsize=8)
        ax.text(0.05, 0.94, f"R² = {bet_res['R2']:.5f}",
                transform=ax.transAxes, va="top", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_bet, use_container_width=True)
        plt.close(fig_bet)


# ── TAB 3: BJH / PSD ──────────────────────────────────────────────────────────────
with tab_bjh:
    st.subheader("BJH Pore Size Distribution")
    from bet_analysis import setup_plot_style, C_BJH, C_CUM, N2_CAVITATION_NM

    bjh     = data["bjh"]
    rp      = bjh[:, 0] * 2
    dVdr    = bjh[:, 1]
    cum_Vp  = bjh[:, 2]
    cum_Sap = bjh[:, 3]

    setup_plot_style()
    fig_bjh, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.plot(rp, dVdr, "-", color=C_BJH, lw=1.5)
    ax1.fill_between(rp, dVdr, alpha=0.15, color=C_BJH)
    peak_idx = np.argmax(dVdr)
    ax1.axvline(rp[peak_idx], ls="--", lw=0.9, color=C_BJH, alpha=0.7)
    ax1.text(rp[peak_idx] + 0.3, dVdr[peak_idx] * 0.9,
             f"{rp[peak_idx]:.1f} nm", fontsize=8, color=C_BJH)
    ax1.axvline(N2_CAVITATION_NM, ls=":", lw=0.8, color="0.6")
    ax1.set_xlabel("Pore Diameter (nm)")
    ax1.set_ylabel(r"d$V_p$/d$r_p$ (cm³ g⁻¹ nm⁻¹)")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.set_title("Differential PSD")

    ax2r = ax2.twinx()
    ax2.plot(rp, cum_Vp, "-", color=C_CUM, lw=1.5, label="Vp cumul.")
    ax2r.plot(rp, cum_Sap, "--", color=C_BJH, lw=1.5, label="Sap cumul.")
    ax2.set_xlabel("Pore Diameter (nm)")
    ax2.set_ylabel("Cum. Pore Volume (cm³ g⁻¹)", color=C_CUM)
    ax2r.set_ylabel("Cum. Surface Area (m² g⁻¹)", color=C_BJH)
    ax2.tick_params(axis="y", colors=C_CUM)
    ax2r.tick_params(axis="y", colors=C_BJH)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.set_title("Cumulative Pore Volume")
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_bjh, use_container_width=True)
    plt.close(fig_bjh)

    if show_features and hyst_cls["type"] != "None":
        st.divider()
        st.markdown("**Hysteresis Feature Scores**")
        scores = hyst_cls["scores"]
        score_df = pd.DataFrame(
            [[k, v, "█" * v + "░" * (8 - v)]
             for k, v in sorted(scores.items(), key=lambda x: -x[1])],
            columns=["Type", "Score", "Bar"]
        )
        st.dataframe(score_df, hide_index=True, use_container_width=False)
        st.markdown("**Feature Analysis**")
        feat_df = pd.DataFrame(
            [[k, ("✓" if v is True else "✗" if v is False else str(v))]
             for k, v in hyst_cls["features"].items()],
            columns=["Feature", "Value"]
        )
        st.dataframe(feat_df, hide_index=True, use_container_width=False)


# ── TAB 4: T-PLOT ──────────────────────────────────────────────────────────────
with tab_tplot:
    if not show_tplot:
        st.info("Enable T-Plot analysis from the sidebar options.")
    else:
        st.subheader("T-Plot Micropore Analysis")
        try:
            from tplot_analysis import TPlotAnalyser
            tp = TPlotAnalyser(
                pressure=data["ads"][:, 0],
                volume_adsorbed=data["ads"][:, 1],
                s_bet=s["S_BET"],
                total_pore_volume=s["Vp_total"],
            )
            col_t1, col_t2 = st.columns([1, 2])
            with col_t1:
                res = tp.results
                st.markdown("**T-Plot Results**")
                st.table(pd.DataFrame({
                    "Parameter": ["S_BET", "S_ext", "S_micro", "V_micro", "V_meso"],
                    "Value": [
                        f"{res.get('S_BET', float('nan')):.2f}",
                        f"{res.get('S_ext', float('nan')):.2f}",
                        f"{res.get('S_micro', float('nan')):.2f}",
                        f"{res.get('V_micro', float('nan')):.4f}",
                        f"{res.get('V_meso', float('nan')):.4f}",
                    ],
                    "Unit": ["m² g⁻¹", "m² g⁻¹", "m² g⁻¹", "cm³ g⁻¹", "cm³ g⁻¹"],
                }))
            with col_t2:
                buf = io.BytesIO()
                tp.plot_tplot(save_path=buf, sample_name=sample_name)
                buf.seek(0)
                st.image(buf, use_container_width=True)
        except ImportError:
            st.warning("tplot_analysis.py not found. T-Plot module unavailable.")
        except Exception as e:
            st.error(f"T-Plot error: {e}")


# ── TAB 5: DOWNLOAD ──────────────────────────────────────────────────────────────
with tab_download:
    st.subheader("Download Results")

    st.markdown("**📊 Publication Figure (4-panel, 300 dpi)**")
    with st.spinner("Rendering figure…"):
        plt.show = lambda: None
        plot_all(data, iso_cls, hyst_cls, bet_res, sample_name, save=False)
        fig_main = plt.gcf()
        png_bytes = _fig_to_bytes(fig_main)
        plt.close(fig_main)

    st.download_button(
        label="⬇ Download PNG (300 dpi)",
        data=png_bytes,
        file_name=f"{sample_name.replace(' ', '_')}_BET_analysis.png",
        mime="image/png",
    )

    st.divider()
    st.markdown("**📋 CSV Report**")
    report_rows = [
        ["Sample",                sample_name],
        ["S_BET (m2/g)",          f"{s['S_BET']:.3f}"],
        ["Vm (cm3(STP)/g)",       f"{s['Vm']:.4f}"],
        ["C constant",            f"{s['C']:.2f}"],
        ["C valid",               str(bet_res["C_valid"])],
        ["R2",                    f"{bet_res['R2']:.6f}"],
        ["Vp_total (cm3/g)",      f"{s['Vp_total']:.4f}"],
        ["dp_avg (nm)",           f"{s['dp_avg']:.3f}"],
        ["S_BJH (m2/g)",          f"{s['S_BJH']:.3f}"],
        ["BJH_peak_diam (nm)",    f"{s['rp_peak_BJH']*2:.2f}"],
        ["Isotherm type",         iso_cls["type"]],
        ["Hysteresis type",       hyst_cls["type"]],
        ["Hysteresis confidence", hyst_cls.get("confidence", "—")],
    ]
    csv_report = pd.DataFrame(report_rows, columns=["Parameter", "Value"])
    st.download_button(
        label="⬇ Download CSV Report",
        data=csv_report.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_name.replace(' ', '_')}_BET_report.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("**📚 Cite this tool**")
    citation = (
        "Jafari, H. (2026). BET_analyser: Publication-Quality BET/BJH + T-Plot "
        "Analysis Tool (v2.1.0). Zenodo. DOI: 10.5281/zenodo.21104234"
    )
    st.code(citation)
