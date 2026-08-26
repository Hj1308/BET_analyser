# BET_analyser 🔬

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22101160.svg)](https://doi.org/10.5281/zenodo.22101160)
![Version](https://img.shields.io/badge/version-v2.3.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![IUPAC](https://img.shields.io/badge/IUPAC-2015%20compliant-orange?style=flat-square)

**Publication-Quality BET/BJH + T-Plot Analysis Tool**  
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

> **ODS kinetics & catalytic activity?**  
> → See [CatLab-Tools](https://github.com/Hj1308/CatLab-Tools)

---

## What is BET_analyser?

A Python tool for **publication-quality physisorption analysis** from BET instrument XLS/XLSX output, with a Streamlit web app.  
Designed for PhD-level materials characterisation — covers full IUPAC 2015 isotherm and hysteresis classification, BET regression with validity checks, **Rouquerol auto BET range selection**, BJH pore size distribution, T-Plot micropore analysis, and a **Langmuir surface-area** model.

Developed and validated for **graphene-like carbon nitride (C₃N₄)**, MOFs, zeolites, and hierarchical porous materials.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Hj1308/BET_analyser.git
cd BET_analyser
pip install -r BET_requirements.txt

# Run BET + BJH analysis (with Rouquerol auto range)
python bet_analysis.py --file C3N4.xls --sample "C3N4" --rouquerol

# Run T-Plot standalone (demo)
python tplot_analysis.py --s-bet 95.3 --vtot 0.38 --sample "C3N4"

# Or launch the web app
streamlit run app_bet.py
```

---

## 🖥️ Streamlit Web App

`app_bet.py` provides a browser UI for the same analyses as the CLI, without
writing Python:

```bash
pip install -r BET_requirements.txt
streamlit run app_bet.py
```

The app exposes tabs for the **Overview**, **BET**, **Langmuir**, **Rouquerol**,
**BJH / PSD**, **T-Plot**, and a **Download** tab that renders the publication
figure and a CSV report. It accepts XLS, XLSX and CSV uploads, and surfaces the
same IUPAC validity warnings and the t-plot sufficiency gate as the CLI.

---

## 📑 Modules

| Module | File | Description |
|--------|------|-------------|
| **BET/BJH** | `bet_analysis.py` | Isotherm + hysteresis classification, BET regression, BJH PSD, cumulative pore volume, 4-panel figure |
| **Rouquerol** | `rouquerol.py` | Auto BET linear-range selection via the four Rouquerol consistency criteria (IUPAC 2015 / ISO 9277); multi-window scan; instrument-range diagnosis |
| **Langmuir** | `langmuir.py` | Langmuir monolayer capacity, affinity constant, surface area, and propagated regression uncertainty |
| **T-Plot** | `tplot_analysis.py` | Harkins-Jura T-Plot: micropore volume, S_ext, pore type distribution (micro/meso/macro %), 2-panel figure |
| **Web app** | `app_bet.py` | Streamlit UI: XLS/XLSX/CSV upload, all analyses, downloadable figure + CSV report |
| **XLS reader** | `xls_reader.py` | Legacy .xls reader via xlrd API (bypasses pandas engine guard) |

---

## 🎯 Rouquerol BET Range Selection

The classical 0.05–0.35 p/p₀ window is **not unique** — two operators can report BET areas differing by ~20 % on the same isotherm. `rouquerol.py` replaces that subjective choice with the four consistency criteria of Rouquerol et al. (2007), adopted by IUPAC 2015 and ISO 9277:

| # | Criterion | Physical meaning |
|---|-----------|------------------|
| **C1** | C > 0 (and intercept > 0) | Physically meaningful monolayer capacity |
| **C2** | n(1 − p/p₀) increases continuously | Upper bound of the valid BET region (Rouquerol transform maximum) |
| **C3** | p(n_m) lies inside the window | Monolayer pressure must be within the fitted range |
| **C4** | 1/(√C + 1) ≈ p(n_m) within ±20 % | BET theory self-consistency |

All contiguous windows (≥ 4 points) are scanned; among fully valid windows the one with the **most points** (then highest R²) is selected. The instrument's own Starting/End point range is evaluated against the same criteria — matched by **p/p₀ values**, not sheet indices.

```python
from rouquerol import select_bet_range, format_rouquerol_report

result = select_bet_range(ads[:, 0], ads[:, 1])
print(format_rouquerol_report(result, "C3N4"))
```

Run the unit tests:

```bash
pip install pytest
pytest tests/ -v
```

---

## ⚗️ Langmuir Surface Area

`langmuir.py` provides a **complementary monolayer-adsorption model** alongside BET. It fits the linearised Langmuir isotherm `(p/p0)/n` vs `p/p0` to report the monolayer capacity `n_m`, the affinity constant `K`, the specific surface area `S_Langmuir = n_m × 4.353`, the regression `R²`, and the propagated first-order uncertainty on each quantity.

S_Langmuir uses the **same N₂ cross-section factor as BET**, so the two areas are directly comparable. However, Langmuir is *not* an automatic replacement for BET: it assumes monolayer adsorption on uniform, non-interacting sites, so it should be interpreted cautiously for heterogeneous, mesoporous, or multilayer-adsorption systems. It is particularly useful to compare alongside BET for **Type I / microporous isotherms**, where the monolayer model is often physically reasonable.

```python
from langmuir import fit_langmuir_window, format_langmuir_report

result = fit_langmuir_window(ads[:, 0], ads[:, 1])
print(format_langmuir_report(result, "C3N4"))
```

---

## 📊 Output

### BET/BJH (`bet_analysis.py`)

- **4-panel figure** (300 dpi, publication-ready):
  - Panel A — N₂ Adsorption–Desorption Isotherm with hysteresis fill
  - Panel B — BET Plot with regression line + C constant validity flag (⚠ if C < 0)
  - Panel C — BJH Differential Pore Size Distribution (adsorption branch) with N₂ cavitation marker
  - Panel D — Cumulative Pore Volume + S_BET vs S_BJH comparison
- **Console report**: S_BET, C, Vm, Vp_total, d_avg, S_BJH, isotherm type, hysteresis type with scoring, Rouquerol range report (with `--rouquerol`)

### T-Plot (`tplot_analysis.py`)

- **2-panel figure**: t-plot with linear fit | pore type distribution bar
- **Console report**: S_BET, S_ext, S_micro, V_micro, V_meso, V_macro

---

## 🔬 Isotherm Classification (IUPAC 2015)

Ref: Thommes et al., *Pure Appl. Chem.* **87**, 1051–1069 (2015).

| IUPAC Type | Pore Structure | Typical Material |
|------------|----------------|------------------|
| **Type I(a)** | Ultra-micropores < 1 nm; very sharp knee at p/p₀ < 0.01 | Zeolites, activated carbons |
| **Type I(b)** | Micropores 1–2.5 nm; knee extends to ~0.1 | MOFs, hierarchical carbons |
| **Type II** | Non-porous / macroporous; S-shaped | Silica, alumina |
| **Type III** | Weak adsorbate–adsorbent interaction; convex | PTFE, ice |
| **Type IV** | Mesoporous + hysteresis; capillary condensation | SBA-15, MCM-41 |
| **Type V** | Weak interaction + mesoporosity | Certain MOFs |
| **Type VI** | Stepped; uniform non-porous surface | Graphite |

---

## 🔁 Hysteresis Classification (IUPAC 2015)

Automatically scored using 6 physical features (area, slope ratio, closure point, plateau, flatness, loop shape).

| Type | Pore Geometry | Typical Material |
|------|---------------|------------------|
| **H1** | Uniform open-ended cylinders; narrow symmetric loop | SBA-15, MCM-41 |
| **H2** | Ink-bottle pores / pore blocking; triangular loop, steep desorption | Disordered silicas |
| **H3** | Non-rigid slit-shaped aggregates; no limiting adsorption at p/p₀→1 | C₃N₄, clay minerals |
| **H4** | Narrow slit + micropores; nearly flat parallel branches | Microporous carbons |

---

## ✅ IUPAC 2015 Validity Checks

| Check | Behaviour |
|-------|-----------|
| **BET C constant** | `UserWarning` raised if C < 0 — invalid p/p₀ range; adjust `start_pt`/`end_pt` to 0.05 ≤ p/p₀ ≤ 0.35 |
| **Monotonicity** | `UserWarning` raised if BET y-values are not strictly increasing over the selected range |
| **Rouquerol criteria** | Four-criterion consistency check on every candidate window; PASS/FAIL reported per criterion |
| **BJH branch** | Adsorption branch used to avoid the ~3.4 nm N₂ cavitation artefact in desorption BJH at 77 K |
| **Missing data** | `ValueError` with descriptive message if required XLS sheets or row labels are absent |

### Reported validity caveats

Beyond raising on hard errors, the report (and the app) now surfaces IUPAC
validity caveats that were previously silent — a low BET C constant (interpretation
of `n_m` questionable when C < 50), the BET area on a Type I isotherm being an
*apparent* area, BJH underestimating narrow mesopores by 20–30 % below ~10 nm,
and the Gurvich-rule total pore volume being invalid without a high-p/p₀ plateau
(Thommes et al. 2015 §5.1.1, §5.2.2, §7.1, §7.2, §9).

A t-plot **micropore** analysis additionally requires adsorption points below
**p/p₀ ≈ 0.015**, ideally several lower still — check the instrument's
low-pressure specification and measurement-range setting. If the measurement
lacks them, the tool refuses to report a micropore volume rather than printing
`0.0` (Thommes et al. 2015 §6.1). BJH is valid only above ~2 nm pore diameter;
below that, HK/SF or DFT methods are required (Thommes et al. 2015 §7.2, §9).

---

## 📌 Physical Constants (N₂ at 77 K)

All constants are defined as named variables at the top of `bet_analysis.py` (no magic numbers).

| Constant | Value | Definition |
|----------|-------|------------|
| `N2_BET_FACTOR` | 4.353 m² g⁻¹ per cm³(STP) g⁻¹ | N₂ cross-section σ = 0.162 nm², Avogadro + molar volume |
| `N2_TPLOT_SLOPE_FACTOR` | 15.47 m² g⁻¹ per cm³/(g·Å) | Harkins-Jura t-curve conversion |
| `N2_LIQUID_FACTOR` | 1547.0 cm³(STP) per cm³(liquid N₂) | At 77 K |
| `N2_CAVITATION_NM` | 3.4 nm | Forced closure diameter for N₂ at 77 K |

---

## 📦 Usage as a Module

```python
from bet_analysis import read_bet_xls, classify_isotherm, verify_bet
from tplot_analysis import TPlotAnalyser

# Read instrument XLS
data = read_bet_xls("C3N4.xls")
s    = data["summary"]

# Isotherm classification
iso  = classify_isotherm(data["ads"], data["des"])
print(iso["type"], iso["explanation"])

# BET regression with IUPAC validity check + Rouquerol auto range
bet  = verify_bet(data["bet_pts"], s, ads=data["ads"])
print(f"S_BET = {bet['S_BET_calc']:.2f} m²/g  |  C = {bet['C']:.1f}  |  R² = {bet['R2']:.5f}")

# T-Plot micropore analysis
tp = TPlotAnalyser(
    pressure          = data["ads"][:, 0],
    volume_adsorbed   = data["ads"][:, 1],
    s_bet             = s["S_BET"],
    total_pore_volume = s["Vp_total"]
)
tp.print_report(sample_name="C3N4")
tp.plot_tplot(save_path="C3N4_tplot.png", sample_name="C3N4")
```

---

## 🗂 Repository Structure

```
BET_analyser/
├── app_bet.py             # Streamlit web application
├── bet_analysis.py        # BET + BJH main script
├── langmuir.py            # Langmuir monolayer-adsorption analysis
├── rouquerol.py           # Rouquerol auto BET range selection
├── tplot_analysis.py      # T-Plot analysis module
├── xls_reader.py          # Legacy .xls reader (xlrd API)
├── conftest.py            # pytest path configuration
├── tests/
│   ├── synthetic_isotherms.py          # closed-form isotherm fixtures
│   ├── test_isotherm_classification.py
│   ├── test_langmuir.py
│   ├── test_rouquerol.py
│   └── test_tplot_two_segment.py
├── .streamlit/            # Streamlit config
├── .github/workflows/     # CI
├── .python-version        # pinned dev Python (3.11)
├── packages.txt           # Streamlit Cloud system deps
├── pyproject.toml         # packaging metadata + dev extra
├── CITATION.cff           # citation metadata
├── CHANGELOG.md           # release history
├── LICENSE                # MIT
├── BET_requirements.txt   # runtime deps
├── requirements.txt       # runtime deps (packaging source)
└── README.md
```

---

## 📚 References

1. Thommes, M. et al. *Pure Appl. Chem.* **2015**, 87, 1051–1069. DOI: [10.1515/pac-2014-1117](https://doi.org/10.1515/pac-2014-1117) — *IUPAC 2015 physisorption classification*
2. Rouquerol, J.; Llewellyn, P.; Rouquerol, F. *Stud. Surf. Sci. Catal.* **2007**, 160, 49–56. DOI: [10.1016/S0167-2991(07)80008-5](https://doi.org/10.1016/S0167-2991(07)80008-5) — *Rouquerol consistency criteria*
3. ISO 9277:2010 — *Determination of the specific surface area of solids by gas adsorption — BET method*
4. Osterrieth, J. W. M. et al. *Adv. Mater.* **2022**, 34, 2201502. DOI: [10.1002/adma.202201502](https://doi.org/10.1002/adma.202201502) — *BETSI multi-region fitting*
5. Rouquerol, J. et al. *Adsorption by Powders and Porous Solids*, 2nd ed.; Academic Press, 2014.
6. Gregg, S.J.; Sing, K.S.W. *Adsorption, Surface Area and Porosity*, 2nd ed.; Academic Press, 1982.
7. Barrett, E.P.; Joyner, L.G.; Halenda, P.P. *J. Am. Chem. Soc.* **1951**, 73, 373–380. DOI: [10.1021/ja01145a126](https://doi.org/10.1021/ja01145a126) — *BJH method*
8. Brunauer, S.; Emmett, P.H.; Teller, E. *J. Am. Chem. Soc.* **1938**, 60, 309–319. DOI: [10.1021/ja01269a023](https://doi.org/10.1021/ja01269a023) — *BET theory*

---

## 🔀 Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

---

## 🔗 Related Repositories

| Repo | Purpose |
|------|---------|
| [CatLab-Tools](https://github.com/Hj1308/CatLab-Tools) | ODS kinetics, TOF/TON, Arrhenius, residual diagnostics |
| [EISforge](https://github.com/Hj1308/EISforge) | EIS analysis + ML |
| [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) | SEM particle sizing |
| [Raman-analysis](https://github.com/Hj1308/Raman-analysis) | Raman spectroscopy toolkit |

---

## Cite This Software

If you use BET_analyser in your research, please cite:

> Jafari, H. (2026). *BET_analyser: Publication-Quality BET/BJH + T-Plot Analysis Tool* (v2.3.0). Zenodo.  
> DOI: [10.5281/zenodo.22101160](https://doi.org/10.5281/zenodo.22101160)

---

## License

MIT — free to use, modify, and distribute.
