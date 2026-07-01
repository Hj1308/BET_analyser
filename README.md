# BET_analyser 🔬

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21104234.svg)](https://doi.org/10.5281/zenodo.21104234)
![Version](https://img.shields.io/badge/version-v2.1.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![IUPAC](https://img.shields.io/badge/IUPAC-2015%20compliant-orange?style=flat-square)

**Publication-Quality BET/BJH + T-Plot Analysis Tool**  
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

> **ODS kinetics & catalytic activity?**  
> → See [CatLab-Tools](https://github.com/Hj1308/CatLab-Tools)

---

## What is BET_analyser?

A Python command-line tool for **publication-quality physisorption analysis** from BET instrument XLS output.  
Designed for PhD-level materials characterisation — covers full IUPAC 2015 isotherm and hysteresis classification, BET regression with validity checks, BJH pore size distribution, and T-Plot micropore analysis.

Developed and validated for **graphene-like carbon nitride (C₃N₄)**, MOFs, zeolites, and hierarchical porous materials.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Hj1308/BET_analyser.git
cd BET_analyser
pip install -r BET_requirements.txt

# Run BET + BJH analysis
python bet_analysis.py --file C3N4.xls --sample "C3N4"

# Run T-Plot standalone (demo)
python tplot_analysis.py --s-bet 95.3 --vtot 0.38 --sample "C3N4"
```

---

## 📑 Modules

| Module | File | Description |
|--------|------|-------------|
| **BET/BJH** | `bet_analysis.py` | Isotherm + hysteresis classification, BET regression, BJH PSD, cumulative pore volume, 4-panel figure |
| **T-Plot** | `tplot_analysis.py` | Harkins-Jura T-Plot: micropore volume, S_ext, pore type distribution (micro/meso/macro %), 2-panel figure |

---

## 📊 Output

### BET/BJH (`bet_analysis.py`)

- **4-panel figure** (300 dpi, publication-ready):
  - Panel A — N₂ Adsorption–Desorption Isotherm with hysteresis fill
  - Panel B — BET Plot with regression line + C constant validity flag (⚠ if C < 0)
  - Panel C — BJH Differential Pore Size Distribution (adsorption branch) with N₂ cavitation marker
  - Panel D — Cumulative Pore Volume + S_BET vs S_BJH comparison
- **Console report**: S_BET, C, Vm, Vp_total, d_avg, S_BJH, isotherm type, hysteresis type with scoring

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
| **BJH branch** | Adsorption branch used to avoid the ~3.4 nm N₂ cavitation artefact in desorption BJH at 77 K |
| **Missing data** | `ValueError` with descriptive message if required XLS sheets or row labels are absent |

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

# BET regression with IUPAC validity check
bet  = verify_bet(data["bet_pts"], s)
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
├── bet_analysis.py        # BET + BJH main script (v2.1.0)
├── tplot_analysis.py      # T-Plot analysis module
├── BET_requirements.txt   # numpy, pandas, matplotlib, scipy, tabulate, xlrd
└── README.md
```

---

## 📚 References

1. Thommes, M. et al. *Pure Appl. Chem.* **2015**, 87, 1051–1069. DOI: [10.1515/pac-2014-1117](https://doi.org/10.1515/pac-2014-1117) — *IUPAC 2015 physisorption classification*
2. Rouquerol, J. et al. *Adsorption by Powders and Porous Solids*, 2nd ed.; Academic Press, 2014.
3. Gregg, S.J.; Sing, K.S.W. *Adsorption, Surface Area and Porosity*, 2nd ed.; Academic Press, 1982.
4. Barrett, E.P.; Joyner, L.G.; Halenda, P.P. *J. Am. Chem. Soc.* **1951**, 73, 373–380. DOI: [10.1021/ja01145a126](https://doi.org/10.1021/ja01145a126) — *BJH method*
5. Brunauer, S.; Emmett, P.H.; Teller, E. *J. Am. Chem. Soc.* **1938**, 60, 309–319. DOI: [10.1021/ja01269a023](https://doi.org/10.1021/ja01269a023) — *BET theory*

---

## 🔀 Changelog

| Version | Key Changes |
|---------|-------------|
| **v2.1.0** | IUPAC 2015 validity checks (C < 0 warning, monotonicity check); named physical constants; `np.trapz` compatibility fix for NumPy < 2.0; `setup_plot_style()` isolated to prevent import side-effects; Type I(a)/I(b) sub-classification; BJH adsorption branch noted in report |
| **v2.0.0** | T-Plot module (`tplot_analysis.py`); 6-feature hysteresis scoring (H1–H4); confidence level output; N₂ cavitation marker on BJH panel |
| **v1.0.0** | Initial release: BET regression, BJH PSD, IUPAC isotherm classification, 4-panel figure |

---

## 🔗 Related Repositories

| Repo | Purpose |
|------|---------|
| [CatLab-Tools](https://github.com/Hj1308/CatLab-Tools) | ODS kinetics, TOF/TON, Arrhenius, residual diagnostics |
| [EISforge-](https://github.com/Hj1308/EISforge-) | EIS analysis + ML |
| [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) | SEM particle sizing |
| [Raman-analysis](https://github.com/Hj1308/Raman-analysis) | Raman spectroscopy toolkit |

---

## Cite This Software

If you use BET_analyser in your research, please cite:

> Jafari, H. (2026). *BET_analyser: Publication-Quality BET/BJH + T-Plot Analysis Tool* (v2.1.0). Zenodo.  
> DOI: [10.5281/zenodo.21104234](https://doi.org/10.5281/zenodo.21104234)

---

## License

MIT — free to use, modify, and distribute.
