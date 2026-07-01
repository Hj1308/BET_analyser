# BET_analyser 🔬

**Publication-Quality BET/BJH + T-Plot Analysis Tool**  
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

---

## What this tool does

Reads the XLS output from a BET instrument and computes:

| Module | File | Description |
|---|---|---|
| **BET/BJH** | `bet_analysis.py` | Isotherm + hysteresis classification, BET regression, BJH pore size distribution, cumulative pore volume, 4-panel figure |
| **T-Plot** | `tplot_analysis.py` | Harkins-Jura T-Plot: micropore volume, S_ext, pore type distribution (micro/meso/macro %), 2-panel figure |

---

## Installation

```bash
git clone https://github.com/Hj1308/BET_analyser.git
cd BET_analyser
pip install -r BET_requirements.txt
```

---

## Usage

### BET + BJH (from XLS file)

```bash
python bet_analysis.py --file C3N4.xls --sample "C3N4"
```

### T-Plot (as module — recommended with bet_analysis.py)

```python
from bet_analysis import read_bet_xls
from tplot_analysis import TPlotAnalyser

data = read_bet_xls("C3N4.xls")
s    = data["summary"]

tp = TPlotAnalyser(
    pressure          = data["ads"][:, 0],
    volume_adsorbed   = data["ads"][:, 1],
    s_bet             = s["S_BET"],
    total_pore_volume = s["Vp_total"]
)

tp.print_report(sample_name="C3N4")
tp.plot_tplot(save_path="C3N4_tplot.png", sample_name="C3N4")
```

### T-Plot standalone (demo data)

```bash
python tplot_analysis.py --s-bet 95.3 --vtot 0.38 --sample "C3N4"
```

---

## Output

### BET analysis (`bet_analysis.py`)
- 4-panel figure: Isotherm | BET plot | BJH PSD | Cumulative pore volume
- Console report: surface area, pore volume, isotherm type, hysteresis type
- ⚠ Automatic warning if BET C constant is negative (IUPAC 2015 validity check)

### T-Plot (`tplot_analysis.py`)
- 2-panel figure: t-plot with linear fit | pore type distribution bar
- Console report: S_BET, S_ext, S_micro, V_micro, V_meso, V_macro

---

## Isotherm & Hysteresis Classification

Follows **IUPAC 2015** recommendations (Thommes et al., *Pure Appl. Chem.* **87**, 1051–1069).

| IUPAC Type | Pore Structure |
|---|---|
| Type I(a) | Ultra-micropores < 1 nm (zeolites, activated carbons — very sharp knee at p/p₀ < 0.01) |
| Type I(b) | Micropores 1–2.5 nm (MOFs, hierarchical carbons — knee extends to ~0.1) |
| Type II | Non-porous / macroporous |
| Type III | Weak interaction, multilayer |
| Type IV | Mesoporous + hysteresis |
| Type V | Weak interaction + mesoporosity |
| Type VI | Stepped, uniform non-porous surface |

| Hysteresis | Pore Geometry |
|---|---|
| H1 | Uniform open cylinders |
| H2 | Ink-bottle / pore blocking |
| H3 | Slit-shaped (e.g. C₃N₄, clay) |
| H4 | Micropore + slit-shaped |

---

## Physical Constants (N₂ at 77 K)

All magic numbers are defined as named constants at the top of `bet_analysis.py`:

| Constant | Value | Meaning |
|---|---|---|
| `N2_BET_FACTOR` | 4.353 m² g⁻¹ per cm³(STP) g⁻¹ | Cross-sectional area σ = 0.162 nm² |
| `N2_TPLOT_SLOPE_FACTOR` | 15.47 m² g⁻¹ per cm³/(g·Å) | Harkins-Jura conversion |
| `N2_LIQUID_FACTOR` | 1547.0 cm³(STP) per cm³(liquid N₂) | At 77 K |
| `N2_CAVITATION_NM` | 3.4 nm | Forced closure diameter for N₂ at 77 K |

---

## IUPAC 2015 Validity Checks

- **BET C constant**: A `UserWarning` is raised if C < 0, indicating the selected p/p₀ range is outside the valid BET region. Adjust `start_pt`/`end_pt` so that 0.05 ≤ p/p₀ ≤ 0.35.
- **Monotonicity**: A warning is raised if BET linearisation y-values are not strictly increasing over the selected range.
- **BJH branch**: Surface area and pore size distribution are computed from the **adsorption branch** to avoid the ~3.4 nm N₂ cavitation artefact present in desorption BJH at 77 K.

---

## Related Repositories

- [CatLab-Tools](https://github.com/Hj1308/CatLab-Tools) — Kinetics, TOF, TOC, unit conversion
- [EISforge-](https://github.com/Hj1308/EISforge-) — EIS analysis + ML
- [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) — SEM particle sizing
- [Raman-analysis](https://github.com/Hj1308/Raman-analysis) — Raman spectroscopy toolkit
