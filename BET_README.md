# BET Analysis Tool

Python tool for BET/BJH analysis of physisorption data — auto-classifies isotherm type (I–VI) and hysteresis loop (H1–H4), verifies BET regression, and generates publication-ready figures directly from instrument XLS output.

## What it does

| Analysis | Output |
|---|---|
| Isotherm type (IUPAC Type I–VI) | Auto-classification + explanation |
| Hysteresis type (IUPAC H1–H4) | Scoring system + confidence % |
| BET plot verification | R², slope, intercept, Vm — matched to instrument point range |
| BJH differential PSD | dVp/drp vs pore diameter (adsorption branch) |
| Cumulative pore volume | Vp and surface area vs diameter |
| BET vs BJH comparison | Surface area ratio and agreement note |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/bet-analysis.git
cd bet-analysis
pip install -r requirements.txt
```

## Usage

```bash
python bet_analysis.py --file YOUR_FILE.xls --sample "Sample Name"
```

Input: XLS file directly from BET instrument (tested with BELSORP).
Output: `Sample_Name_BET_analysis.png` (300 dpi, publication-ready)

## Notes

- BJH analysis uses the **adsorption branch** (more reliable per IUPAC 2015 and Microtrac guidelines)
- A cavitation artifact at ~3.4 nm is marked on the BJH plot — this is a physical phenomenon of N₂ at 77 K, not a real pore
- BET regression uses the exact point range reported by the instrument (`Starting point` / `End point` in the XLS)
- Hysteresis classification is based on a multi-feature scoring system (slope ratio, loop shape, plateau detection, closure point)

## Hysteresis Classification Logic

| Feature | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| Slope ratio (des/ads) | ~1 | >2 | ~1 | ~1 |
| Loop shape | narrow | triangular | wide | narrow |
| Plateau on adsorption | yes | yes | no | no |
| Flat at low p/p₀ | no | no | no | yes |
| Pore structure | uniform cylinders | ink-bottle | slit/plate aggregates | slit + micropores |

## Project Structure

```
bet-analysis/
├── bet_analysis.py     # main script
├── requirements.txt    # dependencies
└── README.md
```

## License
MIT
