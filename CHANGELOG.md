# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Critical:** `np.trapz` (removed in NumPy 2.0) broke `classify_hysteresis` at
  runtime, and with it both the CLI and the Streamlit app. Replaced with a
  `np.trapezoid` / `np.trapz` shim.
- **Critical:** the Gurvich STP→liquid conversion factor was wrong by 2.393×;
  **any previously reported micropore volume was ~2.4× too low**.
- BJH differential pore-size distribution was plotted as `dV/dr` against a
  diameter axis, making the area under the curve 2× the true pore volume.
- Hysteresis was detected from the presence of a desorption array rather than
  loop area, making Types I, II, III and VI unreachable.
- The Type II branch required a plateau that a genuine Type II cannot have, so
  concave isotherms were reported as Type III.
- The t-plot fitted a single line where the method requires two, reporting the
  total surface area as the external surface area (Thommes et al. 2015;
  Lippens & de Boer 1964).
- `--no-show` disabled saving instead of display.
- Hysteresis scoring double-counted `is_flat_low`; score ties were resolved by
  dict insertion order.

### Added

- IUPAC validity warnings — BET C constant (§5.1.1), apparent BET area on
  Type I (§5.2.2, §5.1.1), BJH narrow-mesopore underestimate (§7.2, §9), and
  Gurvich pore volume without a plateau (§7.1) — Thommes et al. (2015).
- t-plot data-sufficiency gate: micropore quantities are now refused with an
  explanation rather than reported as `0.0` when the measurement cannot
  support them.
- Halsey reference t-curve as an option; synthetic isotherm test fixtures;
  two-segment t-plot tests.

## [2.3.0] - 2026-08-25

### Added

- Langmuir surface-area tab with monolayer capacity (`n_m`), affinity constant
  (`K`), `S_Langmuir`, `R²`, and propagated uncertainty.
- Langmuir physical-fit validation and CSV safety gate.
- BET sensitivity heatmap (BEaTmap-style) with selected-window marker and
  unique-window diagnostic.

## [2.2.0] - 2026-08-25

### Added

- Rouquerol auto BET range selection (`rouquerol.py`) with the four consistency
  criteria and a multi-window scan.
- BET uncertainty propagation (`σ(S_BET)`, `σ(C)` from `linregress` stderr).
- `R² ≥ 0.999` linearity filter for window selection.
- Rouquerol tab in the Streamlit app; instrument range matched by p/p₀ rather
  than sheet indices.
- Adjustable T-Plot fit window; unit tests and CI; `.xls` reading via
  `xls_reader` restored.

## [2.1.0] - 2026-07-01

### Added

- IUPAC 2015 validity checks (C < 0 warning, monotonicity check).
- Named physical constants; `np.trapz` compatibility fix; `setup_plot_style()`
  isolated to prevent import side-effects; Type I(a)/I(b) sub-classification;
  BJH adsorption branch noted in the report.

## [2.0.0] - 2026-06-14

### Added

- T-Plot module (`tplot_analysis.py`).
- 6-feature hysteresis scoring (H1–H4) with a confidence level.
- N₂ cavitation marker on the BJH panel.

## [1.0.0] - 2026-05-14

### Added

- Initial release: BET regression, BJH pore-size distribution, IUPAC isotherm
  classification, 4-panel figure.
