# Roadmap

Items below were identified during the Phase 1 review and **deliberately
deferred**. None blocks use of the tool, but each is a real gap. Phases 1–2 are
complete (v3.0.0); this file records what remains so it is not forgotten.

---

## Phase 3

### Code items

#### 1. GCB reference t-curve (highest value)

The t-plot uses Harkins-Jura, which derives from oxidic surfaces. Microtrac
AppNote B-AD-010 shows the instrument using a GCB (graphitised carbon black)
reference for a carbon sample, which is the correct family for these materials.

Evidence it matters: across the five audit samples `S_external / S_BET` falls
monotonically with rising BET C constant — 1.489 at C 9.3, 1.234 at 32.1, 1.176
at 34.5, 1.069 at 57.6, 0.960 at 95.8 — crossing 1.0 near C ≈ 65. With n = 5
that is indicative, not proven.

`REFERENCE_CURVES` in `tplot_analysis.py` is already a name → callable registry,
so adding a tabulated curve is a data file plus an interpolating callable, with
no change to the fitting code. A request for the numerical data has been sent to
Microtrac support; Kaneko et al., *J. Colloid Interface Sci.* (2010) publishes
standard αs data for GCB and NGCB as a fallback source. If the table gives Γ in
μmol/m², the conversion is direct: `t [Å] = Γ × 0.3467`.

#### 2. Type IV(a) / IV(b) discrimination

Thommes et al. (2015) §4.2: hysteresis accompanies capillary condensation only
above a critical pore width (~4 nm for N₂ in cylindrical pores at 77 K); below
it, completely reversible Type IV(b) isotherms occur. The classifier can only
route a loop-free sample to I/II/III/VI, so **Type IV(b) is unreachable**.
Discriminating them needs pore-size input. A `# TODO` with this citation is
already in `classify_isotherm`.

#### 3. Type I(a) / I(b) criterion is a proxy

§4.2 distinguishes them by pore width — I(a) for mainly narrow micropores below
~1 nm, I(b) for broader distributions including wider micropores and narrow
mesopores below ~2.5 nm. The code uses `frac_ultra`, the fraction of maximum
uptake reached below p/p₀ = 0.01, with a 0.5 threshold. That correlates with
pore width but does not measure it. **BJH data is already parsed**, so the real
criterion is implementable.

#### 4. `N2_CAVITATION_NM = 3.4` has no citation

Traced to the initial commit as a hardcoded value. It is *correct* for what it
marks — the spurious desorption-BJH peak at the p/p₀ ≈ 0.42 forced closure — but
it is a different quantity from IUPAC's critical neck diameter (§4.3.1 gives
ca. 5–6 nm for N₂ at 77 K). Needs a source in the code comment so a future
reader does not mistake one for the other.

#### 5. `app_bet.py` test coverage

No unit tests. CI has a byte-compile check, which catches syntax errors only.
Real coverage would use Streamlit's `AppTest` harness.

### Experimental protocol

#### 6. Measurement guidance — not a code item

All five audit samples fail the t-plot data-sufficiency gate. Counts of
adsorption points below p/p₀ = 0.08 and 0.015:

| sample | min p/p₀ | < 0.08 | < 0.015 |
|--------|----------|--------|---------|
| 9.xls | 0.0291 | 3 | 0 |
| 14H.xls | 0.0068 | 2 | 1 |
| 13BgOH.xls | 0.0119 | 1 | 1 |
| g-OH.xls | 0.0136 | 1 | 1 |
| 10.xls | 0.0168 | 1 | 0 |

A t-plot micropore analysis needs at least one point below p/p₀ ≈ 0.015 and at
least three below 0.08. `14H.xls` reached 0.0068, so the instrument is capable —
the limitation is the measurement-range setting, not the hardware. Until a run
with adequate low-pressure sampling exists, the two-segment t-plot cannot be
exercised on real data at all. No software change addresses this.
