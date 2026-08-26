"""
Langmuir monolayer-adsorption analysis.
=======================================
The Langmuir model assumes monolayer adsorption on energetically uniform
sites with no lateral interactions between adsorbed molecules:

        n = n_m · K (p/p0) / (1 + K (p/p0))

in which n is the adsorbed amount (cm³(STP) g⁻¹), n_m is the monolayer
capacity (cm³(STP) g⁻¹), K is the affinity constant (reciprocal relative
pressure units) and p/p0 is the relative pressure.

Linearisation (Langmuir linear plot):

        (p/p0) / n = 1 / (K · n_m) + (p/p0) / n_m

so a linear regression of (p/p0)/n against p/p0 yields

        slope     = 1 / n_m
        intercept = 1 / (K · n_m)

and therefore

        n_m = 1 / slope
        K   = slope / intercept

S_Langmuir is converted to a specific surface area using the same N₂ factor
as the BET model (σ = 0.162 nm², N₂ at 77 K) so that the two areas are
directly comparable:

        S_Langmuir = n_m · N2_LANGMUIR_FACTOR

This module deliberately does **not** apply the Rouquerol criteria (those are
BET-specific) and does not claim that Langmuir replaces BET. S_Langmuir is a
complementary descriptor: interpret it cautiously for heterogeneous,
mesoporous, multilayer-adsorption or microporous / Type-I systems.

Model applicability (domain checks added to `fit_langmuir_window`):
- The Langmuir model describes monolayer saturation; it is not applicable
  when a hysteresis loop is present (capillary condensation implies
  multilayer adsorption).
- The isotherm should approach a plateau at high p/p0 (saturation); a
  continuously rising isotherm violates the monolayer assumption.
- S_Langmuir should not exceed S_BET by more than LANGMUIR_S_BET_MARGIN
  (default 20 %); for N₂ physisorption at 77 K, the BET area already
  includes multilayer contributions, so a larger Langmuir area is
  physically implausible.
- R² below LANGMUIR_R2_THRESHOLD (default 0.99) indicates the linear
  Langmuir model does not adequately describe the data.

These checks are reported as `model_applicable`; `physical_fit` continues
to mean only that the fitted parameters (slope, intercept, n_m, K, S) are
positive and finite.

Author  : Hoda Jafari | github.com/Hj1308
License : MIT
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress

# N2 at 77 K: σ = 0.162 nm² → m²/g per cm³(STP)/g (identical to the BET factor,
# so BET and Langmuir areas are directly comparable).
N2_LANGMUIR_FACTOR = 4.353
MIN_LANGMUIR_POINTS = 3

# Applicability thresholds for Langmuir model (heuristics, documented).
# R² below this suggests the linear Langmuir model does not describe the data.
LANGMUIR_R2_THRESHOLD = 0.99
# S_Langmuir may not exceed S_BET by more than this margin (20 %) for N₂
# physisorption at 77 K — the BET area already includes multilayer
# contributions, so a larger Langmuir area is physically implausible.
LANGMUIR_S_BET_MARGIN = 0.20


def langmuir_linear_y(p_rel: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Standard linear Langmuir ordinate: y = (p/p0) / n."""
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = p_rel / n
    return y


def fit_langmuir_window(
    p_rel: np.ndarray,
    n: np.ndarray,
    *,
    has_hysteresis: bool = False,
    has_plateau: bool = True,
    S_BET: float | None = None,
) -> dict:
    """Linear Langmuir fit on a selected window, with propagated uncertainty.

    scipy.stats.linregress returns the standard errors of the slope
    (``stderr``) and intercept (``intercept_stderr``). First-order error
    propagation through n_m = 1/slope, S = n_m·F and K = slope/intercept
    gives (slope–intercept covariance neglected, consistent with the BET
    module):

        σ(n_m)      = σ_slope / slope²
        σ(S)        = F · σ(n_m)
        σ(K)        = |K| · √[(σ_slope/slope)² + (σ_intercept/intercept)²]

    Parameters
    ----------
    p_rel, n :
        Relative pressure and adsorbed amount (cm³(STP) g⁻¹) for the
        adsorption-branch points inside the selected window.
    has_hysteresis :
        True if the source isotherm exhibits a hysteresis loop (capillary
        condensation); Langmuir assumes monolayer saturation and is not
        applicable in that case.
    has_plateau :
        True if the isotherm approaches a plateau at high p/p0; a continuously
        rising isotherm violates the monolayer saturation assumption.
    S_BET :
        BET surface area (m²/g) for comparison; if provided, S_Langmuir is
        checked against S_BET with LANGMUIR_S_BET_MARGIN.

    Returns
    -------
    dict with fit results, plus:
      - ``physical_fit``: all fitted parameters (slope, intercept, n_m, K,
        S_Langmuir) are positive and finite.
      - ``model_applicable``: True only if physical_fit is True AND all
        domain-applicability checks pass (no hysteresis, plateau present,
        S_Langmuir ≤ S_BET*(1+margin) if S_BET given, R² ≥ threshold).
      - ``applicability``: dict with individual check results and thresholds.
    """
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)

    if p_rel.ndim != 1 or n.ndim != 1:
        raise ValueError("p_rel and n must be 1-D arrays.")
    if p_rel.shape != n.shape:
        raise ValueError("p_rel and n must be 1-D arrays of equal length.")
    if len(p_rel) < MIN_LANGMUIR_POINTS:
        raise ValueError(
            f"Langmuir fit requires at least {MIN_LANGMUIR_POINTS} points; "
            f"got {len(p_rel)}."
        )
    if not np.all(np.isfinite(p_rel)):
        raise ValueError("p_rel contains non-finite values.")
    if not np.all(np.isfinite(n)):
        raise ValueError("n contains non-finite values.")
    if np.any(p_rel <= 0) or np.any(p_rel >= 1):
        raise ValueError("p_rel must satisfy 0 < p/p0 < 1.")
    if np.any(n <= 0):
        raise ValueError("n must be strictly positive (adsorbed amount > 0).")

    y = langmuir_linear_y(p_rel, n)
    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite Langmuir ordinate; check p/p0 > 0 and n > 0.")

    reg = linregress(p_rel, y)
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    R2 = float(reg.rvalue ** 2)
    sigma_slope = float(reg.stderr)
    sigma_intercept = float(reg.intercept_stderr)

    # n_m = 1 / slope
    if abs(slope) > 1e-30:
        n_m = 1.0 / slope
        sigma_n_m = sigma_slope / slope ** 2
    else:
        n_m = np.nan
        sigma_n_m = np.nan

    # S_Langmuir = n_m · factor
    if np.isfinite(n_m):
        S_Langmuir = n_m * N2_LANGMUIR_FACTOR
        sigma_S_Langmuir = N2_LANGMUIR_FACTOR * sigma_n_m
    else:
        S_Langmuir = np.nan
        sigma_S_Langmuir = np.nan

    # K = slope / intercept
    if abs(slope) > 1e-30 and abs(intercept) > 1e-30:
        K = slope / intercept
        sigma_K = abs(K) * np.hypot(
            sigma_slope / abs(slope),
            sigma_intercept / abs(intercept),
        )
    else:
        K = np.nan
        sigma_K = np.nan

    positive_slope = bool(np.isfinite(slope) and slope > 0)
    positive_intercept = bool(np.isfinite(intercept) and intercept > 0)
    positive_n_m = bool(np.isfinite(n_m) and n_m > 0)
    positive_K = bool(np.isfinite(K) and K > 0)
    physical_fit = bool(
        positive_slope
        and positive_intercept
        and positive_n_m
        and positive_K
        and np.isfinite(S_Langmuir)
        and S_Langmuir > 0
    )

    # ── Model applicability checks ───────────────────────────────
    r2_ok = bool(R2 >= LANGMUIR_R2_THRESHOLD)
    plateau_ok = bool(has_plateau)
    no_hyst_ok = not has_hysteresis

    area_ok = True
    if (
        S_BET is not None
        and np.isfinite(S_BET)
        and S_BET > 0
        and np.isfinite(S_Langmuir)
    ):
        area_ok = bool(S_Langmuir <= S_BET * (1.0 + LANGMUIR_S_BET_MARGIN))

    model_applicable = bool(
        physical_fit and r2_ok and plateau_ok and no_hyst_ok and area_ok
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "R2": R2,
        "n_m": float(n_m),
        "K": float(K),
        "S_Langmuir": float(S_Langmuir),
        "sigma_slope": sigma_slope,
        "sigma_intercept": sigma_intercept,
        "sigma_n_m": float(sigma_n_m),
        "sigma_K": float(sigma_K),
        "sigma_S_Langmuir": float(sigma_S_Langmuir),
        "x": p_rel,
        "y": y,
        "p_min": float(np.min(p_rel)),
        "p_max": float(np.max(p_rel)),
        "n_points": int(len(p_rel)),
        "physical_fit": physical_fit,
        "model_applicable": model_applicable,
        "applicability": {
            "r2_ok": r2_ok,
            "r2_threshold": LANGMUIR_R2_THRESHOLD,
            "plateau_ok": plateau_ok,
            "no_hysteresis_ok": no_hyst_ok,
            "area_ok": area_ok,
            "s_bet_margin": LANGMUIR_S_BET_MARGIN,
        },
        "positive_slope": positive_slope,
        "positive_intercept": positive_intercept,
        "positive_n_m": positive_n_m,
        "positive_K": positive_K,
    }


def format_langmuir_report(result: dict, sample_name: str = "Sample") -> str:
    """Human-readable Langmuir report string."""
    applicable = result.get("model_applicable", result.get("physical_fit", False))
    status = "PASS" if applicable else "FAIL"
    lines = [
        f"Langmuir Surface Area — {sample_name}",
        f"  p/p0 window        : {result['p_min']:.4f} – {result['p_max']:.4f}"
        f"  ({result['n_points']} points)",
        f"  status             : {status}",
        f"  S_Langmuir         : {result['S_Langmuir']:.2f}"
        f" ± {result['sigma_S_Langmuir']:.2f} m² g⁻¹",
        f"  n_m                : {result['n_m']:.2f}"
        f" ± {result['sigma_n_m']:.2f} cm³(STP) g⁻¹",
        f"  K                  : {result['K']:.1f}"
        f" ± {result['sigma_K']:.1f} (p/p0)⁻¹",
        f"  R²                 : {result['R2']:.6f}",
    ]
    if "applicability" in result:
        ap = result["applicability"]
        lines.append(
            f"  applicability        : "
            f"no_hysteresis={ap['no_hysteresis_ok']}, "
            f"plateau={ap['plateau_ok']}, "
            f"R2≥{ap['r2_threshold']:.2f}={ap['r2_ok']}, "
            f"S_L≤S_BET*(1+{ap['s_bet_margin']:.0%})={ap['area_ok']}"
        )
    return "\n".join(lines)
