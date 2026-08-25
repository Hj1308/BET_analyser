"""
Rouquerol consistency criteria for BET linear-range selection.
===============================================================
The classical 0.05–0.35 p/p0 window is not unique: two operators can
report BET areas that differ by ~20 % on the same isotherm. Rouquerol
et al. (2007) and IUPAC 2015 / ISO 9277 replace that guess with four
consistency checks that make the linear range essentially unique.

This module does **not** re-implement pyGAPS. It is the missing piece
in BET_analyser: a defensible BET range, plus a simple multi-window
scan (BETSI-style enumeration, not BETSI's full statistical engine).

Four criteria (IUPAC Technical Report, Thommes et al. 2015):
  1. C > 0 (and intercept > 0 so Vm is physically meaningful).
  2. n(1 − p/p0) increases continuously with p/p0 over the window
     (the Rouquerol transform has not yet reached its maximum).
  3. The monolayer loading nm corresponds to a p/p0 that lies inside
     the selected window.
  4. The theoretical monolayer pressure 1/(√C + 1) agrees with the
     experimental p(nm) within a relative tolerance (default 20 %).

Selection rule (v2): the four criteria are necessary but not always
sufficient — on Type IV isotherms with capillary condensation, a wide
window can pass all four while the BET plot is visibly curved
(R² ≈ 0.92) and S_BET is overestimated by ~30 %. Therefore, among
Rouquerol-valid windows only those with R² ≥ 0.999 are kept, and the
largest of these (then highest R²) is selected.

References
----------
Rouquerol, J.; Llewellyn, P.; Rouquerol, F. Stud. Surf. Sci. Catal.
    2007, 160, 49–56. https://doi.org/10.1016/S0167-2991(07)80008-5
Thommes, M. et al. Pure Appl. Chem. 2015, 87, 1051–1069.
    https://doi.org/10.1515/pac-2014-1117
ISO 9277:2010. Determination of the specific surface area of solids
    by gas adsorption — BET method.
Osterrieth, J. W. M. et al. Adv. Mater. 2022, 34, 2201502.
    https://doi.org/10.1002/adma.202201502  (BETSI multi-region idea)

Author  : Hoda Jafari | github.com/Hj1308
License : MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress

# N2 at 77 K: σ = 0.162 nm² → m²/g per cm³(STP)/g
N2_BET_FACTOR = 4.353
MIN_POINTS_DEFAULT = 4
CRITERION4_TOL_DEFAULT = 0.20
R2_THRESHOLD_DEFAULT = 0.999


def rouquerol_transform(p_rel: np.ndarray, n: np.ndarray) -> np.ndarray:
    """n(1 − p/p0). Upper BET bound is the first maximum of this curve."""
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)
    return n * (1.0 - p_rel)


def bet_linear_y(p_rel: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Standard BET ordinate: 1 / [n (p0/p − 1)] = (p/p0) / [n (1 − p/p0)]."""
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)
    denom = n * (1.0 - p_rel)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = p_rel / denom
    return y


def fit_bet_window(p_rel: np.ndarray, n: np.ndarray) -> dict:
    """Linear BET fit on one contiguous window."""
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)
    y = bet_linear_y(p_rel, n)
    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite BET ordinate; check p/p0 < 1 and n > 0.")
    slope, intercept, r, *_ = linregress(p_rel, y)
    denom = slope + intercept
    Vm = np.nan if abs(denom) < 1e-30 else 1.0 / denom
    C = np.nan if abs(intercept) < 1e-30 else 1.0 + slope / intercept
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r ** 2),
        "Vm": float(Vm) if np.isfinite(Vm) else np.nan,
        "C": float(C) if np.isfinite(C) else np.nan,
        "y": y,
        "S_BET": float(Vm * N2_BET_FACTOR) if np.isfinite(Vm) else np.nan,
    }


def increasing_prefix_mask(p_rel: np.ndarray, n: np.ndarray) -> np.ndarray:
    """True up to (and including) the last point before n(1−p/p0) decreases."""
    t = rouquerol_transform(p_rel, n)
    keep = np.ones(len(p_rel), dtype=bool)
    dt = np.diff(t)
    for i, d in enumerate(dt):
        if d <= 0:
            keep[i + 1 :] = False
            break
    return keep


def monolayer_pressure_theory(C: float) -> float:
    """p/p0 at monolayer from BET theory: 1 / (√C + 1)."""
    if not np.isfinite(C) or C <= 0:
        return np.nan
    return float(1.0 / (np.sqrt(C) + 1.0))


def monolayer_pressure_experimental(p_rel: np.ndarray, n: np.ndarray, nm: float) -> float:
    """p/p0 at which adsorbed amount equals nm, by linear interpolation."""
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)
    if not np.isfinite(nm):
        return np.nan
    order = np.argsort(n)
    f = interp1d(n[order], p_rel[order], bounds_error=False, fill_value=np.nan)
    val = f(nm)
    return float(val) if np.isfinite(val) else np.nan


@dataclass
class RouquerolWindow:
    i0: int
    i1: int
    n_points: int
    p_min: float
    p_max: float
    slope: float
    intercept: float
    R2: float
    Vm: float
    C: float
    S_BET: float
    pm_exp: float
    pm_theory: float
    c1_C_positive: bool
    c2_n1mp_increasing: bool
    c3_nm_in_range: bool
    c4_pm_consistency: bool
    valid: bool
    p: np.ndarray = field(repr=False)
    n: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "i0": self.i0,
            "i1": self.i1,
            "n_points": self.n_points,
            "p_min": self.p_min,
            "p_max": self.p_max,
            "slope": self.slope,
            "intercept": self.intercept,
            "R2": self.R2,
            "Vm": self.Vm,
            "C": self.C,
            "S_BET": self.S_BET,
            "pm_exp": self.pm_exp,
            "pm_theory": self.pm_theory,
            "c1_C_positive": self.c1_C_positive,
            "c2_n1mp_increasing": self.c2_n1mp_increasing,
            "c3_nm_in_range": self.c3_nm_in_range,
            "c4_pm_consistency": self.c4_pm_consistency,
            "valid": self.valid,
        }


def evaluate_window(
    p_rel: np.ndarray,
    n: np.ndarray,
    i0: int,
    i1: int,
    *,
    criterion4_tol: float = CRITERION4_TOL_DEFAULT,
) -> Optional[RouquerolWindow]:
    """Score one inclusive index window [i0, i1] against the four criteria."""
    pw = np.asarray(p_rel[i0 : i1 + 1], dtype=float)
    nw = np.asarray(n[i0 : i1 + 1], dtype=float)
    if len(pw) < MIN_POINTS_DEFAULT:
        return None
    if np.any(pw <= 0) or np.any(pw >= 1) or np.any(nw <= 0):
        return None
    try:
        fit = fit_bet_window(pw, nw)
    except (ValueError, FloatingPointError):
        return None

    t = rouquerol_transform(pw, nw)
    c2 = bool(np.all(np.diff(t) > 0))
    c1 = bool(
        np.isfinite(fit["C"])
        and fit["C"] > 0
        and np.isfinite(fit["Vm"])
        and fit["Vm"] > 0
        and fit["intercept"] > 0
    )
    pm_exp = monolayer_pressure_experimental(pw, nw, fit["Vm"])
    c3 = bool(np.isfinite(pm_exp) and (pw.min() <= pm_exp <= pw.max()))
    pm_th = monolayer_pressure_theory(fit["C"])
    if np.isfinite(pm_exp) and np.isfinite(pm_th) and pm_exp > 0:
        c4 = abs(pm_th - pm_exp) / pm_exp <= criterion4_tol
    else:
        c4 = False
    valid = bool(c1 and c2 and c3 and c4)

    return RouquerolWindow(
        i0=int(i0),
        i1=int(i1),
        n_points=int(i1 - i0 + 1),
        p_min=float(pw.min()),
        p_max=float(pw.max()),
        slope=fit["slope"],
        intercept=fit["intercept"],
        R2=fit["R2"],
        Vm=fit["Vm"],
        C=fit["C"],
        S_BET=fit["S_BET"],
        pm_exp=pm_exp,
        pm_theory=pm_th,
        c1_C_positive=c1,
        c2_n1mp_increasing=c2,
        c3_nm_in_range=c3,
        c4_pm_consistency=c4,
        valid=valid,
        p=pw,
        n=nw,
        y=fit["y"],
    )


def select_bet_range(
    p_rel: np.ndarray,
    n: np.ndarray,
    *,
    min_points: int = MIN_POINTS_DEFAULT,
    criterion4_tol: float = CRITERION4_TOL_DEFAULT,
    r2_threshold: float = R2_THRESHOLD_DEFAULT,
    restrict_to_increasing: bool = True,
) -> dict:
    """Enumerate contiguous windows and pick the best Rouquerol-valid range.

    Selection (v2):
      1. keep windows passing all four Rouquerol criteria
      2. of those, keep only windows with R² ≥ r2_threshold (linearity)
      3. pick the window with the most points, then highest R²

    Step 2 is essential on Type IV isotherms: capillary condensation can
    let a wide window pass all four criteria while the BET plot is curved
    (R² ≈ 0.9), overestimating S_BET by tens of percent.

    Parameters
    ----------
    p_rel, n :
        Adsorption branch, relative pressure and amount adsorbed
        (cm³(STP) g⁻¹ or any consistent extensive unit).
    """
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)
    if p_rel.ndim != 1 or n.ndim != 1 or len(p_rel) != len(n):
        raise ValueError("p_rel and n must be 1-D arrays of equal length.")

    order = np.argsort(p_rel)
    p_s, n_s = p_rel[order], n[order]
    finite = np.isfinite(p_s) & np.isfinite(n_s) & (p_s > 0) & (p_s < 1) & (n_s > 0)
    p_s, n_s = p_s[finite], n_s[finite]

    t_full = rouquerol_transform(p_s, n_s)
    if restrict_to_increasing:
        mask = increasing_prefix_mask(p_s, n_s)
        search_idx = np.where(mask)[0]
    else:
        search_idx = np.arange(len(p_s))

    if len(search_idx) < min_points:
        search_idx = np.arange(len(p_s))

    candidates: list[RouquerolWindow] = []
    for a in range(len(search_idx)):
        for b in range(a + min_points - 1, len(search_idx)):
            i0 = int(search_idx[a])
            i1 = int(search_idx[b])
            if i1 - i0 + 1 < min_points:
                continue
            win = evaluate_window(
                p_s, n_s, i0, i1, criterion4_tol=criterion4_tol
            )
            if win is not None:
                candidates.append(win)

    valid = [c for c in candidates if c.valid]
    linear = [c for c in valid if c.R2 >= r2_threshold]
    pool = linear if linear else (valid if valid else candidates)
    pool.sort(key=lambda c: (c.n_points, c.R2), reverse=True)
    best = pool[0] if pool else None

    return {
        "best": best,
        "valid_windows": valid,
        "n_candidates": len(candidates),
        "n_valid": len(valid),
        "n_linear": len(linear),
        "r2_threshold": r2_threshold,
        "p_sorted": p_s,
        "n_sorted": n_s,
        "rouquerol_transform": t_full,
        "selection_rule": (
            f"Rouquerol-valid + R²≥{r2_threshold}, then max points, then max R²"
        ),
    }


def diagnose_instrument_range(
    p_rel: np.ndarray,
    n: np.ndarray,
    i0: int,
    i1: int,
    *,
    criterion4_tol: float = CRITERION4_TOL_DEFAULT,
) -> RouquerolWindow:
    """Check the instrument Starting/End point window against Rouquerol."""
    p_rel = np.asarray(p_rel, dtype=float)
    n = np.asarray(n, dtype=float)
    win = evaluate_window(p_rel, n, i0, i1, criterion4_tol=criterion4_tol)
    if win is None:
        raise ValueError("Instrument BET window is too short or non-physical.")
    return win


def format_rouquerol_report(result: dict, sample_name: str = "Sample") -> str:
    best: Optional[RouquerolWindow] = result["best"]
    lines = [
        f"Rouquerol BET range — {sample_name}",
        f"  candidates scanned : {result['n_candidates']}",
        f"  valid windows      : {result['n_valid']}",
        f"  selection          : {result['selection_rule']}",
    ]
    if best is None:
        lines.append("  No usable window found.")
        return "\n".join(lines)
    flag = "PASS" if best.valid else "FAIL — no fully consistent window; showing best compromise"
    lines.extend(
        [
            f"  status             : {flag}",
            f"  p/p0 window        : {best.p_min:.4f} – {best.p_max:.4f}  ({best.n_points} points)",
            f"  S_BET              : {best.S_BET:.3f} m² g⁻¹",
            f"  Vm                 : {best.Vm:.4f}",
            f"  C                  : {best.C:.2f}",
            f"  R²                 : {best.R2:.6f}",
            f"  C1 C > 0           : {best.c1_C_positive}",
            f"  C2 n(1−p/p0) ↑     : {best.c2_n1mp_increasing}",
            f"  C3 nm in range     : {best.c3_nm_in_range}  (p_m,exp = {best.pm_exp:.4f})",
            f"  C4 1/(√C+1) match  : {best.c4_pm_consistency}  (p_m,th = {best.pm_theory:.4f})",
        ]
    )
    return "\n".join(lines)
