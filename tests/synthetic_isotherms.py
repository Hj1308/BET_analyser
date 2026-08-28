"""
synthetic_isotherms.py — physically-grounded N2 isotherms with known IUPAC type.

Generated from closed-form adsorption equations, NOT hand-placed points, so the
"expected" classification follows from the physics rather than from a guess:

  Type I(a)/I(b) : Langmuir monolayer saturation (genuine plateau)
  Type II/III    : BET / BDDT n-layer multilayer  (Brunauer et al., J. Am. Chem.
                   Soc. 62, 1723 (1940)) — Type II is C >> 1, Type III is C < 1
  Type IV/V      : multilayer + capillary-condensation step + hysteresis loop
  Type VI        : superposed stepwise layer transitions

Reference for the target classification:
  Thommes, M. et al., Pure Appl. Chem. 87, 1051-1069 (2015).

These exist because no measured sample in tests/ reaches a saturation plateau,
so Types I, II and VI are otherwise never exercised.

NOTE on Type II: IUPAC defines Type II as *unrestricted* monolayer-multilayer
adsorption, i.e. uptake rises without limit as p/p0 -> 1. A genuine Type II
therefore has no high-pressure plateau. See test_isotherm_classification.py.
"""
import numpy as np

def _grid(n_lo=14, n_hi=18):
    """p/p0 grid: dense below 0.1 (micropore region), then regular."""
    lo = np.logspace(np.log10(1e-4), np.log10(0.1), n_lo, endpoint=False)
    hi = np.linspace(0.1, 0.995, n_hi)
    return np.unique(np.concatenate([lo, hi]))

def bet(x, vm, C, n=None):
    """BET isotherm. n=None -> infinite layers (diverges at x->1);
    n=int -> BDDT n-layer form, finite everywhere (Brunauer et al. 1940)."""
    if n is None:
        return vm * C * x / ((1 - x) * (1 + (C - 1) * x))
    num = 1 - (n + 1) * x**n + n * x**(n + 1)
    den = 1 + (C - 1) * x - C * x**(n + 1)
    return vm * C * x / (1 - x) * num / den

def langmuir(x, vm, b):
    """Langmuir monolayer saturation -> genuine plateau."""
    return vm * b * x / (1 + b * x)

def step(x, x0, width, height):
    """Capillary condensation step (sigmoid in log-pressure)."""
    return height / (1 + np.exp(-(np.log(x) - np.log(x0)) / width))

# ---------------------------------------------------------------- generators
def type_Ia(x):
    # ultra-micropores: saturates below p/p0 = 0.01
    return langmuir(x, 120.0, 4000.0)

def type_Ib(x):
    # wider micropores: knee extends to ~0.1
    return langmuir(x, 120.0, 120.0) + 8.0 * x

def type_II(x):
    # non-porous, strong interaction. NOTE: rises without limit as x->1
    return bet(x, 40.0, 120.0, n=9)

def type_III(x):
    # weak adsorbate-adsorbent interaction, C < 1 -> convex throughout
    return bet(x, 40.0, 0.35, n=9)

def type_IV(x):
    # mesoporous: BET multilayer + capillary condensation + saturation plateau
    return bet(x, 30.0, 90.0, n=5) * (x < 0.45) + \
           (bet(np.minimum(x, 0.45), 30.0, 90.0, n=5) + step(x, 0.55, 0.09, 95.0)) * (x >= 0.45)

def type_V(x):
    # weak interaction + mesoporosity
    return bet(x, 30.0, 0.5, n=5) * (x < 0.45) + \
           (bet(np.minimum(x, 0.45), 30.0, 0.5, n=5) + step(x, 0.60, 0.09, 110.0)) * (x >= 0.45)

def type_VI(x):
    # stepwise multilayer on a uniform surface
    v = 6.0 * x
    for i, x0 in enumerate([0.12, 0.38, 0.68]):
        v = v + step(x, x0, 0.035, 30.0)
    return v

def desorption(x, v_ads, shift, close_at):
    """Desorption branch: shifted to lower p/p0 above the closure point."""
    m = x >= close_at
    xd = x[m]
    vd = np.interp(np.clip(xd / (1 - shift * (1 - (xd - close_at) / (1 - close_at))), 0, 1),
                   x, v_ads)
    vd = np.maximum(vd, np.interp(xd, x, v_ads))
    return np.column_stack([xd, vd])


def desorption_closed(x, v_ads, amplitude, close_at):
    """H1 desorption branch that closes at close_at.

    Separation is added in the *volume* domain (vd = v_ads + A·sin(pi·u))
    rather than the *pressure* domain, so the two branches stay nearly
    parallel: condensation and evaporation keep the same steepness, as
    expected for a single pore-size distribution.  The bump vanishes at both
    ends, so the loop is closed at the bottom and the top.
    """
    m = x >= close_at
    xd = x[m]
    u = (xd - close_at) / (1.0 - close_at)
    vd = v_ads[m] + amplitude * np.sin(np.pi * u)
    return np.column_stack([xd, vd])

CASES = {
    "TypeIa_noHyst":  (type_Ia,  None),
    "TypeIb_noHyst":  (type_Ib,  None),
    "TypeII_noHyst":  (type_II,  None),
    "TypeIII_noHyst": (type_III, None),
    "TypeVI_noHyst":  (type_VI,  None),
    # The loop width (shift) is meant to be representative of an H1 loop and
    # was deliberately NOT tuned against the detection threshold; with
    # shift = 0.15 the normalised loop area (~0.033) sits well clear of it.
    "TypeIV_H1":      (type_IV,  (0.15, 0.45)),
    "TypeV_H2":       (type_V,   (0.10, 0.50)),
}

def build(name):
    f, hyst = CASES[name]
    x = _grid()
    v = f(x)
    ads = np.column_stack([x, v])
    des = np.empty((0, 2)) if hyst is None else desorption(x, v, *hyst)
    return ads, des

EXPECTED = {
    "TypeIa_noHyst":  "Type I(a)",
    "TypeIb_noHyst":  "Type I(b)",
    "TypeII_noHyst":  "Type II",
    "TypeIII_noHyst": "Type III",
    "TypeVI_noHyst":  "Type VI",
    "TypeIV_H1":      "Type IV(a)",
    "TypeV_H2":       "Type V",
}


def negligible_desorption(ads, excess=0.001, above=0.6):
    """A desorption branch that is physically NOT a hysteresis loop.

    Sits `excess` (default 0.1%) above the adsorption branch, giving a
    normalised loop area of ~4e-4 — far below any sane hysteresis threshold.
    Used to probe whether `has_hyst` responds to loop area or merely to the
    presence of a non-empty array.
    """
    m = ads[:, 0] > above
    return np.column_stack([ads[m, 0], ads[m, 1] * (1.0 + excess)])


if __name__ == "__main__":
    for n in CASES:
        a, d = build(n)
        print(f"{n:16s} expect={EXPECTED[n]:10s} ads={len(a):3d} des={len(d):3d}  "
              f"Va.max={a[:,1].max():8.2f}")
