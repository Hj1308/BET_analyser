"""
test_betsi_reference.py — regression test against the BETSI round-robin data.

``examples/betsi_HKUST-1.csv`` and ``examples/betsi_Zeolite-13X.csv`` are N2
(77 K) adsorption isotherms from the BETSI round-robin study (Osterrieth et
al., Adv. Mater. 2022, 34, 2201502, CC BY, github.com/fairen-group/betsi-gui).
The Rouquerol-consistent BET surface area computed here must reproduce the
published BETSI optimal BET areas.

Tolerance: the 2% relative tolerance reflects the round-robin's own spread —
independent labs analysing the same reference isotherm typically agree to
within a few percent, so 2% is the acceptance band within which the tool
reproduces the consensus value. It is an acceptance criterion, not a target to
tune toward; a failure is a finding.
"""

import os

import pytest

from bet_analysis import read_bet_xls
from rouquerol import select_bet_range

# Relative acceptance tolerance for S_BET vs the published BETSI value.
# Source: the BETSI round-robin reports between-laboratory agreement of a few
# percent on shared reference isotherms, so 2% is the band within which the
# Rouquerol selection must reproduce the published BETSI optimal BET area.
BETSI_S_BET_TOLERANCE = 0.02

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(os.path.dirname(_HERE), "examples")

# (filename, published BETSI optimal BET area in m²/g)
BETSI_CASES = [
    ("betsi_HKUST-1.csv", 1556.0),
    ("betsi_Zeolite-13X.csv", 833.0),
]


@pytest.mark.parametrize("filename, published_sbet", BETSI_CASES)
def test_rouquerol_sbet_matches_betsi(filename, published_sbet):
    path = os.path.join(_EXAMPLES, filename)
    data = read_bet_xls(path)

    result = select_bet_range(data["ads"][:, 0], data["ads"][:, 1])
    best = result["best"]
    assert best is not None, f"{filename}: no Rouquerol window was found"

    rel = abs(best.S_BET - published_sbet) / published_sbet
    assert rel <= BETSI_S_BET_TOLERANCE, (
        f"{filename}: Rouquerol S_BET = {best.S_BET:.2f} m²/g differs from the "
        f"published BETSI value ({published_sbet} m²/g) by {rel:.3%} — outside "
        f"the {BETSI_S_BET_TOLERANCE:.0%} tolerance."
    )
