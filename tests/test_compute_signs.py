from bggcohomology.bggcomplex import BGGComplex
from bggcohomology.compute_signs import compute_signs

import pytest


def check_signs(bgg, signs):
    """Check if product of signs for each cycle is -1"""
    for c in bgg.cycles:
        if (
            signs[(c[0], c[1])]
            * signs[(c[1], c[2])]
            * signs[(c[3], c[2])]
            * signs[(c[4], c[3])]
            != -1
        ):
            return False
    return True


@pytest.mark.parametrize("root_system", ["A2", "G2", "B2", "A3", "A4"])
def test_signs(root_system):
    bgg = BGGComplex(root_system)
    signs = compute_signs(bgg)
    assert check_signs(bgg, signs) == True
