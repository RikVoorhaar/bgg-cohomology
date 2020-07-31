from bggcohomology.quantum_center import *
from bggcohomology.bggcomplex import BGGComplex
from bggcohomology.la_modules import *

import pytest

@pytest.mark.parametrize("root_system", ["A2", "G2", "B2", "A3"])
def test_total_trivial_dim(root_system):
    """Total dimension of principal block is given by (h+1)**r, with h coxeter number
    and r the rank.
    """
    BGG = BGGComplex(root_system)
    mu = (0,)*BGG.rank
    total_betti = 0
    for a,b,i,j,k in all_abijk(BGG):
        coker = Eijk_basis(BGG,j,k)
        cohom = BGGCohomology(BGG, Mjk(BGG,j,k),coker=coker)
        total_betti+=cohom.betti_number(cohom.cohomology(i, mu=mu))
    theoretical_dim = (2*len(BGG.neg_roots)/BGG.rank+1)**BGG.rank
    assert theoretical_dim == total_betti