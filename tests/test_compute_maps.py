from bggcohomology.compute_maps import BGGMapSolver
from bggcohomology.bggcomplex import BGGComplex
from bggcohomology.weight_set import WeightSet

import pytest

@pytest.mark.parametrize("root_system", ["A2", "G2", "B2"])
@pytest.mark.parametrize("mu", [(0, 0), (1, 2), (2, 2)])
def test_compute_maps(root_system, mu):
    bgg = BGGComplex(root_system)
    ws = WeightSet.from_bgg(bgg)
    mu = ws.make_dominant(mu)[0] # make sure weight is dominant
    mapsolver= BGGMapSolver(bgg,mu)
    mapsolver.solve()
    assert mapsolver.check_maps()==True