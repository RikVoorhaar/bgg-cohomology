from bggcohomology.weight_set import WeightSet
from bggcohomology.bggcomplex import BGGComplex

import pytest


@pytest.mark.parametrize("root_system", ["A2", "G2", "B2"])
@pytest.mark.parametrize("mu", [(0, 0), (1, 2), (1, 1)])
@pytest.mark.parametrize("w", ["", "1", "12"])
@pytest.mark.parametrize("from_bgg", [True, False])
def test_init_weight_set(root_system, mu, from_bgg, w):
    if from_bgg:
        bgg = BGGComplex(root_system)
        ws = WeightSet.from_bgg(bgg)
    else:
        ws = WeightSet(root_system)
    assert ws.weight_to_tuple(ws.tuple_to_weight(mu)) == mu

    def sage_dot_action(w, mu):
        w_sage = ws.weyl_dic[w]
        mu_sage = ws.tuple_to_weight(mu)
        new_mu = w_sage.action(mu_sage + ws.rho) - ws.rho
        return ws.weight_to_tuple(new_mu)

    assert sage_dot_action(w, mu) == tuple(ws.dot_action(w, mu))


def test_compute_weights():
    ws = WeightSet("A3")
    all_weights = ws.compute_weights([(0, 0, 0), (1, 2, 1), (0, -1, 1), (0, 2, 0)])
    assert len(all_weights) == 3
    assert all_weights[0] == ((0, 0, 0), (0, 0, 0), 0)
    assert all_weights[1] == ((1, 2, 1), (1, 2, 1), 0)
    assert all_weights[2] == ((0, 2, 0), (1, 2, 1), 2)


def test_highest_weight_rep():
    ws = WeightSet("A1")
    assert ws.highest_weight_rep_dim((0,)) == 1
    assert ws.highest_weight_rep_dim((100,)) == 201

    ws = WeightSet("A3")
    assert ws.highest_weight_rep_dim((0, 0, 0)) == 1
    assert ws.highest_weight_rep_dim((1, 1, 1)) == 15
