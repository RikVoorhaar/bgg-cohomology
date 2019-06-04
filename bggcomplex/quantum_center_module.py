from lie_algebra_module import *


class QuantumFactory(object):
    """Class for building the module M_jk, T_jk needed for computing the center of the small quantum group,
    among other things."""

    def __init__(self, BGG):
        self.BGG = BGG
        self.factory = LieAlgebraModuleFactory(BGG.LA)

        self.modules = dict()
        self.modules['g'] = self.factory.construct_module(subalgebra='g', action='adjoint')
        self.modules['n'] = self.factory.construct_module(subalgebra='n', action='adjoint')
        self.modules['u'] = self.factory.construct_module(subalgebra='u', action='coadjoint')
        self.modules['b'] = self.factory.construct_module(subalgebra='b', action='adjoint')

        self._init_phi()

    def phi(self, m):
        out = dict()
        for e in self.modules['u'].basis_keys:
            f_i = self.factory.dual_root_dict[e]
            coad_e_i = self.modules['u'].action(m, self.modules['u'].basis()[e])
            for e_j, c in coad_e_i.monomial_coefficients().items():
                out[TensorProduct(f_i, e_j)] = c
        return out

    def _init_phi(self):
        self.phi_image = [fm for fm in [self.phi(m) for m in self.modules['b'].lie_algebra_basis] if len(fm) > 0]
        bbasis = self.factory.basis['b']
        b_roots = [self.factory.string_to_root[b] for b in bbasis]
        b_lie_alg_basis = [self.modules['b'].lie_algebra_basis[br] for br in b_roots]
        self.phi_b_pairs = [(b, self.phi(m)) for b, m in zip(bbasis,b_lie_alg_basis)]

    def _M_r(self, j, k, r):
        u_part = self.modules['u'].symmetric_power(j + k / 2 - r)
        g_part = self.modules['g'].alternating_power(r)
        n_part = self.modules['n'].alternating_power(j - r)
        return LieAlgebraModule.tensor_product(u_part, g_part, n_part)

    def _M_module_list(self, j, k):
        return [((j + k / 2 - r, r, j - r), self._M_r(j, k, r)) for r in range(j + k / 2 + 1)]

    def M_module(self, j, k):
        return LieAlgebraModule.direct_sum(*[m for _, m in self._M_module_list(j, k)])

    def _b_insert(self, Mjk):
        new_basis = []
        for (num_u, num_g, num_n), module in Mjk:
            index = num_g + 1
            basis = module.basis_keys
            for b in self.factory.basis['b']:
                for m in basis:
                    if num_g == 0:
                        new_basis.append(DirectSum(index, m.replace(1, b)))
                    elif num_g == 1:
                        new_key = AlternatingProduct(b, m[1]).sort()
                        if new_key.parity() != 0:
                            new_basis.append(DirectSum(index, m.replace(1, new_key)))
                    else:
                        new_key = m[1].insert(b).sort()
                        if new_key.parity() != 0:
                            new_basis.append(DirectSum(index, m.replace(1, new_key)))
        return new_basis

    def _phi_insert(self, Mjk):
        new_basis = []
        for (num_u, num_g, num_n), module in Mjk:
            basis = module.basis_keys
            for m in basis[:]:
                for phi_dict in self.phi_image:
                    new_dict = dict()
                    for (key2, key1), coeff in phi_dict.items():
                        if num_u == 0:
                            new_key = key1
                        elif num_u == 1:
                            new_key = SymmetricProduct(key1, m[0])
                        else:
                            new_key = m[0].insert(key1)
                        new_m = m.replace(0, new_key)

                        if num_n == 0:
                            new_key = key2
                        elif num_n == 1:
                            new_key = AlternatingProduct(key2, m[-1])
                            coeff *= new_key.parity()
                            new_key.sort()
                        else:
                            new_key = m[-1].insert(key2)
                            coeff *= new_key.parity()
                            new_key.sort()
                        if coeff != 0:
                            new_m = new_m.replace(-1, new_key)
                            new_dict[new_m] = coeff
                    if len(new_dict) > 0:
                        new_dict = {DirectSum(num_g, k): v for k, v in new_dict.items()}
                        new_basis.append(new_dict)
        return new_basis

    def _insert_both(self, Mjk):
        new_basis = []
        for (num_u, num_g, num_n), module in Mjk:
            index = num_g + 1
            basis = module.basis_keys
            for b, phi_dict in self.phi_b_pairs:
                for m in basis:
                    new_dict = defaultdict(int)

                    # insert b part
                    if num_g == 0:
                        new_dict[DirectSum(index, m.replace(1, b))] += 1
                    elif num_g == 1:
                        new_key = AlternatingProduct(b, m[1])
                        if new_key.parity() != 0:
                            new_dict[DirectSum(index, m.replace(1, new_key.sort()))] += new_key.parity()
                    else:
                        new_key = m[1].insert(b)
                        if new_key.parity() != 0:
                            new_dict[DirectSum(index, m.replace(1, new_key.sort()))] += new_key.parity()

                    # insert phi part
                    for (key2, key1), coeff in phi_dict.items():
                        if num_u == 0:
                            new_key = key1
                        elif num_u == 1:
                            new_key = SymmetricProduct(key1, m[0])
                        else:
                            new_key = m[0].insert(key1)
                        new_m = m.replace(0, new_key)

                        if num_n == 0:
                            new_key = key2
                        elif num_n == 1:
                            new_key = AlternatingProduct(key2, m[-1])
                            coeff *= new_key.parity()
                            new_key.sort()
                        else:
                            new_key = m[-1].insert(key2)
                            coeff *= new_key.parity()
                            new_key.sort()

                        if coeff != 0 and new_key.parity() != 0:
                            new_m = new_m.replace(-1, new_key)
                            new_dict[DirectSum(num_g, new_m)] += coeff

                    if len(new_dict) > 0:
                        # new_dict = {DirectSum(num_g, k): v for k, v in new_dict.items()}
                        new_basis.append(dict(new_dict))
        return new_basis

    def T_spanning_set(self, j, k):
        Mjk = self._M_module_list(j - 1, k)
        #T_span = self._phi_insert(Mjk)
        #T_span += [{b: 1} for b in self._b_insert(Mjk)]
        T_span = self._insert_both(Mjk)
        return T_span

    def get_weight(self, obj):
        zero = self.factory.lattice.zero()
        if isinstance(obj, (TensorProduct, SymmetricProduct, AlternatingProduct)):
            return sum((self.get_weight(k) for k in obj.keys), zero)
        elif isinstance(obj, DirectSum):
            return self.get_weight(obj.key)
        elif isinstance(obj, str):
            if obj[0] in ('f', 'e'):
                return self.factory.string_to_root[obj]
            else:
                return zero
        else:
            return zero

    def weight_module(self, j, k):
        return WeightModuleWithRelations(self.BGG.LA.base_ring(), self.M_module(j, k),
                                         self.get_weight, self.T_spanning_set(j, k))
