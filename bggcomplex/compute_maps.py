from numpy import array,int16,zeros
from sympy.utilities.iterables import multiset_partitions
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class BGGMapSolver:
    """A class encoding the methods to compute all the maps in the BGG complex"""
    def __init__(self,BGG,root):
        self.BGG= BGG

        self.action_dic = {red_word: array(BGG.dot_action(w, root)) for red_word, w in
                      BGG.reduced_word_dic.items()}
        self.maps = {}
        self._compute_initial_maps()
        self.problems = []


    def _compute_initial_maps(self):
        """If the difference the dot action between source and target vertex is a multiple of a simple root,
        then the map must be of form f_i^k, where k is the multiple of the simple root,
        and i is the index of the simple root"""
        for s, t in self.BGG.arrows:
            diff = self.action_dic[s] - self.action_dic[t]
            if len(diff.nonzero()[0]) == 1: #of only one element of the dot action difference is non-zero...
                i = diff.nonzero()[0][0] + 1
                self.maps[(s, t)] = self.BGG.PBW(self.BGG.LA.f(i)) ** sum(diff)

    def get_available_problems(self):
        """Find all the admissible cycles in the BGG graph where we know three out of the four maps,
        Call the unknown map 'f', then for each such cycle make a tuple of form.
        (edge with f, multi-degree of f, 'left'/'right', g, h),
        where we want to solve fg=h if 'left' and fg=h if 'right'"""
        for c in self.BGG.cycles:
            edg = (c[0:2], c[1:3], c[4:2:-1], c[3:1:-1]) #unpack cycle into its four edges
            mask = [e in self.maps for e in edg]
            if sum(mask) == 3: #if we know three of the maps...
                if mask.index(False) == 0:
                    self.problems.append((edg[0], self.action_dic[edg[0][0]] - self.action_dic[edg[0][1]], 'left',
                                          self.maps[edg[1]], self.maps[edg[2]] * self.maps[edg[3]]))
                if mask.index(False) == 1:
                    self.problems.append((edg[1], self.action_dic[edg[1][0]] - self.action_dic[edg[1][1]], 'right',
                                          self.maps[edg[0]], self.maps[edg[2]] * self.maps[edg[3]]))
                if mask.index(False) == 2:
                    self.problems.append((edg[2], self.action_dic[edg[2][0]] - self.action_dic[edg[2][1]], 'left',
                                          self.maps[edg[3]], self.maps[edg[0]] * self.maps[edg[1]]))
                if mask.index(False) == 3:
                    self.problems.append((edg[3], self.action_dic[edg[3][0]] - self.action_dic[edg[3][1]], 'right',
                                          self.maps[edg[2]], self.maps[edg[0]] * self.maps[edg[1]]))


    def _is_nonzero(self,l):
        """returns True if a tuple of simple roots corresponds to a basis element of the Lie algebra, False if not.
        e.g. [1 1]->False, because[ f_1,f_1]=0, and [1 1 2]->True/False depending on whether [-2,-1,..] is in the lattice"""
        if len(l) == 1:
            return True
        if len(set(l)) == 1:
            return False
        if tuple(l) in self.BGG.allowed_tuples:
            return True
        return False

    def _prune(self,p):
        """Return True if all of the constituent list in input partition correspond to Lie algebra basis element,
        False otherwise"""
        for l in p:
            if not self._is_nonzero(l):
                return False
        return True

    def _list_to_PBW(self,l):
        """Given a list of simple roots (e.g. [1 1 2 3]) turn it into element PBW[-alpha[1]*2-alpha[2]-alpha[3]]"""
        root = sum(-self.BGG.lattice.alpha()[i] for i in l)
        return self.BGG.PBW.algebra_generators()[root]

    def _partition_to_PBW(self,p):
        """Given a partition of the multiset encoding the degree, compute the corresponding monomial in the PBW basis,
        e.g. [[2,3],[1,2]]->PBW[-alpha[2]-alpha[3]]*PBW[-alpha[1]-alpha[2]]"""
        pbws = [self._list_to_PBW(l) for l in p]
        return reduce(lambda x, y: x * y, pbws)

    def compute_PBW_basis_multidegree(self,deg):
        """Given a mutlidegree (e.g. [2,1,2]), compute a (PBW) basis of all the monomials in U(g) with this multidegree"""

        #turn multidegree in symmetric word, e.g. [2 1 2]->[1 1 2 3 3]
        sym = []
        for i, n in enumerate(deg):
            sym += ((i + 1,) * n)

        #compute all the multiset partitions of list, remove those which don't correspond to PBW basis elements
        parts = list(multiset_partitions(sym))
        parts = [p for p in parts if self._prune(p)]

        #sort elements of set of multiset partitions such that they are in the order sagemath's pbw_basis routine expects
        #it seems a<b if len(a)<len(b), and if len equal then a<b if a[0]>b[0]
        parts = [sorted(p, key=lambda l: (len(l), -l[0])) for p in parts]

        return [self._partition_to_PBW(p) for p in parts]

    @staticmethod
    def vectorize_polynomial(polynomial,monomial_to_index):
        coeffs = polynomial.monomial_coefficients()
        vector = zeros(len(monomial_to_index), dtype=int16)
        for monomial, coefficient in coeffs.items():
            vector[monomial_to_index[str(monomial)]] = coefficient
        return vector

    @staticmethod
    def vectorize_polynomial_list(polynomial_list,monomial_to_index):
        row = []
        col = []
        data = []
        for row_num, polynomial in enumerate(polynomial_list):
            for monomial, coefficient in polynomial.monomial_coefficients().items():
                col.append(row_num)
                row.append(monomial_to_index[str(monomial)])
                data.append(coefficient)
        return csr_matrix((data, (row, col)), shape=(len(polynomial_list), len(monomial_to_index)), dtype=int16)
    

    def solve_problem(self,problem):
        basis = self.compute_PBW_basis_multidegree(problem[1])
        if problem[2] == 'right':
            LHS = [problem[3] * p for p in basis]
        if problem[2] == 'left':
            LHS = [p * problem[3] for p in basis]
        monomial_to_index = {}
        i = 0
        for l in LHS:
            for monomial in l.monomials():
                if str(monomial) not in monomial_to_index:
                    monomial_to_index[str(monomial)] = i
                    i += 1
        sol = spsolve(self.vectorize_polynomial_list(LHS,monomial_to_index),
                      self.vectorize_polynomial(problem[4],monomial_to_index)
                      ).astype(int16)

        return sum(int(c) * basis[i] for i, c in enumerate(sol))