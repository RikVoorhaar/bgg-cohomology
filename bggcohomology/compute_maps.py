"""
Compute the maps in the BGG complex.

Uses a PBW basis for the universal envoloping algebra of n together with some basic linear algebra.
Works for any dominant weight.
"""


from numpy import array, int16, zeros, around, greater_equal, array_equal


from sage.matrix.constructor import matrix

# from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.rational import Rational
from sage.modules.free_module_element import vector


class BGGMapSolver:
    """A class encoding the methods to compute all the maps in the BGG complex"""

    def __init__(self, BGG, weight, pbar=None, cached_results=None):
        self.BGG = BGG

        self.pbar = pbar

        self.action_dic = {
            w: BGG.alpha_sum_to_array(BGG.fast_dot_action(w, weight))
            for w in BGG.reduced_words
        }
        self.max_len = max(len(v) for v in self.action_dic.keys())

        if cached_results is not None:
            self.maps = cached_results
        else:
            self.maps = dict()
        self.num_trivial_maps = self._compute_initial_maps()
        self.n_non_trivial_maps = len(self.BGG.arrows) - self.num_trivial_maps
        self.problem_dic = dict()

    def _compute_initial_maps(self):
        """If the difference the dot action between source and target vertex is a multiple of a simple root,
        then the map must be of form f_i^k, where k is the multiple of the simple root,
        and i is the index of the simple root"""
        num_trivial_maps = 0
        for s, t in self.BGG.arrows:
            diff = self.action_dic[s] - self.action_dic[t]
            if (
                len(diff.nonzero()[0]) == 1
            ):  # of only one element of the dot action difference is non-zero...
                i = diff.nonzero()[0][0] + 1
                self.maps[(s, t)] = self.BGG.PBW(self.BGG.LA.f(i)) ** sum(diff)
                num_trivial_maps += 1
        return num_trivial_maps

    def _get_available_problems(self, final_column=None):
        """Find all the admissible cycles in the BGG graph where we know three out of the four maps,
        for each such cycle, create a dictionary containing all the information needed to solve
        for the remaining fourth map. We only store the problem of lowest total degree for any undetermined edge."""
        for c in self.BGG.cycles:
            edg = (
                c[0:2],
                c[1:3],
                c[4:2:-1],
                c[3:1:-1],
            )  # unpack cycle into its four edges
            mask = [e in self.maps for e in edg]

            if sum(mask) == 3:  # if we know three of the maps...
                problem = dict()
                problem["tot_deg"] = sum(self.action_dic[c[0]] - self.action_dic[c[2]])
                index_unknown_edg = mask.index(False)
                problem["edge"] = edg[index_unknown_edg]

                # only store the problem if either we didn't have a problem for this edge,
                # or the problem we had for this edge was of higher degree
                current_edge = problem["edge"]
                if current_edge in self.problem_dic:
                    best_degree = self.problem_dic[current_edge]["tot_deg"]
                    if problem["tot_deg"] < best_degree:
                        problem_viable = True
                    else:
                        problem_viable = False
                else:
                    problem_viable = True

                if (
                    final_column is not None
                ):  # Only solve problems where the edge ends in the target column
                    if len(current_edge[-1]) != final_column:
                        problem_viable = False

                if problem_viable:
                    if index_unknown_edg == 0:
                        problem["deg"] = (
                            self.action_dic[edg[0][0]] - self.action_dic[edg[0][1]]
                        )
                        problem["deg_RHS"] = (
                            -self.action_dic[edg[3][1]] + self.action_dic[edg[2][0]]
                        )
                        problem["side"] = "left"
                        problem["known_LHS"] = self.maps[edg[1]]
                        problem["RHS"] = self.maps[edg[3]] * self.maps[edg[2]]
                    if index_unknown_edg == 1:
                        problem["deg"] = (
                            self.action_dic[edg[1][0]] - self.action_dic[edg[1][1]]
                        )
                        problem["deg_RHS"] = (
                            -self.action_dic[edg[3][1]] + self.action_dic[edg[2][0]]
                        )
                        problem["side"] = "right"
                        problem["known_LHS"] = self.maps[edg[0]]
                        problem["RHS"] = self.maps[edg[3]] * self.maps[edg[2]]
                    if index_unknown_edg == 2:
                        problem["deg"] = (
                            self.action_dic[edg[2][0]] - self.action_dic[edg[2][1]]
                        )
                        problem["deg_RHS"] = (
                            -self.action_dic[edg[1][1]] + self.action_dic[edg[0][0]]
                        )
                        problem["side"] = "left"
                        problem["known_LHS"] = self.maps[edg[3]]
                        problem["RHS"] = self.maps[edg[1]] * self.maps[edg[0]]
                    if index_unknown_edg == 3:
                        problem["deg"] = (
                            self.action_dic[edg[3][0]] - self.action_dic[edg[3][1]]
                        )
                        problem["deg_RHS"] = (
                            -self.action_dic[edg[1][1]] + self.action_dic[edg[0][0]]
                        )
                        problem["side"] = "right"
                        problem["known_LHS"] = self.maps[edg[2]]
                        problem["RHS"] = self.maps[edg[1]] * self.maps[edg[0]]

                    self.problem_dic[current_edge] = problem

    def solve(self, column=None):
        """Iterate over all the problems to find all the maps, and return the result"""

        # Iterate in the opposite direction if we only need a column on the far side of the middle.
        if (column is not None) and (column > self.max_len / 2):
            columns = range(column, self.max_len + 1)[::-1]
        else:
            if column is not None:
                columns = range(column + 2)
            else:
                columns = range(self.max_len + 1)

        # Configure the progress bar with the right number of maps.
        if self.pbar is not None:
            if column is None:
                self.pbar.reset(total=self.n_non_trivial_maps)
            else:
                tot = 0
                for c in columns:
                    for s, t in self.BGG.arrows:
                        if (len(t) == c) and ((s, t) not in self.maps):
                            tot += 1
                self.pbar.reset(total=tot)

        for c in columns:
            self._get_available_problems(final_column=c)
            for problem in self.problem_dic.values():
                self.solve_problem(problem)
                if self.pbar is not None:
                    self.pbar.update()

        return self.maps

    def multidegree_to_root_sum(self, deg):
        """Compute a PBW basis of a given multi-degree"""
        queue = [(deg, [0])]
        output = []
        while len(queue) > 0:
            vect, partition = queue.pop()
            for index in range(partition[-1], len(self.BGG.neg_roots)):
                root = self.BGG.neg_roots[index]
                if all(greater_equal(vect, root)):
                    if array_equal(vect, root):
                        output.append(partition[1:] + [index])
                    else:
                        queue.append((vect - root, partition + [index]))
        return output

    def partition_to_PBW(self, partition):
        output = 1
        for index in partition:
            root = sum(
                -self.BGG.lattice.alpha()[i + 1] * int(n)
                for i, n in enumerate(self.BGG.neg_roots[index])
            )
            output *= self.BGG.PBW_alg_gens[root]
        return output

    def monomial_to_tuple(self, monomial):
        return tuple(self.BGG.alpha_to_index[r] for r in monomial.to_word_list())

    def vectorize_polynomial(self, polynomial, target_basis):
        """given a dictionary of monomial->index, turn a polynomial into a vector"""
        coeffs = polynomial.monomial_coefficients()

        vectorized = vector(QQ, len(target_basis))
        for monomial, coefficient in coeffs.items():
            key = self.monomial_to_tuple(monomial)
            vectorized[target_basis[key]] = coefficient
        return vectorized

    def vectorize_polynomials_list(self, polynomial_list, target_basis):
        vectorized = matrix(QQ, len(polynomial_list), len(target_basis))
        for row_index, polynomial in enumerate(polynomial_list):
            for monomial, coefficient in polynomial.monomial_coefficients().items():
                key = self.monomial_to_tuple(monomial)
                vectorized[row_index, target_basis[key]] = coefficient
        return vectorized

    def solve_problem(self, problem):
        """solve the division problem in PBW basis"""
        basis = [
            self.partition_to_PBW(partition)
            for partition in self.multidegree_to_root_sum(problem["deg"])
        ]

        if problem["side"] == "right":
            LHS = [p * problem["known_LHS"] for p in basis]
        if problem["side"] == "left":
            LHS = [problem["known_LHS"] * p for p in basis]

        target_basis = {
            tuple(partition): i
            for i, partition in enumerate(
                self.multidegree_to_root_sum(problem["deg_RHS"])
            )
        }

        A = self.vectorize_polynomials_list(LHS, target_basis)
        b = self.vectorize_polynomial(problem["RHS"], target_basis)

        sol = A.T.solve_right(b)

        output = sum(Rational(c) * basis[i] for i, c in enumerate(sol))

        self.maps[problem["edge"]] = output

    def dual_edge(self, edge):
        return self.BGG.dual_words[edge[1]], self.BGG.dual_words[edge[0]]

    @staticmethod
    def get_monomial_to_index(LHS):
        """deprecated"""
        monomial_to_index = dict()
        i = 0
        for l in LHS:
            for monomial in l.monomials():
                if str(monomial) not in monomial_to_index:
                    monomial_to_index[str(monomial)] = i
                    i += 1
        return monomial_to_index

    def check_maps(self):
        """Checks whether the squares commute. Returns True if there are no problems, False otherwise."""
        for c in self.BGG.cycles:
            edg = (c[0:2], c[1:3], c[4:2:-1], c[3:1:-1])
            if (
                self.maps[edg[1]] * self.maps[edg[0]]
                != self.maps[edg[3]] * self.maps[edg[2]]
            ):
                return False
        else:
            return True
