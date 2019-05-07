from numpy import array,int16,zeros,around,greater_equal,array_equal
from sympy.utilities.iterables import multiset_partitions
from scipy.sparse import csr_matrix
#from scipy.sparse.linalg import spsolve,lsmr
from scipy.linalg import solve,lstsq
from time import time

class BGGMapSolver:
    """A class encoding the methods to compute all the maps in the BGG complex"""
    def __init__(self,BGG,root):
        self.BGG= BGG

        self.action_dic = {red_word: array(BGG.dot_action(w, root)) for red_word, w in
                      BGG.reduced_word_dic.items()}
        self.maps = {}
        self._compute_initial_maps()
        self.problem_dic = {}

        self.timer = {'linalg':0,'mult':0,'vect':0,'basis':0,'index':0}

    def _compute_initial_maps(self):
        """If the difference the dot action between source and target vertex is a multiple of a simple root,
        then the map must be of form f_i^k, where k is the multiple of the simple root,
        and i is the index of the simple root"""
        for s, t in self.BGG.arrows:
            diff = self.action_dic[s] - self.action_dic[t]
            if len(diff.nonzero()[0]) == 1: #of only one element of the dot action difference is non-zero...
                i = diff.nonzero()[0][0] + 1
                self.maps[(s, t)] = self.BGG.PBW(self.BGG.LA.f(i)) ** sum(diff)

    def _get_available_problems(self):
        """Find all the admissible cycles in the BGG graph where we know three out of the four maps,
        for each such cycle, create a dictionary containing all the information needed to solve
        for the remaining fourth map. We only store the problem of lowest total degree for any undetermined edge."""
        for c in self.BGG.cycles:
            edg = (c[0:2], c[1:3], c[4:2:-1], c[3:1:-1]) #unpack cycle into its four edges
            mask = [e in self.maps for e in edg]
            if sum(mask) == 3: #if we know three of the maps...
                problem={}
                problem['tot_deg'] = sum(self.action_dic[c[0]] - self.action_dic[c[2]])

                if mask.index(False) == 0:
                    problem['edge']=edg[0]
                    problem['deg']=self.action_dic[edg[0][0]] - self.action_dic[edg[0][1]]
                    problem['side']='left'
                    problem['known_LHS']=self.maps[edg[1]]
                    problem['RHS']=self.maps[edg[2]] * self.maps[edg[3]]
                if mask.index(False) == 1:
                    problem['edge']=edg[1]
                    problem['deg']=self.action_dic[edg[1][0]] - self.action_dic[edg[1][1]]
                    problem['side']='right'
                    problem['known_LHS']=self.maps[edg[0]]
                    problem['RHS']=self.maps[edg[2]] * self.maps[edg[3]]
                if mask.index(False) == 2:
                    problem['edge']=edg[2]
                    problem['deg']=self.action_dic[edg[2][0]] - self.action_dic[edg[2][1]]
                    problem['side']='left'
                    problem['known_LHS']=self.maps[edg[3]]
                    problem['RHS']=self.maps[edg[0]] * self.maps[edg[1]]
                if mask.index(False) == 3:
                    problem['edge']=edg[3]
                    problem['deg']=self.action_dic[edg[3][0]] - self.action_dic[edg[3][1]]
                    problem['side']='right'
                    problem['known_LHS']=self.maps[edg[2]]
                    problem['RHS']=self.maps[edg[0]] * self.maps[edg[1]]

                #only store the problem if either we didn't have a problem for this edge,
                #or the problem we had for this edge was of higher degree
                current_edge=problem['edge']
                if current_edge in self.problem_dic:
                    existing_problem = self.problem_dic[current_edge]
                    if existing_problem['tot_deg']>problem['tot_deg']:
                        self.problem_dic[current_edge]=problem
                else:
                    self.problem_dic[current_edge] = problem

    def problems(self):
        """Generator yielding problems. Once we run out of problems, we go through all the cycles to look for more problems.
        If we run out then, it means we found all the maps"""
        while True:
            if (len(self.problem_dic)) == 0:
                self._get_available_problems()
                if len(self.problem_dic) ==0:
                    break

            _, problem = self.problem_dic.popitem()
            yield problem

    def solve(self):
        """Iterate over all the problems to find all the maps, and return the result"""
        for problem in self.problems():
            self.solve_problem(problem)
        return self.maps

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

    def multidegree_to_root_sum(self,deg):
        queue = [(deg, [0])]
        output = []
        while len(queue) > 0:
            vect, partition = queue.pop()
            for index in range(
                    partition[-1], len(self.BGG.neg_roots)):
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
            root = sum(-self.BGG.lattice.alpha()[i + 1] * int(n) for i, n in enumerate(self.BGG.neg_roots[index]))
            output *= self.BGG.PBW_alg_gens[root]
        return output

    @staticmethod
    def vectorize_polynomial(polynomial,monomial_to_index):
        """given a dictionary of monomial->index, turn a polynomial into a vector"""
        coeffs = polynomial.monomial_coefficients()
        vector = zeros(len(monomial_to_index), dtype=int16)
        for monomial, coefficient in coeffs.items():
            vector[monomial_to_index[str(monomial)]] = coefficient
        return vector

    @staticmethod
    def vectorize_polynomial_list(polynomial_list,monomial_to_index):
        """given a dictionary of monomial->index, turn a list of polynomials into a sparse matrix.
        Currently not used."""
        row = []
        col = []
        data = []
        for row_num, polynomial in enumerate(polynomial_list):
            for monomial, coefficient in polynomial.monomial_coefficients().items():
                row.append(row_num)
                col.append(monomial_to_index[str(monomial)])
                data.append(coefficient)
        return csr_matrix((data, (row, col)), shape=(len(monomial_to_index), len(polynomial_list)), dtype=int16)

    def solve_problem(self,problem):
        """solve the division problem in PBW basis"""
        t = time()
        #basis = self.compute_PBW_basis_multidegree(problem['deg'])
        basis = [self.partition_to_PBW(partition) for partition in self.multidegree_to_root_sum(problem['deg'])]
        self.timer['basis']+=time()-t

        t =time()
        if problem['side'] == 'right':
            LHS = [problem['known_LHS'] * p for p in basis]
        if problem['side'] == 'left':
            LHS = [p * problem['known_LHS'] for p in basis]
        self.timer['mult']+=time()-t


        t=time()
        monomial_to_index = {}
        i = 0
        for l in LHS:
            for monomial in l.monomials():
                if str(monomial) not in monomial_to_index:
                    monomial_to_index[str(monomial)] = i
                    i += 1
        self.timer['index']=time()-t

        t=time()
        A = array([self.vectorize_polynomial(p, monomial_to_index) for p in LHS]).T
        b = self.vectorize_polynomial(problem['RHS'], monomial_to_index)
        self.timer['vect']+=time()-t

        t = time()
        if A.shape[0] == A.shape[1]:
            sol = around(solve(A, b)).astype(int16)
        else:
            sol = lstsq(A, b)
            sol = around(sol[0]).astype(int16) #without the 'around' type conversion goes wrong for whatever reason

        output = sum(int(c) * basis[i] for i, c in enumerate(sol))
        self.timer['linalg']+=time()-t

        self.maps[problem['edge']] = output

    def check_maps(self):
        """Check for each cycle whether it commutes, for debugging purposes"""
        problems_found = False
        for c in self.BGG.cycles:
            edg = (c[0:2], c[1:3], c[4:2:-1], c[3:1:-1])
            if self.maps[edg[0]]*self.maps[edg[1]]!=self.maps[edg[2]]*self.maps[edg[3]]:
                print("Problem found at cycle",c)
                problems_found=True
        if not problems_found:
            print("checked %d cycles, with no problems found!"%len(self.BGG.cycles))