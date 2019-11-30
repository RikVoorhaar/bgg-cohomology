"""
Implement the BGG complex

Uses compute_maps.py and compute_signs.py to obtain the maps with associated signs of the BGG complex.
To use the BGG complex to compute cohomology we can use fast_module.
"""

from itertools import groupby,chain

import os
import pickle

#from sage.all import *
from sage.rings.rational_field import QQ
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.algebras.lie_algebras.lie_algebra import LieAlgebra
from sage.combinat.root_system.weyl_group import WeylGroup
from sage.graphs.digraph import DiGraph
from sage.modules.free_module_element import vector

from numpy import array

from compute_signs import compute_signs
from compute_maps import BGGMapSolver

from pbw import PoincareBirkhoffWittBasis

from fast_module import WeightSet

from collections import defaultdict

from IPython.display import display, Math, Latex


class BGGComplex:
    """A class encoding all the things we need of the BGG complex"""
    def __init__(self, root_system, pickle_directory = None):
        self.root_system = root_system
        self.W = WeylGroup(root_system)
        self.domain = self.W.domain()
        self.LA = LieAlgebra(QQ, cartan_type=root_system)
        self.PBW = PoincareBirkhoffWittBasis(self.LA, None, 'PBW', cache_degree=5)
        #self.PBW = self.LA.pbw_basis()
        self.PBW_alg_gens = self.PBW.algebra_generators()
        self.lattice = self.domain.root_system.root_lattice()
        self.S = self.W.simple_reflections()
        self.T = self.W.reflections()

        self._compute_weyl_dictionary()
        self._construct_BGG_graph()

        self.find_cycles()

        self.simple_roots = self.domain.simple_roots().values()
        self.rank = len(self.simple_roots)

        # for PBW computations we need to put the right order on the negative roots.
        # This order coincides with that of the sagemath source code.
        lie_alg_order = {k: i for i, k in enumerate(self.LA.basis().keys())}
        ordered_roots = sorted([self.weight_to_alpha_sum(r) for r in self.domain.negative_roots()],
                               key=lambda rr: lie_alg_order[rr])
        self.neg_roots = [-self.alpha_sum_to_array(r) for r in ordered_roots]
        # self.neg_roots = sorted([-array(self._weight_to_tuple(r)) for r in self.domain.negative_roots()],
        #                         key=lambda l: (sum(l), tuple(l)))

        self.alpha_to_index = {self.weight_to_alpha_sum(-self._tuple_to_weight(r)):i for i,r in enumerate(self.neg_roots)}
        self.zero_root = self.domain.zero()

        self.pickle_directory = pickle_directory
        if pickle_directory is None:
            self.pickle_maps = False
        else:
             self.pickle_maps = True

        if self.pickle_maps:
            self._maps = self.read_maps()
        else:
            self._maps = dict()

        self.rho = self.domain.rho()

        self._action_dic = dict()
        for s, w in self.reduced_word_dic.items():
            self._action_dic[s] = {i: self.weight_to_alpha_sum(w.action(mu))
                                      for i, mu in dict(self.domain.simple_roots()).items()}
        self._rho_action_dic = dict()
        for s, w in self.reduced_word_dic.items():
            self._rho_action_dic[s] = self.weight_to_alpha_sum(w.action(self.rho)-self.rho)
        
    def _compute_weyl_dictionary(self):
        """Construct a dictionary enumerating all of the elements of the Weyl group.
        The keys are reduced words of the elements"""
        self.reduced_word_dic={''.join([str(s) for s in g.reduced_word()]):g for g in self.W}
        self.reduced_word_dic_reversed=dict([[v,k] for k,v in self.reduced_word_dic.items()])
        self.reduced_words = sorted(self.reduced_word_dic.keys(),key=len) #sort the reduced words by their length
        long_element = self.W.long_element()
        self.dual_words = {
            s: self.reduced_word_dic_reversed[long_element*w]
            for s, w in self.reduced_word_dic.items()
        } # the dual word is the word times the longest element

        self.column = defaultdict(list)
        max_len = 0
        for red_word in self.reduced_words:
            length = len(red_word)
            max_len=max(max_len, length)
            self.column[length] += [red_word]
        self.max_word_length = max_len

    def _construct_BGG_graph(self):
        """Find all the arrows in the BGG Graph.
        There is an arrow w->w' if len(w')=len(w)+1 and w' = t.w for some t in T."""
        self.arrows=[]
        for w in self.reduced_words:
            for t in self.T:
                product_word = self.reduced_word_dic_reversed[t*self.reduced_word_dic[w]]
                if len(product_word)==len(w)+1:
                    self.arrows+=[(w,product_word)]
        self.arrows = sorted(self.arrows,key=lambda t: len(t[0])) #sort the arrows by the word length
        self.graph= DiGraph(self.arrows)
    
    def plot_graph(self):
        """Create a pretty plot of the BGG graph. Each word length is encoded by a different color.
        Usage: _.plot_graph().plot()"""
        BGGVertices=sorted(self.reduced_words,key=len) 
        BGGPartition=[list(v) for k,v in groupby(BGGVertices,len)]

        BGGGraphPlot = self.graph.to_undirected().graphplot(partition=BGGPartition,vertex_labels=None,vertex_size=30)
        display(BGGGraphPlot.plot())

    def find_cycles(self):
        """Find all the admitted cycles in the BGG graph. An admitted cycle consists of two paths a->b->c and a->b'->c,
         where the word length increases by 1 each step. The cycles are returned as tuples (a,b,c,b',a)."""

        # only compute cycles if we haven't yet done so already
        # this isn't very pythonic, but it works
        try:
            self.cycles
        except AttributeError:
            # for faster searching, make a dictionary of pairs (v,[u_1,...,u_k]) where v is a vertex and u_i
            # are vertices such that there is an arrow v->u_i
            first = lambda x: x[0]
            second = lambda x: x[1]
            outgoing={k:map(second,v) for k,v in groupby(sorted(self.arrows,key=first),first)}
            # outgoing[max(self.reduced_words,key=lambda x: len(x))]=[]
            outgoing[self.reduced_word_dic_reversed[self.W.long_element()]]=[]

            # make a dictionary of pairs (v,[u_1,...,u_k]) where v is a vertex and u_i are vertices such that
            # there is an arrow u_i->v
            incoming={k:map(first,v) for k,v in groupby(sorted(self.arrows,key=second),second)}
            incoming['']=[]

            # enumerate all paths of length 2, a->b->c, where length goes +1,+1
            self.cycles=chain.from_iterable([[a+(v,) for v in outgoing[a[-1]]] for a in self.arrows])

            # enumerate all paths of length 3, a->b->c->b' such that b' != b,
            # b<b' in lexicographic order (to avoid duplicates) and length goes +1,+1,-1
            self.cycles=chain.from_iterable([[a+(v,) for v in incoming[a[-1]] if v > a[1]] for a in self.cycles])

            # enumerate all cycles of length 4, a->b->c->b'->a such that b'!=b and length goes +1,+1,-1,-1
            self.cycles=[a+(a[0],) for a in self.cycles if a[0] in incoming[a[-1]]]

        return self.cycles

    def compute_signs(self, force_recompute=False):
        """Computes signs for all the edges so that the product of signs around any admissible cycle is -1.
        Returns a dictionary with the edges as keys and the signs as values."""
        if not force_recompute:
            try:  # Not super pythonic, but alright
                return self.signs
            except AttributeError:
                pass

        self.signs = compute_signs(self)
        return self.signs

    def compute_maps(self, root, column=None, check=False, pbar=None):
        """For the given weight, compute the maps of the BGG complex"""

        # If the maps are not in the cache, compute them and cache the result
        if root in self._maps:
            cached_result = self._maps[root]
        else:
            cached_result = None
        MapSolver = BGGMapSolver(self,  root, pbar=pbar, cached_results=cached_result)
        self._maps[root] = MapSolver.solve(column=column)
        if check:
            maps_OK = MapSolver.check_maps()
            if not maps_OK:
                raise ValueError('For root %s the map solver produced something wrong' % root)

        if self.pickle_maps:
            self.store_maps()

        return self._maps[root]

    def read_maps(self):
        target_path = os.path.join(self.pickle_directory, self.root_system + r'_maps.pkl')
        try:
            with open(target_path, 'rb') as file:
                maps = pickle.load(file)
                return maps
        except IOError:
            return dict()

    def store_maps(self):
        target_path = os.path.join(self.pickle_directory, self.root_system + r'_maps.pkl')
        try:
            with open(target_path, 'wb') as file:
                pickle.dump(self._maps, file, pickle.HIGHEST_PROTOCOL)
        except IOError:
            pass

    def _weight_to_tuple(self,weight):
        """Decompose a weight into a tuple encoding the weight as a linear combination of the simple roots"""
        b=weight.to_vector()
        b=matrix(b).transpose()
        A=[list(a.to_vector()) for a in self.simple_roots]
        A=matrix(A).transpose()

        return tuple(A.solve_right(b).transpose().list())

    def weight_to_alpha_sum(self,weight):
        """Express a weight in the lattice as a linear combination of alpha[i]'s. These objects form the keys
        for elements of the Lie algebra, and for factors in the universal enveloping algebra."""
        tuple = self._weight_to_tuple(weight)
        alpha = self.lattice.alpha()
        zero = self.lattice.zero()
        return sum((int(c)*alpha[i+1] for i, c in enumerate(tuple)), zero)

    def alpha_sum_to_array(self,weight):
        output = array(vector(ZZ,self.rank))
        for i,c in weight.monomial_coefficients().items():
            output[i-1]=c
        return output

    def _tuple_to_weight(self,t):
        """Turn a tuple encoding a linear combination of simple roots back into a weight"""
        return sum(int(a)*b for a, b in zip(t,self.simple_roots))

    def dot_action(self,reflection,weight):
        """Compute the dot action of a reflection on a weight. The reflection should be an element of the Weyl group
        self.W and the weight should be given as a tuple encoding it as a linear combination of simple roots."""
        weight = self._tuple_to_weight(weight)
        new_weight= reflection.action(weight+self.rho)-self.rho
        return self._weight_to_tuple(new_weight)

    def is_dot_regular(self, mu):
        """Check if a weight is dot-regular by checking that it has trivial stabilizer"""
        for w in self.reduced_words[1:]:
            if self.fast_dot_action(w,mu)==mu: # Stabilizer is non-empty, mu is not dot regular
                return False
        else:
            return True

    def make_dominant(self, mu):
        for w in self.reduced_words:
            new_mu = self.fast_dot_action(w,mu)
            if new_mu.is_dominant():
                return new_mu, w
        else:
            raise Exception('The weight %s can not be made dominant. Probably it is not dot-regular.' % mu)

    def compute_weights(self, weight_module):
        all_weights = weight_module.weight_dic.keys()

        regular_weights = []
        for mu in all_weights:
            if self.is_dot_regular(mu):
                mu_prime, w = self.make_dominant(mu)
                # mu_prime = self.weight_to_alpha_sum(mu_prime)
                # w = self.reduced_word_dic_reversed[w]
                regular_weights.append((mu, mu_prime, len(w)))
        return all_weights, regular_weights

    def fast_dot_action(self, w, mu):
        action = self._action_dic[w]
        mu_action = sum([action[i] * int(c) for i, c in mu.monomial_coefficients().items()], self.lattice.zero())
        return mu_action + self._rho_action_dic[w]

    def display_pbw(self, f, notebook=True):
        """Typesets an element of PBW of the universal enveloping algebra with LaTeX.
        Options: notebook = True; uses IPython display with math if True, otherwise just returns the LaTeX code."""

        map_string = []
        first_term = True
        for monomial, coefficient in f.monomial_coefficients().items():
            alphas = monomial.dict().items()

            if first_term:
                if coefficient < 0:
                    sign_string = '-'
                else:
                    sign_string = ''
                first_term = False
            else:
                if coefficient < 0:
                    sign_string = '-'
                else:
                    sign_string = '+'
            if abs(coefficient) == 1:
                coeff_string = ''
            else:
                coeff_string = str(abs(coefficient))
            term_strings = [sign_string + coeff_string]
            for alpha, power in alphas:
                if power > 1:
                    power_string = r'^{' + str(power) + r'}'
                else:
                    power_string = ''
                alpha_string = ''.join(str(k) * int(-v) for k, v in alpha.monomial_coefficients().items())
                term_strings.append(r'f_{' + alpha_string + r'}' + power_string)

            map_string.append(r'\,'.join(term_strings))
        if notebook:
            display(Math(' '.join(map_string)))
        else:
            return ' '.join(map_string)

    def _display_map(self, arrow, f):
        """Displays a single arrow plus map of the BGG complex"""

        f_string = self.display_pbw(f, notebook=False)
        display(Math(r'\to'.join(arrow) + r',\,\,' + f_string))

    def display_maps(self, mu):
        """Displays all the maps of the BGG complex for a given mu, in appropriate order.
        mu is assumed to be a tuple encoding it as a linear combination of weights."""

        mu_weight = WeightSet(self).tuple_to_weight(mu)
        if not mu_weight.is_dominant():
            raise ValueError('The weight %s is not dominant.' % mu)
        maps = self.compute_maps(self.weight_to_alpha_sum(mu_weight))
        maps = sorted(maps.items(), key=lambda s: (len(s[0][0]), s[0]))
        for arrow, f in maps:
            self._display_map(arrow, f)