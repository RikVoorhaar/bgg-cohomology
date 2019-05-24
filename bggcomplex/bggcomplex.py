from itertools import groupby,chain

from sage.all import *

from numpy import array

from compute_signs import compute_signs
from compute_maps import BGGMapSolver


class BGGComplex:
    """A class encoding all the things we need of the BGG complex"""
    def __init__(self, root_system):
        self.W = WeylGroup(root_system)
        self.domain = self.W.domain()
        self.LA =  LieAlgebra(QQ, cartan_type=root_system)
        self.PBW = self.LA.pbw_basis()
        self.PBW_alg_gens = self.PBW.algebra_generators()
        self.lattice = self.domain.root_system.root_lattice()
        self.S = self.W.simple_reflections()
        self.T = self.W.reflections()
        
        self._compute_weyl_dictionary()
        self._construct_BGG_graph()
        
        self.rho = self.domain.rho()
        self.simple_roots = self.domain.simple_roots().values()
        self.neg_roots = sorted([-array(self._weight_to_tuple(r)) for r in self.domain.negative_roots()],
                                key=lambda l: (sum(l), tuple(l)))
        self.zero_root = self.domain.zero()
        # self.allowed_tuples = {(self._root_to_list(self._weight_to_tuple(r))) for r in self.W.domain().negative_roots()}
        
    def _compute_weyl_dictionary(self):
        """Construct a dictionary enumerating all of the elements of the Weyl group.
        The keys are recuced words of the elements"""
        self.reduced_word_dic={''.join([str(s) for s in g.reduced_word()]):g for g in self.W}
        self.reduced_word_dic_reversed=dict([[v,k] for k,v in self.reduced_word_dic.items()])
        self.reduced_words = sorted(self.reduced_word_dic.keys(),key=len) #sort the reduced words by their length
    
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
        return BGGGraphPlot

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

            # enumerate all paths of length 3, a->b->c->b' such that b' != b and length goes +1,+1,-1
            self.cycles=chain.from_iterable([[a+(v,) for v in incoming[a[-1]] if v != a[1]] for a in self.cycles])

            # enumerate all cycles of length 4, a->b->c->b'->a such that b'!=b and length goes +1,+1,-1,-1
            self.cycles=[a+(a[0],) for a in self.cycles if a[0] in incoming[a[-1]]]

        return self.cycles

    def compute_signs(self):
        """Computes signs for all the edges so that the product of signs around any admissible cycle is -1.
        Returns a dictionary with the edges as keys and the signs as values."""
        self.signs = compute_signs(self)

    def compute_maps(self,root):
        """Initialize an instance of the map solver"""
        self.find_cycles()
        MapSolver = BGGMapSolver(self,  root)
        return MapSolver.solve()

    def _weight_to_tuple(self,weight):
        """Decompose a weight into a tuple encoding the weight as a linear combination of the simple roots"""
        b=weight.to_vector()
        b=matrix(b).transpose()
        A=[list(a.to_vector()) for a in self.simple_roots]
        A=matrix(A).transpose()
        return transpose(A.solve_right(b)).list()

    def weight_to_alpha_sum(self,weight):
        tuple = self._weight_to_tuple(weight)
        alpha = self.lattice.alpha()
        zero = self.lattice.zero()
        return sum((int(c)*alpha[i+1] for i, c in enumerate(tuple)), zero)

    def _tuple_to_weight(self,t):
        """Turn a tuple encoding a linear combination of simple roots back into a weight"""
        return sum(a*b for a, b in zip(t,self.simple_roots))

    def dot_action(self,reflection,weight):
        """Compute the dot action of a reflection on a weight. The reflection should be an element of the Weyl group
        self.W and the weight should be given as a tuple encoding it as a linear combination of simple roots."""
        weight = self._tuple_to_weight(weight)
        new_weight= reflection.action(weight+self.rho)-self.rho
        return self._weight_to_tuple(new_weight)

    def is_dot_regular(self, mu):
        stab_counter = 0
        for w in self.W:
            if w.action(mu + self.rho) - self.rho == mu:
                stab_counter += 1
        if stab_counter <= 1:
            return True
        else:
            return False

    def make_dominant(self, mu):
        for w in self.W:
            new_mu = w.action(mu + self.rho) - self.rho
            if new_mu.is_dominant():
                return new_mu, w