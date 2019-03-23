from itertools import groupby,chain
#from sage.combinat.root_system.weyl_group import  WeylGroup
#from sage.graphs.digraph import DiGraph
#from sage.matrix import *
from sage.all import *

class BGGComplex:
    """A class encoding all the things we need of the BGG complex"""
    def __init__(self, root_system):
        self.W = WeylGroup(root_system)
        self.S = self.W.simple_reflections()
        self.T = self.W.reflections()
        
        self._compute_weyl_dictionary()
        self._construct_BGG_graph()
        
        self.rho = self.W.domain().rho()
        self.simple_roots = self.W.domain().simple_roots().values()
        self.zero_root = self.W.domain().zero()
        
    def _compute_weyl_dictionary(self):
        self.WeylDic={"":self.W.one()}
        word_length=0
        while len(self.WeylDic)<self.W.order():
            BGGColumn=[s for s in self.WeylDic.keys() if len(s)==word_length]
            for w in BGGColumn:
                for s in self.W.index_set():
                    test_element= self.S[s]*self.WeylDic[w]
                    if test_element not in self.WeylDic.values():
                        self.WeylDic[str(s)+w]=test_element
            word_length+=1
        self.WeylDicReverse=dict([[v,k] for k,v in self.WeylDic.items()])
    
    def _construct_BGG_graph(self):
        arrows=[]
        for w in self.WeylDic.keys():
            for t in self.T:
                product_word = self.WeylDicReverse[t*self.WeylDic[w]]
                if len(product_word)==len(w)+1:
                    arrows+=[(w,product_word)]
        self.graph= DiGraph(arrows)
    
    def plot_graph(self):  
        BGGVertices=sorted(self.WeylDic.keys(),key=len)
        BGGPartition=[list(v) for k,v in groupby(BGGVertices,len)]

        BGGGraphPlot = self.graph.to_undirected().graphplot(partition=BGGPartition,vertex_labels=None,vertex_size=30)
        return BGGGraphPlot

    def _weight_to_tuple(self,weight):
        b=weight.to_vector()
        b=matrix(b).transpose()
        A=[list(a.to_vector()) for a in self.simple_roots]
        A=matrix(A).transpose()
        return transpose(A.solve_right(b)).list()

    def _tuple_to_weigth(self,t):
        return sum(a*b for a,b in zip(t,self.simple_roots))

    def dot_action(self,reflection,weight_tuple):
        weight = self._tuple_to_weigth(weight_tuple)
        new_weight= reflection.action(weight+self.rho)-self.rho
        return self._weight_to_tuple(new_weight)
        