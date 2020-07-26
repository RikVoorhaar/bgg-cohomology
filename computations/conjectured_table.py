from sage.all import *
from datetime import datetime
import argparse

with open('conjecture.log','w') as log:
    log.write(f'{datetime.now()}\tInitializing...\n')

parser = argparse.ArgumentParser(description="Compute bigraded table C[h+h*]/I")
parser.add_argument("root_system", type=str)
args=parser.parse_args()
root_system = args.root_system

W = WeylGroup(root_system)
var_names = ['x'+str(i) for i in range(W.rank())]+['y'+str(i) for i in range(W.rank())]
R = PolynomialRing(QQ,var_names)
R_vars = [R.gens_dict()[s] for s in var_names]
domain = W.domain()
simple_roots = domain.simple_roots().values()
def root_to_basis(weight):
    b=weight.to_vector()
    b=matrix(b).transpose()
    A=[list(a.to_vector()) for a in simple_roots]
    A=matrix(A).transpose()

    return tuple(A.solve_right(b).transpose().list())
gens = W.generators()
def weyl_to_string(w):
    return ''.join(str(s) for s in w.reduced_word())
weyl_on_basis = {weyl_to_string(w):matrix([root_to_basis(w.action(r)) for r in simple_roots]) for w in W}
weyl_on_poly_x = dict()
weyl_on_poly_y = dict()
for w_string,w in weyl_on_basis.items():
    weyl_on_poly_x[w_string] = [sum(R.gens_dict()['x'+str(i)]*w[j][i] for i in range(W.rank())) for j in range(W.rank())]
    weyl_on_poly_y[w_string] = [sum(R.gens_dict()['y'+str(i)]*w[j][i] for i in range(W.rank())) for j in range(W.rank())]

def bigraded_basis(a, b, output_as_tuple=True):
    x_part = OrderedPartitions(a+W.rank(),W.rank()).list()
    x_part = [tuple(x-1 for x in xp) for xp in x_part]
    y_part = OrderedPartitions(b+W.rank(),W.rank()).list()
    y_part = [tuple(x-1 for x in xp) for xp in y_part]
    basis = [(x,y) for x in x_part for y in y_part]
    if output_as_tuple:
        return basis
    else:
        return [tuple_to_monomial(t) for t in basis]

def bigraded_action(s,tup,basis,basis_dict):
    wx = weyl_on_poly_x[s]
    wy = weyl_on_poly_y[s]
    polx = prod(x**i for x,i in zip(wx,tup[0]))
    poly = prod(y**i for y,i in zip(wy,tup[1]))
    pol = polx*poly
    output = vector(QQ,len(basis))
    for coeff,monom in pol:
        index = basis_dict[tuple(monom.exponents()[0])]
        output[index]=coeff
    return output

def trace_map(tup,basis,basis_dict):
    return sum(bigraded_action(s,tup,basis,basis_dict) for s in weyl_on_basis)

def tuple_to_monomial(tup):
    if type(tup[0]) is tuple:
        tup = sum(tup,tuple())
    return prod(R_vars[i]**j for i,j in enumerate(tup))

def vec_to_poly(vec,basis):
    return sum(tuple_to_monomial(basis[i])*c for i,c in enumerate(vec))

@cached_function
def bigraded_invariants(a,b):
    basis = bigraded_basis(a,b)
    basis_dict = {x+y:i for i,(x,y) in enumerate(basis)}
    polys = [trace_map(t,basis,basis_dict) for t in basis]
    polys = [p for p in polys if p is not 0]
    polys = span(polys).basis()
    return [vec_to_poly(v,basis) for v in polys]

def quotient_dim(a,b):
    a_pairs = OrderedPartitions(a+2,2)
    a_pairs = [tuple(x-1 for x in xp) for xp in a_pairs]
    b_pairs = OrderedPartitions(b+2,2)
    b_pairs = [tuple(x-1 for x in xp) for xp in b_pairs]
    ab_tuples = [x+y for x in a_pairs for y in b_pairs]
    ideal = []
    for a1,a2,b1,b2 in ab_tuples:
        if a2==b2==0:
            continue
        basis = bigraded_basis(a1,b1,False)
        invariants = bigraded_invariants(a2,b2)
        ideal+=[x*y for x in basis for y in invariants]
    basis = bigraded_basis(a,b)
    basis_dict = {x+y:i for i,(x,y) in enumerate(basis)}
    ideal_vects=[]
    for pol in ideal:
        output = vector(QQ,len(basis))
        for coeff,monom in pol:
            try:
                index = basis_dict[tuple(monom.exponents()[0])]
            except KeyError:
                print(monom,coeff)
                print(a,b)
                raise
            output[index]=coeff
        ideal_vects.append(output)
    return len(basis)-matrix(ideal_vects).rank()

max_ab = len(domain.negative_roots())
all_ab = [(a,b) for a in range(max_ab+1) for b in range(min(a+1,max_ab-a+1))]
with open('conjecture.log','a') as log:
    log.write(f'{datetime.now()}\tType {root_system}. Total entries: {len(all_ab)}.\n')
quotient_dims = dict()
for i,(a,b) in enumerate(all_ab):
    quotient_dims[(a,b)] = quotient_dim(a,b)
    with open('conjecture.log','a') as log:
        log.write(f'{datetime.now()}\tEntry {i+1}/{len(all_ab)}, (a,b)={(a,b)}, dim={quotient_dims[(a,b)]}.\n')
bigraded_table = [[quotient_dims[(a,b)] for b in range(min(a+1,max_ab-a+1))] for a in range(max_ab+1)][::-1]
with open('conjecture.log','a') as log:
    log.write(f'{datetime.now()}\tComputation finished. Final result: \n\n')
    for row in bigraded_table:
        log.write('\t'.join(str(r) for r in row)+'\n')
