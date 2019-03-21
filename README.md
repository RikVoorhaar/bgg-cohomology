# BGG


The plan is to write a module in Python/Sage to compute the cohomology of the BGG complex for certain modules to compute the center of the small quantum group. 
We did this in Mathematica for G2, but we want to be able to do it for a larger class of Lie algebras. We will do this in Python/Sage for the following reasons:
- Mathematica is closed source, as opposed to Sage and Python.
- More existing tools to do computations with Lie algebras and non-commutative algebra
- Better implementation of some algorithms should give a speed-up

## Outline

The computation of the BGG cohomology consists of several steps. 
- We begin with a Lie algebra $\mathfrak g$ defined by a Cartan matrix.
- It's Weyl group has the structure of a multipartite directed graph, partioned by word length
- Given a weight $\lambda$ we can associate to each vertex of the graph a weight
- Each arrow in the graph represents a map between weight modules. In certain cases we know a priori what these maps are.
- Using the constraint that some cycles in the graph commute, we compute the remaining maps.
- We define a sign distribution on the edges of the graph such that the product of signs around a cycle is always -1. This way we obtain a differential on the the entire complex.
- We need to define the action of these maps on the weight modules
- We compute a basis for all the relevant weight modules
- We compute the cohomology of the complex

## To do:
We need to write scripts that do the following things
- (Rik) Given a cartan matrix, construct the BGG graph 
- (Rik) Compute the dot action 
- (Nicolas) Solve equations of form $q\cdot p_1 = p_2$ in the ring of non-commutative polynomials Z<x1,...,xn>/I, where I is an ideal (the Serre relations)
- Compute the maps of the BGG complex
- (Rik) Find a sign distribution for the BGG complex
- (Nicolas) Define the adjoint action and (restricted) coadjoint action on the Lie algebra (in the Chevalley basis $h_i$, $e_I$, $f_I$)
- Extend the ajoint/coadjoint action to Tensor/Wedge/Symmetric products of elements of the Lie algebra and it's subspaces $\mathfrak{u},\mathfrak{n},\mathfrak{b},\mathfrak{h}$
- Construct a basis of the relevant weight modules
- Compute the BGG cohomology