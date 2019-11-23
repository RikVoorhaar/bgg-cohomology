"""
Implementation of the module describing the hochschild cohomology of the center of the small quantum group.
"""
from bggcomplex import BGGComplex
from fast_module import FastLieAlgebraCompositeModule, FastModuleFactory
from IPython.display import display, Math, Latex
import cohomology
import numpy as np
from tqdm.auto import tqdm
from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix


def Mijk(BGG, i, j, k, subset=[]):
    """Define the module Mijk. Here i is the cohomology degreee, j and k are parameters."""
    factory = FastModuleFactory(BGG.LA)

    component_dic = {'u': factory.build_component('u', 'coad', subset=subset),
                     'g': factory.build_component('g', 'ad', subset=subset),
                     'n': factory.build_component('n', 'ad', subset=subset)}

    dim_n = len(factory.parabolic_n_basis(subset))

    components = []
    # This is the direct sum decomposition of the module
    # The two checks are to ensure wedge^{j-r}n is well defined.
    for r in range(j + k // 2 + 1):
        if (j - r <= dim_n) and (j - r >= 0):
            components.append([('u', j + k // 2 - r, 'sym'), ('g', r, 'wedge'), ('n', j - r, 'wedge')])

    module = FastLieAlgebraCompositeModule(factory, components, component_dic)

    return module

def sort_sign(A)
    """Sort a numpy array which is sorted except for the first entry.
     Compute the signs of the permutations sorting them.
     Returns the sorted array, the signs for each row, 
     and a mask with False for every row where sign is zero (i.e. if there is a duplicate entry)"""
    num_cols = A.shape[1]
    if num_cols>1:
        signs=(-1)**(sum([(A[:,0]<A[:,i]) for i in range(1,num_cols)]))
        mask=sum([(A[:,0]==A[:,i]) for i in range(1,num_cols)])==0
        return (np.sort(A),signs,mask)
    else:
        return (A,np.ones(len(A),dtype=int),np.ones(len(A),dtype=bool))

def compute_phi(BGG,subset= []):
    """Computes the map b->g\oplus n\otimes u. It is returned as a dictionary
    b-> n\otimes u, where the latter is stored as an array with rows (n_index, u_index, coefficient)"""

    factory = FastModuleFactory(BGG.LA)

    b_basis = factory.parabolic_p_basis(subset)
    n_basis = factory.parabolic_n_basis(subset)

    # adjoint action of b on n
    ad_dic = factory.adjoint_action_tensor(b_basis,n_basis)

    # look up adjoint action for each b
    phi_image = {b:[] for b in b_basis}
    for (b,n),u_dic in ad_dic.items():
        u,coeff = u_dic.items()[0]
        phi_image[b].append(np.array([u,factory.dual_root_dict[n],coeff]))

    # process the adjoint action image into the right format
    for b in phi_image.keys():
        img = phi_image[b]
        if len(img)>0:
            phi_image[b]=np.vstack(img)
        else:
            phi_image[b]=[]
    return phi_image


def Tijk_basis(BGG, i, j, k, subset=[], pbar=None):
    """Gives a basis of the quotient Ejk = Mjk/Tjk for each weight component"""
    factory = FastModuleFactory(BGG.LA)

    # Change progress bar status.
    if pbar is not None:
        pbar.reset()
        pbar.set_description('Creating modules')

    #target module
    wc_mod = Mijk(BGG, i, j, k, subset=subset)

    #source module
    wc_rel = Mijk(BGG, i, j - 1, k, subset=subset)

    coker_dic = dict()

    # For each mu, count the dimension of the space of relations
    mu_source_counters = dict()

    # The image of the map phi: b -> g\oplus n\otimes u
    phi_image = compute_phi(BGG, subset=subset)

    if pbar is not None:
        pbar.set_description('Computing modules')


    for mu, components in wc_rel.weight_components.items():
        for b, n_tensor_u in phi_image.items():

            # after inserting b or n_tensor_u, the weight changes
            new_mu = tuple(mu + factory.weight_dic[b])
            if new_mu not in mu_source_counters:
                mu_source_counters[new_mu] = 0

            # put relations in sparse matrix.
            sparse_relations = []
            for comp_num, basis in components:
                # Find the index of the direct sum component at which to insert respectively g and u \otimes n
                r = wc_rel.components[comp_num][1][1]
                t_un_insert = None
                t_g_insert = None
                for c, comp_list in enumerate(wc_mod.components):
                    if comp_list[1][1] == r:
                        t_un_insert = c
                    if comp_list[1][1] == r + 1:
                        t_g_insert = c

                num_rows = len(basis)
                # Compute the number of copies of u, g, n in the source component
                num_u, num_g, num_n = (j - 1 + k // 2 - r, r, j - 1 - r)

                # give each relation a different index
                basis_enum = np.arange(mu_source_counters[new_mu], mu_source_counters[new_mu] + num_rows,
                                       dtype=int).reshape((-1, 1))
                mu_source_counters[new_mu] += num_rows

                # Insert g
                if t_g_insert is not None:
                    # insert g in the right slot, and then sort it
                    # we use a mask to get rid of entries which are zero
                    new_basis = np.hstack(
                        [basis_enum, basis[:, :num_u], b * np.ones(shape=(num_rows, 1), dtype=int), basis[:, num_u:]])
                    sort_basis, signs, mask = sort_sign(new_basis[:, num_u + 1:num_u + num_g + 2])
                    new_basis[:, num_u + 1:num_u + num_g + 2] = sort_basis
                    new_basis = new_basis[mask]

                    # For each row, look up the index in the target module
                    # then put the tuple (soruce, target, sign) in the list of relations.
                    basis_g = np.zeros(shape=(len(new_basis), 3), dtype=int)
                    for row_num, (row, sign) in enumerate(zip(new_basis, signs[mask])):
                        source = row[0]
                        target = wc_mod.weight_comp_index_numbers[new_mu][tuple(list(row[1:]) + [t_g_insert])]
                        basis_g[row_num] = [source, target, -sign]
                    if len(basis_g) > 0:
                        sparse_relations.append(basis_g)

                # Insert n_tensor_u
                if t_un_insert is not None:
                    for n, u, coeff in n_tensor_u:
                        # make two constant columns with the index of inserted n and u respectively
                        n_column = n * np.ones(shape=(num_rows, 1), dtype=int)
                        u_column = u * np.ones(shape=(num_rows, 1), dtype=int)

                        # insert the u and the n, sort the u (no signs) and sort the n (but correct for signs)
                        new_basis = np.hstack(
                            [basis_enum, u_column, basis[:, :num_u + num_g], n_column, basis[:, num_u + num_g:]])
                        new_basis[:, 1:num_u + 2] = np.sort(new_basis[:, 1:num_u + 2])
                        sort_basis, signs, mask = sort_sign(new_basis[:, num_u + num_g + 2:])
                        new_basis[:, num_u + num_g + 2:] = sort_basis
                        signs = coeff * signs[mask]
                        new_basis = new_basis[mask]

                        # For each row, look up the index in the target module
                        # then put the tuple (soruce, target, sign*coefficient) in the list of relations.
                        basis_un = np.zeros(shape=(len(new_basis), 3), dtype=int)
                        for row_num, (row, sign) in enumerate(zip(new_basis, signs)):
                            source = row[0]
                            target = wc_mod.weight_comp_index_numbers[new_mu][tuple(list(row[1:]) + [t_un_insert])]
                            basis_un[row_num] = [source, target, sign]
                        if len(basis_un) > 0:
                            sparse_relations.append(basis_un)

            # concatenate the relations into a single array
            if len(sparse_relations) > 0:
                if new_mu not in coker_dic:
                    coker_dic[new_mu] = []
                coker_dic[new_mu] += sparse_relations

    # we now have a dictionary sending each weight mu to its relations
    coker_dic = {mu: cohomology.sort_merge(np.concatenate(rels)) for mu, rels in coker_dic.items()}

    T = dict()
    total_rels = len(coker_dic)
    if pbar is not None:
        pbar.reset(total=total_rels)
        pbar.set_description('Computing kernels')

    # now we need to compute the cokernel (= right kernel) of the relations to get a basis
    for mu, rels in coker_dic.items():
        # compute the dimension of source and target
        source_dim = mu_source_counters[mu]
        target_dim = wc_mod.dimensions[mu]

        # build the matrix of relations
        sparse_dic = dict()
        for source, target, coeff in rels:
            sparse_dic[(source, target)] = coeff
        M = matrix(ZZ, sparse_dic, nrows=source_dim, ncols=target_dim, sparse=True)

        if pbar is not None:
            pbar.update()

        # compute the right kernel, store it in a dictionary
        T[mu] = M.right_kernel().basis_matrix()
    return T

def all_abijk(BGG,s=0,subset=[],half_only=False):
    """Returns a list of all the (a,b) in the bigraded table. If half_only=True then only
    half of the table is returned; the other half can be deduced by symmetry."""
    dim_n = len(FastModuleFactory(BGG.LA).parabolic_n_basis(subset))
    output = []
    # the maximal value of a is the dimension of n, +1 depending on the parity of s.
    s_mod = s%2
    a_max = dim_n+(1+s_mod)%2

    # make a list of all the pairs (a,b)
    for a_iterator in range(a_max):
        a = 2*a_iterator+s_mod
        for b in range(a+1):
            if (a+b) % 2 == 0:
                output.append((a,b,(a-b)//2,(a+b)//2,s-a))

    # if half_only, throw out half of the values
    # we throw something away depending on the sum a+b
    if half_only:
        new_out = []
        max_a = max(s[0] for s in output)
        for a,b,i,j,k in output:
            if a+b<=max_a+1:
                new_out.append((a,b,i,j,k))
        return sorted(new_out,key=lambda s: (s[0]+s[1],s[0],s[1]))
    else:
        return output

def display_ab_dic(ab_dic,extend_half = False):
    """Generates LaTeX code to display the bigraded table. Takes a dictionary (a,b) -> LaTeX string.
    If extend_half = True, we extend the table by using the symmetry"""

    # extend by symmetry
    if extend_half:
        max_a = max(a for a,b in ab_dic.keys())
        max_a = max_a + (max_a%2)
        for (a,b),v in ab_dic.items():
            ab_dic[(max_a-b,max_a-a)] = v

    # Find all the values of a,b
    a_set = set()
    b_set = set()
    for a,b in ab_dic.keys():
        a_set.add(a)
        b_set.add(b)
    a_set = sorted(a_set)
    b_set = sorted(b_set)

    # string telling \array how to format the columns
    column_string = r'{r|' + ' '.join('l'*len(a_set))+r'}'
    rows = list()

    # for each a,b that's actually in the dictionary
    for a in a_set:
        # add the left most entry of the row
        row_strings = [r'{\scriptstyle i+j='+str(a)+r'}']
        for b in b_set:
            if (a,b) in ab_dic:
                row_strings.append(ab_dic[(a,b)])
            else:
                row_strings.append('')
        #separate all the entries by an & symbol
        rows.append(r'&'.join(row_strings))

    # add the last row
    rows.append(r'\hline h^{i,j}&'+ '&'.join([r'{\scriptstyle j-i='+str(b)+r'}' for b in b_set]))
    all_strings = r'\\'.join(rows)

    # display the generated latex code.
    display(Math(r'\begin{array}'+column_string+all_strings+r'\end{array}'))