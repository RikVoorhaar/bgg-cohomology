"""
Implementation of the module describing the hochschild cohomology of the center of the small quantum group.
"""
from bggcomplex import BGGComplex
from fast_module import FastLieAlgebraCompositeModule, FastModuleFactory
from IPython.display import display, Math, Latex
import cohomology
import numpy as np
from tqdm.notebook import tqdm
from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix

def Mijk(BGG, i, j, k, subset=[]):
    factory = FastModuleFactory(BGG.LA)

    component_dic = {'u': factory.build_component('u', 'coad', subset=subset),
                     'g': factory.build_component('g', 'ad', subset=subset),
                     'n': factory.build_component('n', 'ad', subset=subset)}

    dim_n = len(factory.parabolic_n_basis(subset))

    components = []
    for r in range(j + k // 2 + 1):
        if (j - r <= dim_n):
            components.append([('u', j + k // 2 - r, 'sym'), ('g', r, 'wedge'), ('n', j - r, 'wedge')])

    module = FastLieAlgebraCompositeModule(factory, components, component_dic)

    return module

def sort_sign(A):
    num_cols = A.shape[1]
    if num_cols>1:
        signs=(-1)**(sum([(A[:,0]<A[:,i]) for i in range(1,num_cols)]))
        mask=sum([(A[:,0]==A[:,i]) for i in range(1,num_cols)])==0
        return (np.sort(A),signs,mask)
    else:
        return (A,np.ones(len(A),dtype=int),np.ones(len(A),dtype=bool))


def compute_phi(BGG, subset=[]):
    factory = FastModuleFactory(BGG.LA)

    component_dic = {'u': factory.build_component('u', 'coad', subset=subset)}
    components_phi = [[('u', 1, 'wedge')]]

    module_phi = FastLieAlgebraCompositeModule(factory, components_phi, component_dic)

    b_basis = factory.parabolic_p_basis(subset)
    u_basis = factory.parabolic_u_basis(subset)
    coad_dic = factory.coadjoint_action_tensor(b_basis, u_basis)

    phi_image = {b: [] for b in b_basis}
    for (b, u), n_dic in coad_dic.items():
        n, coeff = n_dic.items()[0]
        phi_image[b].append(np.array([factory.dual_root_dict[u], n, coeff]))

    for b in phi_image.keys():
        img = phi_image[b]
        if len(img) > 0:
            phi_image[b] = np.vstack(img)
        else:
            phi_image[b] = []
    return phi_image


def Tijk_basis(BGG, i, j, k, subset=[]):
    factory = FastModuleFactory(BGG.LA)
    wc_mod = Mijk(BGG, i, j, k, subset=subset)
    wc_rel = Mijk(BGG, i, j - 1, k, subset=subset)

    coker_dic = dict()
    mu_source_counters = dict()

    phi_image = compute_phi(BGG, subset=subset)
    # phi_image = {i:[] for i,_ in phi_image.items()}

    for mu, components in wc_rel.weight_components.items():

        for b, n_tensor_u in phi_image.items():
            new_mu = tuple(mu + factory.weight_dic[b])
            if new_mu not in mu_source_counters:
                mu_source_counters[new_mu] = 0
            sparse_relations = []
            for comp_num, basis in components:
                r = wc_rel.components[comp_num][1][1]
                t_un_insert = None
                t_g_insert = None
                for c, comp_list in enumerate(wc_mod.components):
                    if comp_list[1][1] == r:
                        t_un_insert = c
                    if comp_list[1][1] == r + 1:
                        t_g_insert = c

                num_rows = len(basis)
                num_u, num_g, num_n = (j - 1 + k // 2 - r, r, j - 1 - r)

                basis_enum = np.arange(mu_source_counters[new_mu], mu_source_counters[new_mu] + num_rows,
                                       dtype=int).reshape((-1, 1))
                mu_source_counters[new_mu] += num_rows

                # Insert g
                if t_g_insert is not None:
                    new_basis = np.hstack(
                        [basis_enum, basis[:, :num_u], b * np.ones(shape=(num_rows, 1), dtype=int), basis[:, num_u:]])
                    sort_basis, signs, mask = sort_sign(new_basis[:, num_u + 1:num_u + num_g + 2])
                    new_basis[:, num_u + 1:num_u + num_g + 2] = sort_basis
                    new_basis = new_basis[mask]
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
                        n_column = n * np.ones(shape=(num_rows, 1), dtype=int)
                        u_column = u * np.ones(shape=(num_rows, 1), dtype=int)
                        new_basis = np.hstack(
                            [basis_enum, u_column, basis[:, :num_u + num_g], n_column, basis[:, num_u + num_g:]])
                        new_basis[:, 1:num_u + 2] = np.sort(new_basis[:, 1:num_u + 2])
                        sort_basis, signs, mask = sort_sign(new_basis[:, num_u + num_g + 2:])
                        new_basis[:, num_u + num_g + 2:] = sort_basis
                        signs = coeff * signs[mask]
                        new_basis = new_basis[mask]
                        basis_un = np.zeros(shape=(len(new_basis), 3), dtype=int)
                        for row_num, (row, sign) in enumerate(zip(new_basis, signs)):
                            source = row[0]
                            target = wc_mod.weight_comp_index_numbers[new_mu][tuple(list(row[1:]) + [t_un_insert])]
                            basis_un[row_num] = [source, target, sign]
                        if len(basis_un) > 0:
                            sparse_relations.append(basis_un)
            if len(sparse_relations) > 0:
                if new_mu not in coker_dic:
                    coker_dic[new_mu] = []
                coker_dic[new_mu] += sparse_relations
    coker_dic = {mu: cohomology.sort_merge(np.concatenate(rels)) for mu, rels in coker_dic.items()}

    T = dict()
    with tqdm(coker_dic.items()) as pbar:
        for mu, rels in pbar:
            source_dim = mu_source_counters[mu]
            target_dim = wc_mod.dimensions[mu]
            sparse_dic = dict()
            for source, target, coeff in rels:
                sparse_dic[(source, target)] = coeff
            pbar.set_description(str((source_dim, target_dim)))
            M = matrix(ZZ, sparse_dic, nrows=source_dim, ncols=target_dim, sparse=True)
            T[mu] = M.right_kernel().basis_matrix()
    return T

def all_abijk(BGG,s=0,subset=[],half_only=True):
    dim_n = len(FastModuleFactory(BGG.LA).parabolic_n_basis(subset))
    output = []
    s_mod = s%2
    a_max = dim_n+(1+s_mod)%2
    for a_iterator in range(a_max):
        a = 2*a_iterator+s_mod
        for b in range(max(0,s-2*a),a+1):
            if (a+b) % 2 == 0:
                output.append((a,b,(a-b)//2,(a+b)//2,s-a))
    if half_only:
        new_out = []
        max_a = max(s[0] for s in output)
        for a,b,i,j,k in output:
            if a+b<=max_a:
                new_out.append((a,b,i,j,k))
        return sorted(new_out,key=lambda s: (s[0]+s[1],s[0],s[1]))
    else:
        return output

def display_ab_dic(ab_dic):
    a_set = set()
    b_set = set()
    for a,b in ab_dic.keys():
        a_set.add(a)
        b_set.add(b)
    a_set = sorted(a_set)
    b_set = sorted(b_set)
    column_string = r'{r|' + ' '.join('l'*len(a_set))+r'}'
    rows = list()
    for a in a_set:
        row_strings = [r'{\scriptstyle i+j='+str(a)+r'}']
        for b in b_set:
            if (a,b) in ab_dic:
                row_strings.append(ab_dic[(a,b)])
            else:
                row_strings.append('')
        rows.append(r'&'.join(row_strings))
    rows.append(r'\hline h^{i,j}&'+ '&'.join([r'{\scriptstyle j-i='+str(b)+r'}' for b in b_set]))
    all_strings = r'\\'.join(rows)
    display(Math(r'\begin{array}'+column_string+all_strings+r'\end{array}'))