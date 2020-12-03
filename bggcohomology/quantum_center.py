"""Implementation of the module describing the hochschild cohomology of flag varieties.

Equivalently this givesthe center of the small quantum group, as described in the
two papers by Anna Lachowska and You Qi:

https://arxiv.org/abs/1604.07380v3

https://arxiv.org/abs/1703.02457v3

Let :math:`G` be a complex simple Lie group, and let :math:`P` be a parabolic subgroup. 
Then we consider the cotangent bundle of the associated partial flag variety:

.. math::
    \\tilde{\\mathcal N}_P :=T^*(G/P)

We are then interested in computing

.. math::
    HH^s(\\tilde{\\mathcal N}_P)\\cong \\bigoplus_{i+j+k=s}H^i(\\tilde{\\mathcal N}_P,\\wedge^jT
    \\tilde{\\mathcal N}_P)^k

This can be computed by using the BGG resolution. We define the following module:

.. math::
    M_j^k = \\bigoplus_r \\operatorname{Sym}^{j-r+k/2}\\mathfrak u_P\\otimes \\wedge^r\\mathfrak
    g\\otimes \\wedge^{j-r}\\mathfrak n_P

Let :math:`\\Delta\\colon\\mathfrak p\\to \\mathfrak g\\oplus \\mathfrak u_P\\otimes\\mathfrak n_P`
be given by the inclusion in the first component and in the second component by the 
adjoint action (after identifying :math:`\\operatorname{End}(\\mathfrak n_P)` with 
:math:`\\mathfrak u_P\\otimes \\mathfrak n_P`). Then :math:`\\Delta` induces a map 
:math:`M_{j-1}^k\\to M_j^k`. We define the module

.. math::
    E_j^k = M_j^k\\big/\\Delta(M_{j-1}^k)

Then the cohomology of the BGG resolution of :math:`E_j^k` in degree :math:`i` with respect 
to a dominant weight :math:`\\mu` computes the multiplicity of :math:`\\mu` of 
:math:`H^i(\\tilde{\\mathcal N}_P,\\wedge^jT\\tilde{\\mathcal N}_P)^k`.
"""
from collections import defaultdict

import numpy as np
from IPython.display import Math, display

from . import cohomology
from .la_modules import LieAlgebraCompositeModule, ModuleFactory, BGGCohomology

from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ

import pickle


def Mjk(BGG, j, k, subset=[]):
    r"""Define the module Mjk.

    .. math::
        M_j^k = \bigoplus_r \operatorname{Sym}^{j-r+k/2}\mathfrak u_P\otimes \wedge^r\mathfrak
        g\otimes \wedge^{j-r}\mathfrak n_P

    Parameters
    ----------
    j : int
    k : int

    Returns
    -------
    LieAlgebraCompositeModule
    """
    factory = ModuleFactory(BGG.LA)

    component_dic = {
        "u": factory.build_component("u", "coad", subset=subset),
        "g": factory.build_component("g", "ad", subset=subset),
        "n": factory.build_component("n", "ad", subset=subset),
    }

    dim_n = len(factory.parabolic_n_basis(subset))

    components = []
    # This is the direct sum decomposition of the module
    # The two checks are to ensure wedge^{j-r}n is well defined.
    for r in range(j + k // 2 + 1):
        if (j - r <= dim_n) and (j - r >= 0):
            components.append(
                [
                    ("u", j + k // 2 - r, "sym"),
                    ("g", r, "wedge"),
                    ("n", j - r, "wedge"),
                ]
            )

    module = LieAlgebraCompositeModule(factory, components, component_dic)

    return module


def _sort_sign(A):
    """Sort a numpy array which is sorted except for the first entry, and give sign of permutation.

    Parameters
    ----------
    A : numpy.array[int, int]

    Returns
    -------
    numpy.array[int, int]
        Sorted array
    numpy.array[int]
        Signs of permutations sorting the array
    numpy.array[bool]
        Mask which is `False` in every row where there was a duplicate entry
    """
    num_cols = A.shape[1]
    if num_cols > 1:
        signs = (-1) ** (
            sum([(A[:, 0] < A[:, i]) for i in range(1, num_cols)])
        )
        mask = sum([(A[:, 0] == A[:, i]) for i in range(1, num_cols)]) == 0
        return (np.sort(A), signs, mask)
    else:
        return (A, np.ones(len(A), dtype=int), np.ones(len(A), dtype=bool))


def compute_phi(BGG, subset=[]):
    r"""Compute the map :math:`\mathfrak b\to \mathfrak n\otimes \mathfrak u`.

    Parameters
    ----------
    BGG : BGGComplex
    subset : list[int] (default: [])
        Subset defining parabolic

    Returns
    -------
    dict[int, np.array[int]]
        Dictionary mapping basis indices of :math:`\mathfrak b` to an array with rows
        `(n_index, u_index, coeff)`, where `n_index` is index in basis of :math:`\mathfrak n`,
        `u_index` is the index in basis of :math:`\mathfrak u`, and `coeff` is the coefficient.
    """
    factory = ModuleFactory(BGG.LA)

    b_basis = factory.parabolic_p_basis(subset)
    n_basis = factory.parabolic_n_basis(subset)

    # adjoint action of b on n
    ad_dic = factory.adjoint_action_tensor(b_basis, n_basis)

    # look up adjoint action for each b
    phi_image = {b: [] for b in b_basis}
    for (b, n), u_dic in ad_dic.items():
        u, coeff = next(iter(u_dic.items()))
        phi_image[b].append(np.array([u, factory.dual_root_dict[n], coeff]))

    # process the adjoint action image into the right format
    for b in phi_image.keys():
        img = phi_image[b]
        if len(img) > 0:
            phi_image[b] = np.vstack(img)
        else:
            phi_image[b] = []
    return phi_image


def Eijk_basis(BGG, j, k, subset=[], pbar=None, method=0):
    r"""Give a basis of the quotient :math:`E_j^k = M_j^k\big/\Delta(M_{j-1}^k)`.

    Parameters
    ----------
    BGG : BGGComplex
    j : int
    k : int
    subset : list[int] (default: [])
        Subset defining parabolic
    pbar : tqdm or `None` (default `None`)
        tqdm instance to send status updates to. If `None` this feature is disable.

    Returns
    -------
    CokerCache
        Object that stores basis of :math:`M_j^k` and frame of :math:`\Delta(M_{j-1}^k)`,
        so that basis of :math:`E_j^k` can be computed when needed.
    """
    factory = ModuleFactory(BGG.LA)

    # Change progress bar status.
    if pbar is not None:
        pbar.reset()
        pbar.set_description("Creating modules")

    # target module
    wc_mod = Mjk(BGG, j, k, subset=subset)

    basis_indices = _quotient_basis_indices(wc_mod)

    # source module
    wc_rel = Mjk(BGG, j - 1, k, subset=subset)

    coker_dic = dict()

    # For each mu, count the dimension of the space of relations
    mu_source_counters = dict()

    # The image of the map phi: b -> g\oplus n\otimes u
    phi_image = compute_phi(BGG, subset=subset)

    if pbar is not None:
        pbar.set_description("Computing modules")

    for mu, components in wc_rel.weight_components.items():
        for b, n_tensor_u in phi_image.items():

            # after inserting b or n_tensor_u, the weight changes
            new_mu = tuple(mu + factory.weight_dic[b])
            if new_mu not in mu_source_counters:
                mu_source_counters[new_mu] = 0

            # put relations in sparse matrix.
            sparse_relations = []
            for comp_num, basis in components:
                # Find the index of the direct sum component at which to insert respectively g and u
                # \otimes n
                r = wc_rel.components[comp_num][1][1]
                t_un_insert = None  # component number where u / n are inserted
                t_g_insert = None  # component number where g is inserted
                for c, comp_list in enumerate(wc_mod.components):
                    if comp_list[1][1] == r:
                        t_un_insert = c
                    if comp_list[1][1] == r + 1:
                        t_g_insert = c

                num_rows = len(basis)
                # Compute the number of copies of u, g, n in the source component
                num_u, num_g, num_n = (j - 1 + k // 2 - r, r, j - 1 - r)

                # give each relation a different index
                basis_enum = np.arange(
                    mu_source_counters[new_mu],
                    mu_source_counters[new_mu] + num_rows,
                    dtype=int,
                ).reshape((-1, 1))
                mu_source_counters[new_mu] += num_rows

                # Insert g
                if t_g_insert is not None:
                    # insert g in the right slot, and then sort it
                    # we use a mask to get rid of entries which are zero
                    new_basis = np.hstack(
                        [
                            basis_enum,
                            basis[:, :num_u],
                            b * np.ones(shape=(num_rows, 1), dtype=int),
                            basis[:, num_u:],
                        ]
                    )
                    sort_basis, signs, mask = _sort_sign(
                        new_basis[:, num_u + 1 : num_u + num_g + 2]
                    )
                    new_basis[:, num_u + 1 : num_u + num_g + 2] = sort_basis
                    new_basis = new_basis[mask]

                    # For each row, look up the index in the target module
                    # then put the tuple (soruce, target, sign) in the list of relations.
                    basis_g = np.zeros(shape=(len(new_basis), 3), dtype=int)
                    for row_num, (row, sign) in enumerate(
                        zip(new_basis, signs[mask])
                    ):
                        source = row[0]
                        target = wc_mod.weight_comp_index_numbers[new_mu][
                            tuple(list(row[1:]) + [t_g_insert])
                        ]
                        basis_g[row_num] = [source, target, sign]
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
                            [
                                basis_enum,
                                u_column,
                                basis[:, : num_u + num_g],
                                n_column,
                                basis[:, num_u + num_g :],
                            ]
                        )
                        new_basis[:, 1 : num_u + 2] = np.sort(
                            new_basis[:, 1 : num_u + 2]
                        )
                        sort_basis, signs, mask = _sort_sign(
                            new_basis[:, num_u + num_g + 2 :]
                        )
                        new_basis[:, num_u + num_g + 2 :] = sort_basis
                        signs = coeff * signs[mask]
                        new_basis = new_basis[mask]

                        # For each row, look up the index in the target module
                        # then put the tuple (source, target, sign*coefficient) in the list of relations.
                        basis_un = np.zeros(
                            shape=(len(new_basis), 3), dtype=int
                        )
                        for row_num, (row, sign) in enumerate(
                            zip(new_basis, signs)
                        ):
                            source = row[0]
                            target = wc_mod.weight_comp_index_numbers[new_mu][
                                tuple(list(row[1:]) + [t_un_insert])
                            ]
                            basis_un[row_num] = [source, target, sign]
                        if len(basis_un) > 0:
                            sparse_relations.append(basis_un)

            # concatenate the relations into a single array
            if len(sparse_relations) > 0:
                if new_mu not in coker_dic:
                    coker_dic[new_mu] = []
                coker_dic[new_mu] += sparse_relations

    # we now have a dictionary sending each weight mu to its relations
    coker_dic = {
        mu: cohomology.sort_merge(np.concatenate(rels))
        for mu, rels in coker_dic.items()
    }

    return CokerCache(
        coker_dic,
        mu_source_counters,
        wc_mod.dimensions,
        basis_indices,
        method=method,
    )


def _quotient_basis_indices(mjk):
    g_locations = [
        [i for i, s in enumerate(l) if s == "g"] for l in mjk.type_lists
    ]
    indices = dict()
    for mu in mjk.weight_components.keys():
        quotient_indices = []
        u_basis = mjk.component_dic["u"].basis
        for elem, index in mjk.weight_comp_index_numbers[mu].items():
            elem, comp_num = elem[:-1], elem[-1]
            keep = True
            for j in g_locations[comp_num]:
                if elem[j] not in u_basis:
                    keep = False
                    break
            if keep:
                quotient_indices.append(index)
        indices[mu] = quotient_indices
    return indices


class CokerCache:
    """Wrapper to compute cokernels on demand.

    Stores the image of a map of weight spaces, and computes the
    cokernel of the map when needed. Acts like a dictionary, and
    caches results.

    Parameters
    ----------
    rel_dic : dict[tuple(int), array[int]]
        Dictionary mapping weights to arrays encoding sparse matrix of
        relations to quotient by in DOK format.
    source_dims : dict[tuple(int), int]
        Dictionary mapping weights to dimensions of the weight components
        in the source space.
    target_dims : dict[tuple(int), int]
        Dictionary mapping weights to dimensions of the weight components
        in the target space.
    pbar : tqdm or `None` (default `None`)
        tqdm instance to send status updates to. If `None` this feature is disable.
    """

    def __init__(
        self,
        rel_dic,
        source_dims,
        target_dims,
        basis_indices,
        pbar=None,
        method=0,
    ):
        self.rel_dic = rel_dic
        self.source_dims = source_dims
        self.target_dims = target_dims
        self.computed_cokernels = dict()
        self.basis_indices = basis_indices
        self.pbar = pbar
        self.method = method

    def __getitem__(self, mu):
        if mu in self.computed_cokernels:
            return self.computed_cokernels[mu]
        else:
            rels = self.rel_dic[mu]
            source_dim = self.source_dims[mu]
            target_dim = self.target_dims[mu]
            basis = self.basis_indices[mu]
            if self.method == 0:
                ker = _compute_kernel2(
                    source_dim, target_dim, rels, basis, self.pbar
                )
            else:
                ker = _compute_kernel(source_dim, target_dim, rels, self.pbar)
            self.computed_cokernels[mu] = ker
            return ker

    def __setitem__(self, mu, value):
        self.computed_cokernels[mu] = value

    def __contains__(self, mu):
        return mu in self.rel_dic


def _compute_kernel(source_dim, target_dim, rels, pbar=None):
    """Compute cokernel.

    Given dimensions of source and target of a map,
    as well as the image of the map in DOK sparse matrix format,
    compute the cokernel of this map and return as a (dense) matrix.
    """
    # build the matrix of relations
    sparse_dic = dict()
    for source, target, coeff in rels:
        sparse_dic[(source, target)] = coeff
    M = matrix(ZZ, sparse_dic, nrows=source_dim, ncols=target_dim, sparse=True)
    M = M.dense_matrix()

    if pbar is not None:
        pbar.update()
        pbar.set_description(
            "Computing kernels (%d,%d)" % (M.ncols(), M.nrows())
        )

    # compute the right kernel, store it in a dictionary
    try:
        ker = M.__pari__().matker(flag=1).mattranspose().sage()
    except:
        picklefile = "matrix.pkl"
        with open(picklefile, "wb") as f:
            pickle.dump(M, f)
        print(
            f"""Error in computing matrix kernel of size {M.ncols()} X {M.nrows()}.
        Try increasing the size of the PARI stack. The matrix has been stored in {picklefile}."""
        )
        raise
    return ker


def _compute_kernel2(source_dim, target_dim, rels, basis_indices, check=False):
    inds_complement = list(set(range(target_dim)) - set(basis_indices))

    M = matrix(ZZ, nrows=source_dim, ncols=target_dim)
    for source, target, coeff in rels:
        M[source, target] = coeff

    b = M[:, basis_indices]

    sol = M[:, inds_complement].solve_right(b)

    coker_mat = matrix(ZZ, nrows=target_dim, ncols=len(basis_indices))
    coker_mat[inds_complement] = -sol
    coker_mat[basis_indices] = matrix.identity(len(basis_indices))
    coker_mat = coker_mat.T
    if check:
        assert (coker_mat * M.T).norm() == 0
    return coker_mat


def all_abijk(BGG, s=0, subset=[], half_only=False):
    """Return a list of all the (a,b) in the bigraded table.

    Parameters
    ----------
    BGG : BGGComplex
    s : int (default: 0)
        The cohomological degree of Hochschild complex
    subset : list[int]
        The parabolic subset
    half_only : bool
        If True then only half of the table is returned; the other half can be deduced by symmetry.
    """
    dim_n = len(ModuleFactory(BGG.LA).parabolic_n_basis(subset))
    output = []
    # the maximal value of a is the dimension of n, +1 depending on the parity of s.
    s_mod = s % 2
    a_max = dim_n + (1 + s_mod) % 2

    # make a list of all the pairs (a,b)
    for a_iterator in range(a_max):
        a = 2 * a_iterator + s_mod
        for b in range(a + 1):
            if (a + b) % 2 == 0:
                output.append((a, b, (a - b) // 2, (a + b) // 2, s - a))

    # if half_only, throw out half of the values
    # we throw something away depending on the sum a+b
    if half_only:
        new_out = []
        max_a = max(s[0] for s in output)
        for a, b, i, j, k in output:
            if a + b <= max_a + 1:
                new_out.append((a, b, i, j, k))
        return sorted(new_out, key=lambda s: (s[0] + s[1], s[1], s[0]))
    else:
        return output


def extend_from_symmetry(table, max_a=None):
    """Use symmetry to compute an entire bigraded table from just the top half."""
    if max_a is None:
        max_a = max(a for a, b in table.keys())
    max_a = max_a + (max_a % 2)
    for (a, b), v in list(table.items()):
        table[(max_a - b, max_a - a)] = v
    return table


def display_bigraded_table(table, text_only=False):
    """Generate LaTeX code to display the bigraded table.

    Takes a dictionary (a,b) -> LaTeX string.
    If extend_half = True, we extend the table by using the symmetry
    """
    # Find all the values of a,b
    a_set = set()
    b_set = set()
    for a, b in table.keys():
        a_set.add(a)
        b_set.add(b)
    a_set = sorted(a_set)
    b_set = sorted(b_set)

    # string telling \array how to format the columns
    column_string = r"{r|" + " ".join("l" * len(a_set)) + r"}"
    rows = list()

    # for each a,b that's actually in the dictionary
    for a in a_set:
        # add the left most entry of the row
        row_strings = [r"{\scriptstyle i+j=" + str(a) + r"}"]
        for b in b_set:
            if (a, b) in table:
                row_strings.append(str(table[(a, b)]))
            else:
                row_strings.append("")
        # separate all the entries by an & symbol
        rows.append(r"&".join(row_strings))

    # add the last row
    rows.append(
        r"\hline h^{i,j}&"
        + "&".join([r"{\scriptstyle j-i=" + str(b) + r"}" for b in b_set])
    )
    all_strings = "\\\\\n\t".join(rows)

    # display the generated latex code.
    display_string = (
        r"\begin{array}"
        + column_string
        + "\n\t"
        + all_strings
        + "\n\\end{array}"
    )
    if not text_only:
        display(Math(display_string))
    return display_string


def display_cohomology_stats(cohom_dic, BGG, text_only=False):
    """Display multiplicities and dimensions of all the entries in bigraded table."""
    multiplicities = defaultdict(int)

    bgg_cohomology = BGGCohomology(BGG)
    weight_set = bgg_cohomology.weight_set

    for cohom in cohom_dic.values():
        if cohom is None:
            continue
        for mu, mult in cohom:
            multiplicities[mu] += mult

    if len(multiplicities) == 0:
        return

    betti_numbers = dict()
    latex_strings = dict()
    total_dim = 0
    for mu in sorted(multiplicities.keys()):
        betti_numbers[mu] = weight_set.highest_weight_rep_dim(mu)
        latex_strings[mu] = bgg_cohomology.tuple_to_latex((mu, 1))

        total_dim += betti_numbers[mu] * multiplicities[mu]

    rows = [
        r"\text{module}&\text{multiplicity}&\text{dimension} \\ \hline \text{all}&&"
        + str(total_dim)
        + " "
    ]
    for mu in multiplicities.keys():
        rows.append(
            "&".join(
                [
                    latex_strings[mu],
                    str(multiplicities[mu]),
                    str(betti_numbers[mu]),
                ]
            )
        )

    array_string = (
        "\\begin{array}{rll}\n\t" + "\\\\\n\t".join(rows) + "\n\\end{array}"
    )
    if not text_only:
        display(Math(array_string))
    return array_string


def prepare_texfile(tables, title=None):
    """Given some tables, join them together and create a LaTeX document to contain them."""
    preamble = [
        r"\documentclass[crop,border=2mm]{standalone}",
        r"",
        r"\usepackage{amsfonts, amsmath, amssymb}",
        r"",
        r"\begin{document}",
        r"\begin{tabular}{l}",
        r"",
    ]
    preamble = "\n".join(preamble)

    post = [r"", r"\end{tabular}", r"\end{document}", r""]
    post = "\n".join(post)

    if title is not None:
        preamble += r"{\huge " + title + r"}\\ \\" + "\n\n"

    table_formulas = "\n".join(
        ["\n$\\displaystyle\n" + tab + "\n$ \\\\ \\\\\n" for tab in tables]
    )

    document = preamble + table_formulas + post

    return document
