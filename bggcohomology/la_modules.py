"""
Lie algebra modules with weight decomposition and their BGG cohomology

Provides functionality to construct weight modules with a Lie algebra action. Given a BGG complex, it can
subsequently compute the cohomology of the module. See the tutorial notebook for example usage.
"""

from IPython.display import display, Math, Latex
import itertools
from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method
from collections import defaultdict
import numpy as np

from . import cohomology

INT_PRECISION = np.int32

__all__ = [
    "LieAlgebraCompositeModule",
    "BGGCohomology",
    "ModuleComponent",
    "ModuleFactory",
    "WeightSet",
]


class LieAlgebraCompositeModule:
    """Class encoding a Lie algebra weight module.

    Parameters
    ----------
    factory : ModuleFactory
        Factory object encoding the building blocks for the lie algebra module
    components : List[List[tuple(str, int, str)]]
        list of lists of triples e.g. of form `[[('g', 2, 'sym'),('n', 3, 'wedge')]]` to denote the
        module :math:`\\mathrm{Sym}^2\\mathfrak g\\otimes \\wedge^3\\mathfrak n`. 
        Or `[[('g',1,'sym')],[('u',1,'sym')]]` to denote :math:`\\mathfrak g\\oplus\\mathfrak u`.
    component_dic : dict[str, ModuleComponent]
        dictionary mapping keys like 'g' or 'n' to their respective lie algebra component

    Attributes
    ----------
    components : List[List[tuple(str, int, str)]]
    component_dic : dict[str, ModuleComponent]
    factory : ModuleFactory
    weight_dic : Dict[int, np.array[np.int32]]
        Dictionary mapping the basis indices to the weights of the Lie algebra elements, 
        encoded as vector with length given by rank of Lie algebra.
    modules : Dict[str, np.array[np.int32]]
        Dictionary mapping module keys (e.g. 'g' or 'n') to a matrix
        list of integer indices of its basis
    len_basis : int
        Number of different indices occuring in the bases of any module
    max_index : int
        Largest index that occurs in the basis of any module
    weight_components : dict[tuple[int], tuple(int,np.ndarray[np.int32, np.int32])]
        dictionary mapping a weight tuple to a pair (i, basis) where i is an integer indexing the 
        direct sum component, and basis is basis of the weight component in same format as output 
        of self.compute_weight_components.
    dimensions : dict[str, int]
        Dictionary containing the total dimension of each weight component
    dimensions_components : list[dict[str, int]]
        For each direct sum component seperately, a dictionary containing the 
        total dimension of each weight component
    weight_comp_index_numbers : dict[str, dict[tuple(int), int]]
        For each weight give a dictionary that maps a tuple encoding the 
        a basis element together with an index encoding in which 
        direct sum component it lies, to an index encoding
        this basis element as basis element of the weight component.
    weight_comp_direct_sum_index_numbers : dict[str, dict[tuple(int), int]]
        For each weight give a dictionary that maps a tuple encoding the 
        a basis element together with an index encoding in which 
        direct sum component it lies, to an index encoding
        this basis element as basis element of the direct sum component of
        the weight component.
    type_lists : list[list[str]]
        Gives the type of the lie_algebra for each tensor component. For example for
        [[('g',2,'sym'),('u',1,'sym')],[('n',2,'wedge')]] this gives
        [['g','g','u'],['n','n']].
    slice_lists : list[list[tuple(str,int,int,int)]]
        For each direct sum component, store a list which gives a slice for the entire
        tensor component for each individual tensor slot.
    action_tensor_dic : dict[str, np.array[np.int32, np.int32, np.int32]]
        For each Lie algebra type, give the order 3 tensor encoding the structure
        coefficients of the action.
    """

    def __init__(self, factory, components, component_dic):
        self.components = components
        self.component_dic = component_dic
        self.factory = factory
        self.weight_dic = factory.weight_dic
        self.modules = {
            k: component_dic[k].basis for k in component_dic.keys()
        }  # basis of each component type
        self._latex_basis_dic = None

        # length of all the indices occurring in any module
        self.len_basis = len(set(itertools.chain.from_iterable(self.modules.values())))
        self.max_index = max(itertools.chain.from_iterable(self.modules.values()))

        # Compute a basis for the module, and group by weight to get a basis for each weight component
        self.weight_components = self.initialize_weight_components()

        # Compute the dimension of each weight component
        self.dimensions = {
            w: sum((len(c) for _, c in p)) for w, p in self.weight_components.items()
        }

        # Compute the dimension of each weight component for each direct sum component
        self.dimensions_components = [
            {
                w: sum((len(c) for c_num, c in p if c_num == j))
                for w, p in self.weight_components.items()
            }
            for j in range(len(self.components))
        ]

        # Creates a dictionary that assigns to each weight a dictionary sending row+direct sum component to an index.
        self.weight_comp_index_numbers = dict()
        self.weight_comp_direct_sum_index_numbers = dict()
        for mu, components in self.weight_components.items():
            i = 0
            basis_dic = dict()
            basis_dic_direct_sum = dict()
            for c, basis in components:
                j = 0
                for b in basis:
                    basis_dic[tuple(list(b) + [c])] = i
                    basis_dic_direct_sum[tuple(list(b) + [c])] = j
                    i += 1
                    j += 1
            self.weight_comp_index_numbers[mu] = basis_dic
            self.weight_comp_direct_sum_index_numbers[mu] = basis_dic_direct_sum

        # for each direct sum component, store a list which lists the module type for each tensor slot
        self.type_lists = []
        for comp in self.components:
            type_list = []
            for c in comp:
                type_list += [c[0]] * c[1]
            self.type_lists.append(type_list)

        # For each direct sum component, store a list which gives a slice for the entire tensor component for
        # each individual tensor slot
        self.slice_lists = []
        for comp in self.components:
            slice_list = []
            start_slice = 0
            for c in comp:
                end_slice = start_slice + c[1]
                for i in range(c[1]):
                    slice_list.append((c[2], i, start_slice, end_slice))
                start_slice = end_slice
            self.slice_lists.append(slice_list)

        self.action_tensor_dic = dict()
        for key, mod in self.component_dic.items():
            self.action_tensor_dic[key] = self.get_action_tensor(mod)

    def construct_component(self, component):
        r"""Construct array of integers representing basis of direct sum component.

        Parameters
        ----------
        component : List[tuple(str, int, str)]
            The direct sum component.  This is one entry of `self.components`
        
        Returns
        -------
        np.ndarray[np.int32, np.int32]
            multi-indices describing basis of component. If we for example have
            [('g',4,'sym'),('n',5,'wedge')]
            as input, then the output will consist of an array of shape {math1}
            The first 4 columns are ordered 4-tuples (with replacement)
            of basis indices of {math2}, and the
            last 5 columns are ordered tuples  (without replacement) of basis indices of {math3}.
        """.format(
            math1=r":math:`9\times {\dim \mathfrak g+3}\choose 4\cdot {\dim \mathfrak n}\choose 5`",
            math2=r":math:`\mahtfrak g`",
            math3=r":math:`\mathfrak n`",
        )

        tensor_components = []
        n_components = len(component)
        for module, n_inputs, tensor_type in component:
            if n_inputs == 1:
                tensor_components.append(
                    np.array(self.modules[module], dtype=INT_PRECISION).reshape((-1, 1))
                )
            else:
                if tensor_type == "sym":  # Symmetric power
                    tensor_components.append(
                        np.array(
                            list(
                                itertools.combinations_with_replacement(
                                    self.modules[module], n_inputs
                                )
                            ),
                            dtype=INT_PRECISION,
                        )
                    )
                elif tensor_type == "wedge":  # Wedge power
                    mod_dim = len(
                        self.modules[module]
                    )  # Raise error if wedge power is too high
                    if n_inputs > mod_dim:
                        raise ValueError(
                            "Cannot take %d-fold wedge power of %d-dimensional module '%s'"
                            % (n_inputs, mod_dim, module)
                        )
                    tensor_components.append(
                        np.array(
                            list(
                                itertools.combinations(self.modules[module], n_inputs)
                            ),
                            dtype=INT_PRECISION,
                        )
                    )
                else:
                    raise ValueError(
                        "Tensor type %s is not recognized (allowed is 1 or 2)"
                        % tensor_type
                    )

        # Slicing using np.indices corresponds to taking a tensor product.
        # This is faster than building the matrix line by line using itertools.product.
        inds = np.indices(
            tuple(len(c) for c in tensor_components), dtype=INT_PRECISION
        ).reshape(n_components, -1)
        output = np.concatenate(
            [c[inds[i]] for i, c in enumerate(tensor_components)], axis=1
        )

        return output

    def compute_weight_components(self, direct_sum_component):
        """Construct the weight components of the module.

        Parameters
        ----------
        direct_sum_component : np.ndarray[np.int32, np.in32]
            output of `self.construct_component`

        Returns
        -------
        dict[tuple[int], np.ndarray[np.int32, np.int32]]
            dictionary mapping a weight tuple to a subset of the input basis.
        """
        # Matrix of weights associated to each lie algebra basis element
        weight_mat = np.array(
            [s[1] for s in sorted(self.weight_dic.items(), key=lambda t: t[0])],
            dtype=np.int64,
        )

        # Total weight for each element of the direct sum component
        tot_weight = np.sum(weight_mat[direct_sum_component], axis=1)

        # We will group by weight, so first we do a sort
        # Then we split every time the weight changes, and store in a dictionary
        argsort = np.lexsort(np.transpose(tot_weight))
        split_dic = {}
        last_weight = tot_weight[
            argsort[0]
        ]  # Keep track of weight, initial weight is first weight.
        current_inds = []
        for i in argsort:
            if np.all(np.equal(tot_weight[i], last_weight)):  # If weight is the same
                current_inds.append(i)
            else:  # If weight is different than before, store old basis elements in dict
                split_dic[tuple(last_weight)] = direct_sum_component[current_inds]
                current_inds = [i]
                last_weight = tot_weight[i]
        # Also register the very last weight in consideration
        split_dic[tuple(last_weight)] = direct_sum_component[current_inds]

        return split_dic

    def initialize_weight_components(self):
        """Compute a basis for each direct sum component and weight component.
        
        Returns
        -------
        dict[tuple[int], tuple(int,np.ndarray[np.int32, np.int32])]
            dictionary mapping a weight tuple to a pair `(i, basis)` where
            `i` is an integer indexing the direct sum component,
            and `basis` is basis of the weight component in same
            format as output of `self.compute_weight_components`.
        """
        weight_components = dict()
        for i, comp in enumerate(self.components):
            direct_sum_component = self.construct_component(comp)
            direct_sum_weight_components = self.compute_weight_components(
                direct_sum_component
            )
            for weight, basis in direct_sum_weight_components.items():
                if weight not in weight_components:
                    weight_components[weight] = list()
                weight_components[weight].append((i, basis))
        return weight_components

    def get_action_tensor(self, component):
        """Computes a tensor encoding the action for a given tensor component.

        Parameters
        ----------
        component : ModuleComponent
            Typically a value of `self.component_dic`

        Returns
        -------
        np.ndarray[int, int ,int]
            The shape of the tensor is (dim(n)+m, max_ind, 3),
            where m is some (typically small integer),
            dim(n) is the dimension of  the lie algebra n<g,
            max_ind is the largest index occurring in the tensor component.
            The last axis stores a triple (s, k, C_ijk) for each pair i,j.
            If i<dim(n), then C_ijk is the structure coefficient, similarly for k.
            If C_ijk is zero for all k, then the tuple is (0,0,0).
            For some i,j there are multiple non-zero C_ijk. If this happens, then
            s gives the row of the next non-zero C_ijk (the column is still j).
            If s = -1 then there are no further non-zero structure coefficients.
            The integer m is then the smallest such that the tensor is big enough.
        """

        # Previously we implemented the action as a dictionary, this will be our starting point
        action_mat = component.action

        # max index is largest index occurring in basis (+1 because of counting starting at 0)
        max_ind = max(component.basis) + 1

        dim_n = len(self.factory.basis["n"])

        # number of extra rows is largest number of non-zero C_ijk for any fixed i,j.
        extra_rows = [0] * max_ind
        for (_, j), v in action_mat.items():
            if len(v) > 1:
                extra_rows[j] += len(v) - 1
        n_extra_rows = max(extra_rows)

        # Initialize tensor of the correct shape
        action_tensor = np.zeros((dim_n + n_extra_rows, max_ind, 3), np.int64)

        # Keep track of the max index for each i.
        s_values = np.zeros(max_ind, np.int64) + dim_n

        # action matrix is of form (i,j):v, where v is a list of pairs (k,C_ijk)
        for (i, j), v in action_mat.items():
            l = len(v)
            v_iterator = iter(v.items())
            if l == 1:  # s=-1, and we can just store the triple in location i,j
                k, C_ijk = next(v_iterator)
                action_tensor[i, j] = (-1, k, C_ijk)
            else:  # Multiple non-zero C_ijk for this i,j
                s = s_values[
                    j
                ]  # Look up number of extra rows already added to this column
                s_values[j] += 1
                k, C_ijk = next(v_iterator)
                action_tensor[i, j] = (
                    s,
                    k,
                    C_ijk,
                )  # First one is still at position (i,j)
                count = 0
                for k, C_ijk in v_iterator:  # Other ones will be in the extra rows
                    count += 1
                    if count >= l - 1:  # For the last in the chain s=-1
                        action_tensor[s, j] = (-1, k, C_ijk)
                    else:  # There is another non-zero C_ijk, and it will be stored in row s+1
                        action_tensor[s, j] = (s + 1, k, C_ijk)
                    s = s_values[j]
                    s_values[j] += 1
        return action_tensor

    def _component_symbols_latex(self, component):
        """Compute a list of latex symbols to put in between indices to display them in latex.
        
        Parameters
        ----------
        component : List[tuple(str, int, str)]
            The direct sum component.  This is one entry of `self.components`
        
        Returns
        -------
        list[str]
            list of symbols to put in between indices
        """
        symbols = []
        for _, num, t in component:
            if t == "sym":
                type_string = r"\odot "
            else:  # t == 'wedge'
                type_string = r"\wedge "
            symbols += [type_string] * (num - 1)
            if num > 0:
                symbols += [r"\otimes "]
        symbols[-1] = ""
        return symbols

    def _component_latex_basis(self, comp_num, basis):
        """Given a basis of a weight component, convert all the indices to latex"""
        comp_symbols = self._component_symbols_latex(self.components[comp_num])
        basis_latex_dic = dict()
        for i, b in enumerate(basis):
            basis_strings = [self.factory.root_latex_dic[j] for j in b]
            basis_latex = "".join(
                list(itertools.chain.from_iterable(zip(basis_strings, comp_symbols)))
            )
            basis_latex_dic[i] = basis_latex
        return basis_latex_dic

    @property
    def latex_basis_dic(self):
        """Compute a dictionary sending each weight to a dictionary of latex strings encoding the 
        basis elements of the associated weight component 
        (one dictionary for each direct sum component)"""
        if self._latex_basis_dic is None:
            basis_dic = dict()
            for mu, wcomps in self.weight_components.items():
                mu_dict = dict()
                for comp_num, basis in wcomps:
                    mu_dict[comp_num] = self._component_latex_basis(comp_num, basis)
                basis_dic[mu] = mu_dict
            self._latex_basis_dic = basis_dic
        return self._latex_basis_dic

    def _weight_latex_basis(self, mu):
        """Turn a list of dictionaries into a list of all their values"""
        return list(
            itertools.chain.from_iterable(
                d.values() for d in self.latex_basis_dic[mu].values()
            )
        )

    def display_action(self, BGG, arrow, dominant_weight):
        """Display the action on a basis. Mainly for debugging purposes.

        Parameters
        ----------
        BGG : BGGComplex
        arrow : tuple(str, str)
            Pair of strings ecndoing an edge in the Bruhat graph
        dominant_weight : tuple[int]
            The dominant weight for which to compute the BGG complex
        """
        weight_set = WeightSet(BGG)
        vertex_weights = weight_set.get_vertex_weights(dominant_weight)
        mu = vertex_weights[arrow[0]]
        new_mu = vertex_weights[arrow[1]]

        mu_weight = weight_set.tuple_to_weight(dominant_weight)
        alpha_mu = BGG.weight_to_alpha_sum(mu_weight)

        bgg_map = BGG.compute_maps(alpha_mu)[arrow]

        source_counter = 0
        for comp_num, weight_comp in self.weight_components[mu]:
            basis_action = cohomology.action_on_basis(
                bgg_map, weight_comp, self, self.factory, comp_num
            )

            target_dic = self.weight_comp_index_numbers[new_mu]
            source_target_pairs = dict()
            for row in basis_action:
                target = target_dic[tuple(list(row[:-2]) + [comp_num])]
                source = row[-2]
                coeff = row[-1]
                if source not in source_target_pairs:
                    source_target_pairs[source] = []
                source_target_pairs[source].append((target, coeff))

            source_latex = self._weight_latex_basis(mu)
            target_latex = self._weight_latex_basis(new_mu)

            for source, targets in source_target_pairs.items():
                source_string = source_latex[source]
                target_strings = []

                first = True
                for target, coeff in targets:
                    if coeff == 1:
                        coeff_string = ""
                    elif coeff == -1:
                        coeff_string = "-"
                    else:
                        coeff_string = str(coeff)

                    if (not first) and (coeff > 0):
                        coeff_string = "+" + coeff_string
                    first = False
                    target_strings.append(
                        coeff_string + (r"(%d)" % target) + target_latex[target]
                    )

                display(
                    Math(
                        r"%d\colon \," % (source + source_counter)
                        + source_string
                        + r"\mapsto "
                        + "".join(target_strings)
                    )
                )
            source_counter += len(weight_comp)


class ModuleComponent:
    """Class encoding a building-block for lie algebra modules.

    This class is a data container.
    
    Parameters
    ----------
    basis : list[int]
        List of integers indexing the basis of a Lie algebra.
    action : dict[tuple(int,int), dict[int, int]]
        Dictionary encoding the structure coefficients :math:`C^k_{i,j}`. This is encoded
        by mapping (i,j) to {k: Cijk if Cijk !=0}.
    factory : ModuleFactory
        The ModuleFactory that created this instance of ModuleComponent
    """

    def __init__(self, basis, action, factory):
        self.basis = basis
        self.action = action
        self.factory = factory

    # these methods are all unused. It seems that we really just use this class as a
    # data container.
    # @staticmethod
    # def action_on_vector(i, X, action):
    #     """Compute the action of a Lie algebra element on a basis element.

    #     Parameters
    #     ----------
    #     i : int
    #         Basis index (element of `self.basis`)
    #     X : element of LieAlgebra

    #     Returns
    #     -------
    #     dict[int, int] : dictionary mapping basis index -> coefficient
    #     """
    #     output = defaultdict(int)
    #     for j, c1 in X.items():
    #         bracket = action[(i, j)]
    #         for k, c2 in bracket.items():
    #             output[k] += c1 * c2
    #     return dict(output)

    # @staticmethod
    # def add_dicts(dict1, dict2):
    #     """Helper function for merging two dicts, summing common keys"""
    #     for k, v in dict2.items():
    #         if k in dict1:
    #             dict1[k] += v
    #         else:
    #             dict1[k] = v
    #     return dict1

    # function is unused, and contains possible error (why is coefficient unused?)
    # def pbw_action_matrix(self, pbw_elt):
    #     """Given a PBW element, compute the matrix encoding the action of this PBW element.
    #     Output is encoded as a list of tuples (basis index, dict), where each dict
    #     consists of target index -> coefficient, iff coefficient is non-zero.
    #     (most dicts are empty in practice."""
    #     total = [(m, dict()) for m in self.basis]
    #     for monomial, coefficient in pbw_elt.monomial_coefficients().items():
    #         sub_total = [(m, {m: 1}) for m in self.basis]
    #         for term in monomial.to_word_list()[::-1]:
    #             index = self.factory.root_to_index[term]
    #             sub_total = [
    #                 (m, self.action_on_vector(index, image, self.action))
    #                 for m, image in sub_total
    #             ]
    #         total = [
    #             (m, self.add_dicts(t, s)) for ((m, t), (_, s)) in zip(total, sub_total)
    #         ]
    #     total = [(m, {k: v for k, v in d.items() if v != 0}) for m, d in total]
    #     return total


class ModuleFactory:
    """A factory class making ModuleComponent.

    It can create modules for (co)adjoint actions on parabolic subalgebras of the input Lie algebra.

    Parameters
    ----------
    lie_algebra : LieAlgebra
        The input lie algebra, typically created as `LieAlgebra(QQ, cartan_type=root_system)`.
        The Lie algebra is assumed to be simple.

    Attributes
    ----------
    lie_algebra : LieAlgebra
    lattice : RootSpace
        The root lattice associated the the simple Lie algebra
    rank : int
        Rank of the Lie algebra / root system
    lie_algebra_basis : dict[RootSpace.element_class, LieAlgebra.element_class]
        Dictionary mapping roots in the rootspace to basis elements of the Lie algebra
    sorted_basis : list[RootSpace.element_class]
        Elements of the root lattice indexing the basis, sorted such that this
        list starts with the negative roots.
    root_to_index : dict[RootSpace.element_class, int]
        Dictionary mapping elements of root lattice to their index in `self.sorted_basis`.
    g_basis : List[int]
        sorted list of indices corresponding to a basis of the entire Lie algebra
    index_to_lie_algebra : dict[int, LieAlgebra.element_class]
        Dictionary mapping basis indices to their corresponding basis element of the Lie algebra
    f_roots : List[RootSpace.element_class]
        List of negative roots
    e_roots : List[RootSpace.element_class]
        List of positive roots
    h_roots : List[RootSpace.element_class]
        List of roots belonging to Cartan subalgebra
    basis : Dict[str, List[int]]
        Dictionary with keys {'u','n','h','b',b+'} and values the list of indices
        corresponding to the subalgebra's of the Lie algebra.
    dual_root_dict : Dict[int, int]
        Dictionary mapping negative roots to their corresponding positive roots and vice versa,
        where the roots are represented by their integer indices. Indices corresponding
        to the Cartan subalgebra are mapped to themselves.
    weight_dic : Dict[int, np.array[np.int32]]
        Dictionary mapping the basis indices to the weights of the Lie algebra elements,
        encoded as vector with length given by rank of Lie algebra.
    """

    def __init__(self, lie_algebra):
        self.lie_algebra = lie_algebra
        self.lattice = (
            lie_algebra.weyl_group().domain().root_system.root_lattice()
        )  # Root lattice
        self.rank = self.lattice.rank()
        self.lie_algebra_basis = dict(self.lie_algebra.basis())

        # Associate to each root in the Lie algebra basis a unique index to be used as a basis subsequently.
        # For practical reasons we want the subspace `n` to have indices 0,...,dim(n)-1
        # Since those elements are the only with negative coefficients in the roots,
        # we can just sort by the first coefficient of the root to ensure this.
        # sorted_basis = sorted(self.lie_algebra_basis.keys(), key=lambda k: k.coefficients()[0])
        self.sorted_basis = list(lie_algebra.indices())[::-1]
        self.root_to_index = {k: i for i, k in enumerate(self.sorted_basis)}
        self.g_basis = sorted(self.root_to_index.values())
        self.index_to_lie_algebra = {
            i: self.lie_algebra_basis[k] for k, i in self.root_to_index.items()
        }

        # Encode seperately a list of negative and positive roots, and the Cartan.
        self.f_roots = list(self.lattice.negative_roots())
        self.e_roots = list(self.lattice.positive_roots())
        self.h_roots = self.lattice.alphacheck().values()

        self.root_latex_dic = {
            i: self.root_to_latex(root) for i, root in enumerate(self.sorted_basis)
        }

        # Make a list of indices for the (non parabolic) 'u','n','h','b',b+'
        self.basis = dict()
        self.basis["u"] = sorted([self.root_to_index[r] for r in self.e_roots])
        self.basis["n"] = sorted([self.root_to_index[r] for r in self.f_roots])
        self.basis["h"] = sorted([self.root_to_index[r] for r in self.h_roots])
        self.basis["b"] = sorted(
            self.basis["n"] + [self.root_to_index[r] for r in self.h_roots]
        )
        self.basis["b+"] = sorted(
            self.basis["u"] + [self.root_to_index[r] for r in self.h_roots]
        )

        # Make a dictionary mapping a root to its dual
        self.dual_root_dict = dict()
        for root in self.e_roots + self.f_roots:
            self.dual_root_dict[self.root_to_index[-root]] = self.root_to_index[root]
        for root in self.h_roots:
            self.dual_root_dict[self.root_to_index[root]] = self.root_to_index[root]

        # Make a dictionary encoding the associated weight for each basis element.
        # Weight is encoded as a np.array with length self.rank and dtype int.
        self.weight_dic = dict()
        for i, r in enumerate(self.sorted_basis):
            if (
                r.parent() == self.lattice
            ):  # If root is an e_root or f_root weight is just the monomial_coefficients
                self.weight_dic[i] = self.dic_to_vec(
                    r.monomial_coefficients(), self.rank
                )
            else:  # If the basis element comes from the Cartan, the weight is zero
                self.weight_dic[i] = np.zeros(self.rank, dtype=INT_PRECISION)

    @staticmethod
    def dic_to_vec(dic, rank):
        """Turn dict encoding sparse vector into dense vector.

        Parameters
        ----------
        dic : Dict[int, int]
            Dictionary index->coefficient of non-zero entries
        rank : int
            Lenght of the vector (i.e. rank of Lie algebra / root system)

        Returns
        -------
        np.array[int]
            Dense vector of length `rank`.
        """

        vec = np.zeros(rank, dtype=INT_PRECISION)
        for key, value in dic.items():
            vec[key - 1] = value
        return vec

    def parabolic_p_basis(self, subset=None):
        """Give parabolic p subalgebra.

        Parameters
        ----------
        subset : List[int] or `None` (default: None)
            Parabolic subset. If `None`, returns non-parabolic.

        Returns
        -------
        List[int]
            list of basis indices corresponding to this subalgebra.
            It is spanned by the subalgebra `b` and the positive whose components
            lie entirely in `subset`.
        """

        if subset is None:
            subset = []
        e_roots_in_span = [
            r
            for r in self.e_roots
            if set(r.monomial_coefficients().keys()).issubset(subset)
        ]
        basis = self.basis["b"] + [self.root_to_index[r] for r in e_roots_in_span]
        return sorted(basis)

    def parabolic_n_basis(self, subset=None):
        """Give parabolic n subalgebra.

        Parameters
        ----------
        subset : List[int] or `None` (default: None)
            Parabolic subset. If `None`, returns non-parabolic.
        
        Returns
        -------
        List[int]
            list of basis indices corresponding to this subalgebra.
            It is spanned by all negative roots whose components are not entirely
            contained entirely in `subset`.
        """

        if subset is None:
            subset = []
        f_roots_not_in_span = [
            r
            for r in self.f_roots
            if not set(r.monomial_coefficients().keys()).issubset(subset)
        ]
        basis = [self.root_to_index[r] for r in f_roots_not_in_span]
        return sorted(basis)

    def parabolic_u_basis(self, subset=None):
        """Give parabolic u subalgebra.

        Parameters
        ----------
        subset : List[int] or `None` (default: None)
            Parabolic subset. If `None`, returns non-parabolic.
        
        Returns
        -------
        List[int]
            list of basis indices corresponding to this subalgebra.
            It is spanned by all positive roots whose components are not entirely
            contained entirely in `subset`.
        """

        if subset is None:
            subset = []
        e_roots_not_in_span = [
            r
            for r in self.e_roots
            if not set(r.monomial_coefficients().keys()).issubset(subset)
        ]
        basis = [self.root_to_index[r] for r in e_roots_not_in_span]
        return sorted(basis)

    def adjoint_action_tensor(self, lie_algebra, module):
        """Compute structure coefficients of the adjoint action of subalgebra on a module.

        Parameters
        ----------
        lie_algbera : List[int]
            List of indices corresponding to the Lie subalgebra
        module : List[int]
            List of indices curresponding to a module (i.e. a subspace of the
            Lie algebra that is fixed under the adjoint action of the subalgebra)

        Returns
        -------
        dict[tuple(int, int), Dict[int,int]]
            Dictionary mapping (i,j) to
            a dictionary index->coeffcient encoding the non-zero structure
            coefficients :math:`C^k_{i,j}`.
        """

        action = defaultdict(dict)
        action_keys = set()
        for i, j in itertools.product(
            lie_algebra, module
        ):  # for all pairs of indices belonging to either subspace

            # Compute Lie bracket.
            bracket = (
                self.index_to_lie_algebra[i]
                .bracket(self.index_to_lie_algebra[j])
                .monomial_coefficients()
            )

            # Convert Lie bracket from monomial in roots to dict mapping index -> coefficient
            bracket_in_basis = {
                self.root_to_index[monomial]: coefficient
                for monomial, coefficient in bracket.items()
            }

            if len(bracket_in_basis) > 0:  # Store action only if it is non-trivial
                action[(i, j)] = bracket_in_basis

            # Store keys of non-zero action in a set to check if module is closed under adjoint action.
            for key in bracket_in_basis.keys():
                action_keys.add(key)

        if not action_keys.issubset(
            set(module)
        ):  # Throw an error if the module is not closed under adjoint action
            raise ValueError("The module is not closed under the adjoint action")

        return action

    def coadjoint_action_tensor(self, lie_algebra, module):
        """Compute structure coefficients of the coadjoint action of subalgebra on a module.

        Parameters
        ----------
        lie_algbera : List[int]
            List of indices corresponding to the Lie subalgebra
        module : List[int]
            List of indices curresponding to a module (i.e. a linear subspace of the Lie algebra)

        Returns
        -------
        dict[tuple(int, int), Dict[int,int]]
            Dictionary mapping (i,j) to
            a dictionary index->coeffcient encoding the non-zero structure
            coefficients :math:`C^k_{i,j}`.
        """

        action = defaultdict(dict)
        module_set = set(module)
        for i, k in itertools.product(
            lie_algebra, module
        ):  # for all pairs of indices belonging to either subspace
            # The alpha_k component of coadjoint action on alpha_j is given by (minus)
            # the alpha_j component of the bracket [alpha_i ,alpha_k_dual], using the Killing form.
            # So first we compute bracket [alpha_i,alpha_k_dual]
            k_dual = self.dual_root_dict[k]
            bracket = (
                self.index_to_lie_algebra[i]
                .bracket(self.index_to_lie_algebra[k_dual])
                .monomial_coefficients()
            )

            # We then iterate of the alpha_j. The alpha_j_dual component of `bracket` is the Killing form
            # <alpha_j, bracket>.
            for monomial, coefficient in bracket.items():
                dual_monomial = self.dual_root_dict[self.root_to_index[monomial]]
                if (
                    dual_monomial in module_set
                ):  # We restrict to the module, otherwise action is not well-defined
                    action[(i, dual_monomial)][
                        k
                    ] = -coefficient  # Minus sign comes from convention
        return action

    def build_component(
        self, subalgebra, action_type="ad", subset=None, acting_lie_algebra="n"
    ):
        """Build a ModuleComponent
        
        Parameters
        ----------
        subalgebra : {'g','n','u','b','b+','p'}
            Character indicating the type of subalgebra. Here 'g' is full Lie algbera,
            'n' corresponds to negative roots, 'u' to positive roots, 'b' to (negative) Borel,
            'b+' to positive Borel, and 'p' to (negative) parabolic.
        action_type : {'ad', 'coad'} (default: 'ad')
            Whether to use adjoint action ('ad') or coadjoint action ('coad')
        subset : List[int] or None (default: None)
            Subset indicating parabolic subalgebra. If `None`, use non-parabolic.
        acting_lie_algebra : {'g','n','u','b','b+','p'}, (default:  'n')
            The Lie (sub)algebra which acts on `subalgebra`.
            `subset` has no effect on this subalgebra, and therefore 'p' and 'b' are equivalent.
        
        Returns
        -------
        ModuleComponent
            The ModuleComponent storing the data for this module.
        """

        if subset is None:
            subset = []

        module_dic = {
            "g": self.g_basis,
            "n": self.parabolic_n_basis(subset),
            "u": self.parabolic_u_basis(subset),
            "p": self.parabolic_p_basis(subset),
            "h": self.basis["h"],
            "b+": self.basis["b+"],
            "b": self.parabolic_p_basis(None),
        }

        # The acting module shouldn't be parabolic. We could make this a feature.
        acting_module_dic = {
            "g": self.g_basis,
            "n": self.parabolic_n_basis(None),
            "u": self.parabolic_u_basis(None),
            "p": self.parabolic_p_basis(None),
            "h": self.basis["h"],
            "b+": self.basis["b+"],
            "b": self.parabolic_p_basis(None),
        }
        if subalgebra not in module_dic.keys():
            raise ValueError("Unknown subalgebra '%s'" % subalgebra)
        if acting_lie_algebra not in module_dic.keys():
            raise ValueError("Unknown subalgebra '%s'" % acting_lie_algebra)

        module = module_dic[subalgebra]
        lie_alg = acting_module_dic[acting_lie_algebra]

        if action_type == "ad":
            action = self.adjoint_action_tensor(lie_alg, module)
        elif action_type == "coad":
            action = self.coadjoint_action_tensor(lie_alg, module)
        else:
            raise ValueError("'%s' is not a valid type of action" % action_type)
        return ModuleComponent(module, action, self)

    def root_to_latex(self, root):
        """Convert a root to a latex expression
        
        Parameters
        ----------
        root : RootSpace.element_class

        Returns
        -------
        str
            String containing LaTeX expression for this root. 
        """
        root_type = ""

        if root in self.h_roots:
            root_type = "h"
        elif root in self.f_roots:
            root_type = "f"
        elif root in self.e_roots:
            root_type = "e"

        mon_coeffs = sorted(root.monomial_coefficients().items(), key=lambda x: x[0])
        root_index = "".join(str(i) * abs(n) for i, n in mon_coeffs)
        root_string = root_type + r"_{%s}" % root_index

        return root_string


class WeightSet:
    """Class to do simple computations with the weights of a weight module.

    Parameters
    ----------
    BGG : BGGComplex
    """

    def __init__(self, BGG):
        self.reduced_words = BGG.reduced_words
        self.weyl_dic = BGG.reduced_word_dic
        self.simple_roots = BGG.simple_roots
        self.rho = BGG.rho
        self.rank = BGG.rank
        self.pos_roots = [self.tuple_to_weight(w) for w in BGG.neg_roots]

        # Matrix of all simple roots, for faster matrix solving
        self.simple_root_matrix = matrix(
            [list(s.to_vector()) for s in self.simple_roots]
        ).transpose()

        self.action_dic, self.rho_action_dic = self.get_action_dic()

    def weight_to_tuple(self, weight):
        """Convert element of weight lattice to a sum of simple roots.

        Parameters
        ----------
        weight : RootSpace.element_class

        Returns
        -------
        tuple[int]
            tuple representing root as linear combination of simple roots
        
        """

        b = weight.to_vector()
        b = matrix(b).transpose()
        return tuple(self.simple_root_matrix.solve_right(b).transpose().list())

    def tuple_to_weight(self, t):
        """Inverse of `weight_to_tuple`
        
        Parameters
        ----------
        t : tuple[int]

        Returns
        -------
        RootSpace.element_class
        """
        return sum(int(a) * b for a, b in zip(t, self.simple_roots))

    def get_action_dic(self):
        """Compute weyl group action as well as action on rho.

        Returns
        -------
        Dict[str, np.array(np.int32, np.int32)] : dictionary mapping each string representing an
            element of the Weyl group to a matrix expressing the action on the simple roots.
        Dict[str, np.array(np.int32)] : dictionary mapping each string representing an
            element of the Weyl group to a vector representing the image
            of the dot action on rho.
        """
        action_dic = dict()
        rho_action_dic = dict()
        for s, w in self.weyl_dic.items():  # s is a string, w is a matrix
            # Compute action of w on every simple root, decompose result in simple roots, encode result as matrix.
            action_mat = []
            for mu in self.simple_roots:
                action_mat.append(self.weight_to_tuple(w.action(mu)))
            action_dic[s] = np.array(action_mat, dtype=INT_PRECISION)

            # Encode the dot action of w on rho.
            rho_action_dic[s] = np.array(
                self.weight_to_tuple(w.action(self.rho) - self.rho), dtype=INT_PRECISION
            )
        return action_dic, rho_action_dic

    def dot_action(self, w, mu):
        """Compute the dot action of w on mu
        
        Parameters
        ----------
        w : str
            string representing the weyl group element
        mu : iterable(int)
            the weight

        Returns
        -------
        np.array[np.int32]
            vector encoding the new weight
        """
        # The dot action w.mu = w(mu+rho)-rho = w*mu + (w*rho-rho).
        # The former term is given by action_dic, the latter by rho_action_dic
        return (
            np.matmul(self.action_dic[w].T, np.array(mu, dtype=INT_PRECISION))
            + self.rho_action_dic[w]
        )

    def is_dot_regular(self, mu):
        """Check if mu has a non-trivial stabilizer under the dot action
        
        Parameters
        ----------
        mu : iterable(int)
            The weight

        Returns
        -------
        bool
            `True` if the weight is dot-regular
        """
        for s in self.reduced_words[1:]:
            if np.all(self.dot_action(s, mu) == mu):
                return False
        # no stabilizer found
        return True

    def compute_weights(self, weights):
        """Finds dot-regular weights and associated dominant weights of a set of weights.

        Parameters
        ----------
        weights : iterable(iterable(int))
            Iterable of weights
        
        returns
        -------
        list(tuple[tuple(int), tuple(int), int])
            list of triples consisting of
            dot-regular weight, associated dominant, and the length of the Weyl group
            element making the weight dominant under the dot action.
        """

        regular_weights = []
        for mu in weights:
            if self.is_dot_regular(mu):
                mu_prime, w = self.make_dominant(mu)
                regular_weights.append((mu, tuple(mu_prime), len(w)))
        return regular_weights

    def is_dominant(self, mu):
        """Use sagemath built-in function to check if weight is dominant
        
        Parameters
        ----------
        mu : iterable(int)
            the weight

        Returns
        -------
        bool
            `True` if weight is dominant
        """

        return self.tuple_to_weight(mu).is_dominant()

    def make_dominant(self, mu):
        """For a dot-regular weight mu, w such that if w.mu is dominant.
        
         Such a w exists iff mu is dot-regular, in which case it is also unique.

        Parameters
        ----------
        mu : iterable(int)
            the dot-regular weight
        
        Returns
        -------
        tuple(int)
            The dominant weight w.mu
        str
            the string representing the Weyl group element w.
        """

        for w in self.reduced_words:
            new_mu = self.dot_action(w, mu)
            if self.is_dominant(new_mu):
                return new_mu, w
        else:
            raise ValueError(
                "Could not make weight %s dominant, probably it is not dot-regular."
            )

    def get_vertex_weights(self, mu):
        """For a given dot-regular mu, return its orbit under the dot-action.
        
        Parameters
        ----------
        mu : iterable(int)

        Returns
        -------
        list[tuple[int]]
            list of weights
        """

        vertex_weights = dict()
        for w in self.reduced_words:
            vertex_weights[w] = tuple(self.dot_action(w, mu))
        return vertex_weights

    def highest_weight_rep_dim(self, mu):
        """Gives dimension of highest weight representation of integral dominant weight.

        Parameters
        ----------
        mu : tuple(int)
            A integral dominant weight

        Returns
        -------
        int
            dimension of highest weight representation.
        """

        mu_weight = self.tuple_to_weight(mu)
        numerator = 1
        denominator = 1
        for alpha in self.pos_roots:
            numerator *= (mu_weight + self.rho).dot_product(alpha)
            denominator *= self.rho.dot_product(alpha)
        return numerator // denominator


class BGGCohomology:
    """Class for computing the BGG cohomology of a module.

    Parameters
    ----------
    BGG : BGGComplex
    weight_module : LieAlgebraCompositeModule or `None`
        The weight module to compute the cohomology of. If `None`,
        this class offers limited functionality.
    coker : Dict[tuple(int), matrix] or None (default: None)
        Dictionary giving a basis of a cokernel for each weight component
        The matrix is in the ordered basis of the weight component
        given by the LieAlgebraCompositeModule.
        If `None`, or if a key is not present in the dicitonary,
        the entire weight component is used as normal and no reduction is performed.
    pbars : iterable(tqdm) or `None` (default: `None`)
        Progress bars to send updates to. If `None`, this feature is disabled.
        Up to two progress bars are supported for more detailed information.

    Attributes
    ----------
    BGG : BGGComplex
    has_coker : bool
        True if `self.coker` is not None
    coker : Dict[tuple(int), matrix] or None
    weight_set : WeightSet
    weight_module : LieAlgebraCompositeModule
    weights : list(tuple(int))
        list of all the weights occuring in the weight module
    num_components : int
        the number of direct sum components
    regular_weights : list(tuple[tuple(int), tuple(int), int])
        list of triples consisting of dot-regular weight, associated dominant, and the length of 
        the Weyl group element making the weight dominant under the dot action.

    """

    def __init__(self, BGG, weight_module=None, coker=None, pbars=None):
        self.BGG = BGG
        self.BGG.compute_signs()  # Make sure BGG signs are computed.

        if coker is not None:
            self.has_coker = True
            self.coker = coker
        else:
            self.has_coker = False

        if pbars is None:
            self.pbar1 = None
            self.pbar2 = None
        if pbars is not None:
            self.pbar1, self.pbar2 = pbars

        self.weight_set = WeightSet(BGG)

        if weight_module is not None:
            self.weight_module = weight_module
            self.weights = weight_module.weight_components.keys()

            self.num_components = len(self.weight_module.components)

            self.regular_weights = self.weight_set.compute_weights(
                self.weights
            )  # Find dot-regular weights

    def cohomology_component(self, mu, i):
        """Compute cohomology BGG_i(mu).

        Parameters
        ----------
        mu : tuple(int)
            Dominant weight used in the BGG complex
        i : int
            Degree in which to compute cohomology, equivalently the column
            in the BGG complex taken.

        Returns
        -------
        int
            The dimension of the cohomology
        
        """
        if self.pbar1 is not None:
            self.pbar1.set_description(str(mu) + ", maps")
        mu_converted = self.BGG.weight_to_alpha_sum(self.BGG._tuple_to_weight(mu))
        self.BGG.compute_maps(mu_converted, pbar=self.pbar2, column=i)

        if self.pbar1 is not None:
            self.pbar1.set_description(str(mu) + ", diff")
        try:
            d_i, chain_dim = cohomology.compute_diff(self, mu, i)
            d_i_minus_1, _ = cohomology.compute_diff(self, mu, i - 1)
        except IndexError as err:
            print(mu, i)
            raise err

        if self.pbar1 is not None:
            self.pbar1.set_description(str(mu) + ", rank1")
        rank_1 = d_i.rank()
        if self.pbar1 is not None:
            self.pbar1.set_description(str(mu) + ", rank2")
        rank_2 = d_i_minus_1.rank()
        return chain_dim - rank_1 - rank_2

    @cached_method
    def cohomology(self, i, mu=None):
        """
        Compute full block of cohomology.

        This is done by computing BGG_i(mu) for all dot-regular mu appearing 
        in the weight module of length i.
        For a given weight mu, if there are no other weights of length i +/- 1 with the
        same associated dominant, then the weight component mu is isolated and the associated
        cohomology is the entire weight module.

        Parameters
        ----------
        i : int
            Degree in which to compute cohomology
        mu : tuple(int) or None (default: None)
            Optionally restrict computation to a particular weight, for example
            if computing for all weights is not interesting or not computationally feasible.

        Returns
        -------
        list[tuple(tuple(int), int)]
            List of pairs (weight, multiplicity),
            where multiplicity is always non-zero, and weight is a dominant weight.
            This gives the mutliplicity of each highest weight rep in the cohomology.
            If list is empty, the cohomology in this degree is trivial.
        """

        if self.pbar1 is not None:
            self.pbar1.set_description("Initializing")

        # Find all dot-regular weights of lenght i, together with their associated dominants.
        length_i_weights = [
            triplet for triplet in self.regular_weights if triplet[2] == i
        ]

        # Split the set of dominant weights in those which are isolated, and therefore don't require
        # us to run the BGG machinery, and those which are not isolated and require the BGG machinery.
        dominant_non_trivial = set()
        dominant_trivial = []
        for w, w_dom, _ in length_i_weights:
            if (mu is not None) and (w_dom != mu):
                continue
            for _, w_prime_dom, l in self.regular_weights:
                if w_prime_dom == w_dom and (l == i + 1 or l == i - 1):
                    dominant_non_trivial.add(w_dom)
                    break
            else:
                dominant_trivial.append((w, w_dom))

        cohomology_dic = defaultdict(int)

        # For isolated weights, multiplicity is just the module dimension
        for w, w_dom in dominant_trivial:
            if (self.has_coker) and (w in self.coker):
                cohom_dim = self.coker[w].nrows()
            else:
                cohom_dim = self.weight_module.dimensions[w]
            if cohom_dim > 0:
                cohomology_dic[w_dom] += cohom_dim

        if self.pbar1 is not None:
            self.pbar1.reset(total=len(dominant_non_trivial))
        # For non-isolated weights, multiplicity is computed through the BGG differential
        for w in dominant_non_trivial:
            cohom_dim = self.cohomology_component(w, i)
            if self.pbar1 is not None:
                self.pbar1.update()
            if cohom_dim > 0:
                cohomology_dic[w] += cohom_dim

        # Return cohomology as sorted list of highest weight vectors and multiplicities.
        # The sorting order is the same as how we normally sort weights
        return sorted(cohomology_dic.items(), key=lambda t: (sum(t[0]), t[0][::-1]))

    def betti_number(self, cohomology):
        """Compute Betti number from list of dominant weights and multiplicities

        Parameters
        ----------
        cohomology : list[tuple(tuple(int), int)]
            List of pairs (weight, multiplicity), typically output of 
            `self.cohomology`.
        
        Returns
        -------
        int
            The Betti number (total dimension)
        """
        if cohomology is None:
            return -1
        betti_num = 0
        for mu, mult in cohomology:
            dim = self.weight_set.highest_weight_rep_dim(mu)
            betti_num += dim * mult
        return betti_num

    def cohomology_LaTeX(
        self,
        i=None,
        mu=None,
        complex_string="",
        only_non_zero=True,
        print_betti=False,
        print_modules=True,
        only_strings=False,
        compact=False,
        skip_zero=False,
    ):
        """Compute cohomology, and display it in a pretty way using LaTeX

        Parameters
        ----------
        i : int or None (defualt: None)
            degree to compute cohomology. If none, compute in all degrees
        mu : tuple(int) or None (default: None)
            weight for which to compute cohomoly, if None compute for all
        complex_string : str (default: "")
            an optional string to print cohomology as `H^i(complex_string) = ...`
        only_non_zer0: bool (default: True)
            If True, print nothing if the cohomology is trivial / 0
        print_betti : bool (default: False)
            Print the Betti numbers
        print_modules : bool (default: True)
            Print a string to represent the dominant weights occuring
        only_strings : bool (default: False)
            Don't print anything, just return a string
        compact : bool (default : False)
            Use more compact notation for the dominant weights
        skip_zero : bool (default: False)
            Skip the zeroth degree of cohomology if `i is None`
        """

        # If there is a complex_string, insert it between brackets, otherwise no brackets.
        if len(complex_string) > 0:
            display_string = r"(%s)=" % complex_string
        else:
            display_string = r"="

        # compute cohomology. If cohomology is trivial and only_non_zero is true, return nothing.
        if i is None:
            if skip_zero:
                all_degrees = range(1, self.BGG.max_word_length + 1)
            else:
                all_degrees = range(self.BGG.max_word_length + 1)
            cohoms = [(j, self.cohomology(j, mu=mu)) for j in all_degrees]

            max_len = max([len(cohom) for _, cohom in cohoms])
            if (max_len == 0) and not skip_zero:
                if only_strings:
                    return "0"
                else:
                    display(Math(r"\mathrm H^\bullet" + display_string + "0"))
        else:
            cohom_i = self.cohomology(i, mu=mu)
            if (
                len(cohom_i) == 0 and not only_non_zero
            ):  # particular i, and zero cohomology:
                if only_strings:
                    return "0"
                else:
                    display(Math(r"\mathrm H^{" + str(i) + r"}" + display_string + "0"))
            cohoms = [(i, cohom_i)]

        for i, cohom in cohoms:
            if (not only_non_zero) or (len(cohom) > 0):
                # Print the decomposition into highest weight modules.
                if print_modules:
                    # Get LaTeX string of the highest weights + multiplicities
                    latex = self.cohom_to_latex(cohom, compact=compact)

                    # Display the cohomology in the notebook using LaTeX rendering
                    if only_strings:
                        return latex
                    else:
                        display(Math(r"\mathrm H^{%d}" % i + display_string + latex))

                # Print just dimension of cohomology
                if print_betti:
                    betti_num = self.betti_number(cohom)
                    if only_strings:
                        return str(betti_num)
                    else:
                        display(
                            Math(
                                r"\mathrm b^{%d}" % i + display_string + str(betti_num)
                            )
                        )

    def tuple_to_latex(self, tup, compact=False):
        """ Get LaTeX string representing a tuple of highest weight vector and it's multiplicity
        
        Parameters
        ----------
        tup : (tuple(int), int)
            Pair of dominant weight and its associated multiplicity
        compact : bool (default False)
            Whether or not to use more compact notation

        Returns
        -------
        str
            LaTeX string representing the highest weight module with multiplicity
        """
        (mu, mult) = tup
        if compact:  # compact display
            alpha_string = ",".join([str(m) for m in mu])
            if sum(mu) == 0:
                if mult > 1:
                    return r"\mathbb{C}^{%d}" % mult
                else:
                    return r"\mathbb{C}"
            if mult > 1:
                return r"L_{%s}^{%d}" % (alpha_string, mult)
            else:
                return r"L_{%s}" % alpha_string

        else:  # not compact display
            # Each entry mu_i in the tuple mu represents a simple root. We display it as mu_i alpha_i
            alphas = []
            for i, m in enumerate(mu):
                if m == 1:
                    alphas.append(r"\alpha_{%d}" % (i + 1))
                elif m != 0:
                    alphas.append(r" %d\alpha_{%d}" % (m, i + 1))

            # If all entries are zero, we just display the string '0' to represent zero weight,
            # otherwise we join the mu_i alpha_i together with a + operator in between
            if mult > 1:
                if len(alphas) == 0:
                    return r"\mathbb{C}^{%d}" % mult
                else:
                    alphas_string = r"+".join(alphas)
                    return r"L\left(%s\right)^{%d}" % (alphas_string, mult)
            else:
                if len(alphas) == 0:
                    return r"\mathbb{C}"
                else:
                    alphas_string = r"+".join(alphas)
                    return r"L\left(%s\right)" % alphas_string

    def cohom_to_latex(self, cohom, compact=False):
        """Represent output of `self.cohomology` by a LaTeX string
        
        Parameters
        ----------
        cohom : list[tuple(tuple(int), int)]
            List of pairs of dominant weight and multiplicity,
            typically output of `self.cohomology`
        compact : bool (default: False)
            Whether or not to use more compact notation
        """

        # Entry hasn't been computed yet
        if cohom is None:
            return "???"
        elif len(cohom) > 0:
            tuples = [self.tuple_to_latex(c, compact=compact) for c in cohom]
            if compact:
                return r"".join(tuples)
            else:
                return r"\oplus ".join(tuples)
        else:  # If there is no cohomology just print the string '0'
            return r"0"

    def _display_coker(self, mu, transpose=False):
        """Display the cokernel of a quotient module. 
        This is mainly implemented for debugging purposes."""
        if self.has_coker:
            if mu in self.coker:
                if not transpose:
                    coker_mat = self.coker[mu]
                    latex_basis = self.weight_module._weight_latex_basis(mu)
                    for row_num, row in enumerate(coker_mat.rows()):
                        row_strings = [r"%d\colon\," % row_num]

                        first = True
                        for i, c in enumerate(row):
                            if c != 0:
                                latex_i = (r"(%d)" % i) + latex_basis[i]
                                if c == 1:
                                    row_string = latex_i
                                elif c == -1:
                                    row_string = "-" + latex_i
                                else:
                                    row_string = str(c) + latex_i
                                if first:
                                    row_strings.append(row_string)
                                    first = False
                                else:
                                    if c > 0:
                                        row_strings.append("+" + row_string)
                                    else:
                                        row_strings.append(row_string)
                        display(Math("".join(row_strings)))

                else:  # it's transposed
                    coker_mat = self.coker[mu]
                    latex_basis = self.weight_module._weight_latex_basis(mu)
                    for col_num, col in enumerate(coker_mat.columns()):
                        col_strings = [
                            (r"(%d)" % col_num) + latex_basis[col_num] + r"\mapsto \,"
                        ]

                        first = True
                        for i, c in enumerate(col):
                            if c != 0:
                                latex_i = r"(%d)" % i
                                if c == 1:
                                    col_string = latex_i
                                elif c == -1:
                                    col_string = "-" + latex_i
                                else:
                                    col_string = str(c) + latex_i
                                if first:
                                    col_strings.append(col_string)
                                    first = False
                                else:
                                    if c > 0:
                                        col_strings.append("+" + col_string)
                                    else:
                                        col_strings.append(col_string)
                        display(Math("".join(col_strings)))
        else:
            print("No cokernel (weight = %s)" % str(mu))
