"""
Lie algebra modules with weight decomposition and their BGG cohomology

Provides functionality to construct weight modules with a Lie algebra action. Given a BGG complex, it can
subsequently compute the cohomology of the module. See the example notebooks on https://github.com/RikVoorhaar/BGG
for explanation of usage.
"""

# import numpy_indexed as npi
from IPython.display import display, Math, Latex
from sympy.utilities.iterables import subsets
from sage.rings.integer_ring import ZZ
import itertools
from sage.matrix.constructor import matrix
from collections import defaultdict
import numpy as np
import warnings

import cohomology

INT_PRECISION = np.int32

class FastLieAlgebraCompositeModule:
    """Class encoding a Lie algebra weight module. Input:
    - `weight_dic`: a dictionary of indices -> flat np.array of length equal to rank of lie algebra
    - `components: A list of lists of three-tuples. The entries of the tuples are respectively
    the key of component_dic, the power (an int) and a string either 'wedge' or 'sym'
    - `component_dic`: a dictionary with values instances of FastModuleComponent"""

    def __init__(self, factory, components, component_dic):
        self.components = components
        self.component_dic = component_dic
        self.factory = factory
        self.weight_dic = factory.weight_dic
        self.modules = {k: component_dic[k].basis for k in component_dic.keys()} #basis of each component type

        # length of all the indices occurring in any module
        self.len_basis = len(set(itertools.chain.from_iterable(self.modules.values())))
        self.max_index = max(itertools.chain.from_iterable(self.modules.values()))

        if self.len_basis*5 < self.max_index:
            warnings.warn("""Maximum index %d is much higher than length of module basis %d. 
                This may cause slowdown""" % (self.max_index,self.len_basis))

        # To compute the hash fast, cache a power of 33 for each basis element mod 2^32
        self.pow_array = 33 ** np.arange(self.max_index+1, dtype=INT_PRECISION)

        # For each direct sum component, compute a list of prime powers of 7919 incrementing at each tensor component
        self.component_primes = [self.compute_component_primes(index) for index in range(len(self.components))]

        # Compute a basis for the module, and group by weight to get a basis for each weight component
        self.weight_components = self.initialize_weight_components()

        # Compute the dimension of each weight component
        self.dimensions = {w: sum((len(c) for _, c in p)) for w, p in self.weight_components.items()}

        # Compute the dimension of each weight component for each direct sum component
        self.dimensions_components = [{w: sum((len(c) for c_num, c in p if c_num == j))
                                       for w, p in self.weight_components.items()}
                                      for j in range(len(self.components))]

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


        self.hash_dic = dict()
        for comps in self.weight_components.values():
            for comp in comps:
                for h, v in zip(self.get_hash(comp), comp[1]):
                    self.hash_dic[h] = v

        self.action_tensor_dic = dict()
        for key, mod in self.component_dic.items():
            self.action_tensor_dic[key] = self.get_action_tensor(mod)

    def construct_component(self, component):
        """Given a list of tuples representing tensor components, return an np array of integers. Each row
        represents one basis element of the tensor product"""
        tensor_components = []
        n_components = len(component)
        for module, n_inputs, tensor_type in component:
            if n_inputs == 1:
                tensor_components.append(np.array(self.modules[module], dtype=INT_PRECISION).reshape((-1, 1)))
            else:
                if tensor_type == 'sym':  # Symmetric power
                    tensor_components.append(
                        np.array(list(itertools.combinations_with_replacement(self.modules[module], n_inputs)),
                                 dtype=INT_PRECISION)
                    )
                elif tensor_type == 'wedge':  # Wedge power
                    mod_dim = len(self.modules[module])  # Raise error if wedge power is too high
                    if n_inputs > mod_dim:
                        raise ValueError('Cannot take %d-fold wedge power of %d-dimensional module \'%s\'' %
                                         (n_inputs, mod_dim, module))
                    tensor_components.append(
                        np.array(list(itertools.combinations(self.modules[module], n_inputs)), dtype=INT_PRECISION)
                    )
                else:
                    raise ValueError('Tensor type %s is not recognized (allowed is 1 or 2)' % tensor_type)

        # Slicing using np.indices corresponds to taking a tensor product.
        # This is faster than building the matrix line by line using itertools.product.
        inds = np.indices(tuple(len(c) for c in tensor_components), dtype=INT_PRECISION).reshape(n_components, -1)
        output = np.concatenate([c[inds[i]] for i, c in enumerate(tensor_components)], axis=1)

        return output

    # def compute_weight_components(self, direct_sum_component):
    #     """Compute the weight for each basis element, group the basis by weight.
    #     returns a dictionary mapping tuples of weights to basis of weight component."""
    #     weight_mat = np.array([s[1] for s in sorted(self.weight_dic.items(), key=lambda t: t[0])], dtype=INT_PRECISION)
    #     total_weight = np.sum(weight_mat[direct_sum_component], axis=1)
    #
    #     groupby = npi.group_by(total_weight)
    #     return {tuple(w): l for w, l in
    #             zip(groupby.unique, groupby.split_array_as_list(direct_sum_component))}

    def compute_weight_components(self, direct_sum_component):
        """Compute the weight for each basis element, group the basis by weight.
           returns a dictionary mapping tuples of weights to basis of weight component.
           This version doesn't use npi as dependency"""
        # Matrix of weights associated to each lie algebra basis element
        weight_mat = np.array([s[1] for s in sorted(self.weight_dic.items(), key=lambda t: t[0])], dtype=np.int64)

        # Total weight for each element of the direct sum component
        tot_weight = np.sum(weight_mat[direct_sum_component], axis=1)

        # We will group by weight, so first we do a sort
        # Then we split every time the weight changes, and store in a dictionary
        argsort = np.lexsort(np.transpose(tot_weight))
        split_dic = {}
        last_weight = tot_weight[argsort[0]]  # Keep track of weight, initial weight is first weight.
        current_inds = []
        for i in argsort:
            if np.all(np.equal(tot_weight[i], last_weight)): # If weight is the same
                current_inds.append(i)
            else: # If weight is different than before, store old basis elements in dict
                split_dic[tuple(last_weight)] = direct_sum_component[current_inds]
                current_inds = [i]
                last_weight = tot_weight[i]
        # Also register the very last weight in consideration
        split_dic[tuple(last_weight)] = direct_sum_component[current_inds]

        return split_dic


    def initialize_weight_components(self):
        """Compute a basis for each direct sum component, and compute basis for each weight component"""
        weight_components = dict()
        for i, comp in enumerate(self.components):
            direct_sum_component = self.construct_component(comp)
            direct_sum_weight_components = self.compute_weight_components(direct_sum_component)
            for weight, basis in direct_sum_weight_components.iteritems():
                if weight not in weight_components:
                    weight_components[weight] = list()
                weight_components[weight].append((i, basis))
        return weight_components

    def compute_component_primes(self, index):
        """Given index of a direct sum component, create a list of ints of same length as a basis element.
        For each tensor component the int is the same, and is a mod 2^32 power of 7919"""
        comp_type = [s[1] for s in self.components[index]]
        comp_primes = []
        for i, n in enumerate(comp_type):
            comp_primes += [i] * n
        return 7919 ** np.array(comp_primes, dtype=INT_PRECISION)  # * (541)**index

    def get_hash(self, weight_component, check_conflict=True):
        """ <<This functionality is deprecated.>>
        Compute the hash function for each weight component. Optionally check for hash conflicts. """
        index, vector_list = weight_component

        comp_primes = self.component_primes[index]

        hashes = comp_primes * self.pow_array[vector_list]
        hashes = np.sum(hashes, axis=1, dtype=INT_PRECISION)

        if check_conflict:
            hash_conflicts = len(hashes) - len(set(hashes))
            if hash_conflicts > 0:
                raise ValueError('Found %d hash conflict(s)!' % hash_conflicts)

        return hashes

    def get_action_tensor(self, component):
        """Computes a tensor encoding the action for a given tensor component.
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
        The integer m is then the smallest such that the tensor is big enough."""

        # Previously we implemented the action as a dictionary, this will be our starting point
        action_mat = component.action

        # max index is largest index occurring in basis (+1 because of counting starting at 0)
        max_ind = max(component.basis)+1

        dim_n = len(self.factory.basis['n'])

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
            if l == 1: # s=-1, and we can just store the triple in location i,j
                k, C_ijk = v.items()[0]
                action_tensor[i, j] = (-1, k, C_ijk)
            else:  # Multiple non-zero C_ijk for this i,j
                s = s_values[j] # Look up number of extra rows already added to this column
                s_values[j] += 1
                k, C_ijk = v.items()[0]
                action_tensor[i, j] = (s, k, C_ijk) # First one is still at position (i,j)
                count = 0
                for k, C_ijk in v.items()[1:]: # Other ones will be in the extra rows
                    count += 1
                    if count >= l - 1: # For the last in the chain s=-1
                        action_tensor[s, j] = (-1, k, C_ijk)
                    else: # There is another non-zero C_ijk, and it will be stored in row s+1
                        action_tensor[s, j] = (s + 1, k, C_ijk)
                    s = s_values[j]
                    s_values[j] += 1
        return action_tensor

    def set_pbw_action_matrix(self, pbw_elt):
        """<<DEPRECATED>>
        Given a PBW element, cache a matrix encoding the PBW action for each type of component in component_dic"""
        self.action_matrix = dict()
        for key, component in self.component_dic.items():
            self.action_matrix[key] = component.pbw_action_matrix(pbw_elt)

    def compute_pbw_action(self, weight_comp):
        """<<DEPRECATED>>
        Compute the action of a PBW element on the basis of a weight component.
        The action is computed column-wise as a list of hashes and coefficients.
        In the end the coefficients of the rows with the same hashes are summed.
        Returns a tuple of tensor indices of source, hashes of targets, coefficients.
        Only rows with non-zero coefficients are returned."""
        index, basis = weight_comp
        type_list = self.type_lists[index]
        output_basis = []
        output_hashes = []
        output_coefficients = []
        hashes = self.get_hash(weight_comp)

        for column in range(len(type_list)):
            action = self.action_matrix[type_list[column]]  # Retrieve cached action matrix for this column

            # Make a list of all the unique indices occurring in column.
            # This can be smaller than the list of indices occurring in the module type.
            unique_values = np.unique(basis[:, column])

            # Make a list of tuples (indices, action) such that the index occurs in column and action is non-trivial
            non_zero_actions = [(i, d) for i, d in action if (len(d) > 0 and i in unique_values)]

            # Unpack the list `non_zero_actions` so that it consists of triples (old_index,new_index,coefficient)
            action_queue = list(
                itertools.chain.from_iterable([[tuple([i] + list(t)) for t in d.items()] for i, d in non_zero_actions]))

            # Create a dictionary of slices identifying the rows containing a specific index in the current column
            split_indices = {i: np.flatnonzero(basis[:, column] == i) for i, _ in non_zero_actions}

            # Retrieve the hash multiplier for this column
            column_hash_modifier = self.component_primes[index][column]

            for old_index, new_index, coefficient in action_queue:
                # Changing the hash by this number gives the hash of the output
                hash_modifier = column_hash_modifier*(self.pow_array[new_index] - self.pow_array[old_index])
                new_hashes = hashes[split_indices[old_index]] + hash_modifier

                # Compute the signs for each of the rows. There are no signs for symmetric powers.
                tensor_type, sub_column, start_slice, end_slice = self.slice_lists[index][column]
                coefficient_array = None
                if tensor_type == 'sym':  # Symmetric power
                    coefficient_array = np.full(len(new_hashes), coefficient,dtype=INT_PRECISION)
                if tensor_type == 'wedge':  # Wedge power

                    # Retrieve the other columns of the current tensor component
                    slic = range(start_slice, end_slice)
                    slic.remove(column)
                    other_columns = basis[split_indices[old_index]][:, slic]

                    # For each row, compute the number of indices in the row that are strictly smaller than new_index
                    # Turn the result into a sign +1/-1 giving sign of the permutation sorting the row
                    sort_parity = (-1) ** (sub_column + np.sum(other_columns < new_index, axis=1) % 2)

                    # For each row, check if it already contains new_index, in which case permutation will have parity 0
                    is_double = 1 - np.sum(other_columns == new_index, axis=1)
                    permutation_signs = sort_parity * is_double

                    # Multiply the coefficient with the signs to get the output coefficient array
                    coefficient_array = coefficient * permutation_signs

                # append the resulting source tensor indices, output hashes, coefficients to an output list

                #output_basis.append(split_indices[old_index].reshape(-1))
                output_basis.append(hashes[split_indices[old_index]])

                output_hashes.append(new_hashes)
                output_coefficients.append(coefficient_array)

        # If there are no non-trivial actions, just ouptut an emtpy matrix. Otherwise subsequent code throws errors.
        if len(output_basis) == 0:
            return np.array([], dtype=INT_PRECISION), np.array([], dtype=INT_PRECISION)

        # Concatenate results for each sub_part
        output_basis = np.concatenate(output_basis)
        output_hashes = np.concatenate(output_hashes)
        output_coefficients = np.concatenate(output_coefficients)

        # Sum and merge coefficients of rows with same source tensor indices and target hash
        basis_hash_pairs = np.vstack([output_basis, output_hashes]).T
        gb = npi.group_by(basis_hash_pairs)
        image, coefficients = gb.sum(output_coefficients)

        # Take only non-zero coefficients, output result
        nonzero_coefficients = coefficients.nonzero()

        # check that the algorithm did not produce any garbage hashes
        for hash in output_hashes[nonzero_coefficients]:
            if hash not in self.hash_dic:
                raise ValueError('hash %s is unknown.' % hash)

        return image[nonzero_coefficients], coefficients[nonzero_coefficients]


class FastModuleComponent:
    """Class encoding a building-block for lie algebra modules. Input is a list of basis indices (assumed to be ints),
    a rank 3 tensor $C^k_{i,j}$ encoding the action of the Lie algebra, and a FastModuleFactory object creating this
    instance of FastModuleComponent.
     The action tensor is encoded as a dict (i,j)->{k: C^k_{i,j} for k in basis if C^k_{i,j}!=0}"""
    def __init__(self, basis, action, factory):
        self.basis = basis
        self.action = action
        self.factory = factory

    @staticmethod
    def action_on_vector(i, X, action):
        """Compute the action of a Lie algebra element X on basis element of index i.
        Return output as dictionary index -> coefficient"""
        output = defaultdict(int)
        for j, c1 in X.items():
            bracket = action[(i, j)]
            for k, c2 in bracket.items():
                output[k] += c1 * c2
        return dict(output)

    @staticmethod
    def add_dicts(dict1, dict2):
        """Helper function for merging two dicts, summing common keys"""
        for k, v in dict2.items():
            if k in dict1:
                dict1[k] += v
            else:
                dict1[k] = v
        return dict1

    def pbw_action_matrix(self, pbw_elt):
        """Given a PBW element, compute the matrix encoding the action of this PBW element.
        Output is encoded as a list of tuples (basis index, dict), where each dict
          consists of target index -> coefficient, iff coefficient is non-zero. (most dicts are empty in practice."""
        total = [(m, dict()) for m in self.basis]
        for monomial, coefficient in pbw_elt.monomial_coefficients().items():
            sub_total = [(m, {m: 1}) for m in self.basis]
            for term in monomial.to_word_list()[::-1]:
                index = self.factory.root_to_index[term]
                sub_total = [(m, self.action_on_vector(index, image, self.action)) for m, image in sub_total]
            total = [(m, self.add_dicts(t, s)) for ((m, t), (_, s)) in zip(total, sub_total)]
        total = [(m, {k: v for k, v in d.items() if v != 0}) for m, d in total]
        return total


class FastModuleFactory:
    """A factory class making FastModuleComponent. It can create modules for (co)adjoint actions on parabolic
    subalgebras of the input Lie algebra."""

    def __init__(self, lie_algebra):
        self.lie_algebra = lie_algebra
        self.lattice = lie_algebra.weyl_group().domain().root_system.root_lattice()  # Root lattice
        self.rank = self.lattice.rank()
        self.lie_algebra_basis = dict(self.lie_algebra.basis())

        # Associate to each root in the Lie algebra basis a unique index to be used as a basis subsequently.
        # For practical reasons we want the subspace `n` to have indices 0,...,dim(n)-1
        # Since those elements are the only with negative coefficients in the roots,
        # we can just sort by the first coefficient of the root to ensure this.
        sorted_basis = sorted(self.lie_algebra_basis.keys(), key=lambda k: k.coefficients()[0])
        self.root_to_index = {k: i for i, k in enumerate(sorted_basis)}
        self.g_basis = sorted(self.root_to_index.values())
        self.index_to_lie_algebra = {i: self.lie_algebra_basis[k] for k, i in self.root_to_index.items()}

        # Encode seperately a list of negative and positive roots, and the Cartan.
        self.f_roots = list(self.lattice.negative_roots())
        self.e_roots = list(self.lattice.positive_roots())
        self.h_roots = self.lattice.alphacheck().values()

        # Make a list of indices for the (non parabolic) 'g','u','n','b'
        self.basis = dict()
        self.basis['u'] = sorted([self.root_to_index[r] for r in self.e_roots])
        self.basis['n'] = sorted([self.root_to_index[r] for r in self.f_roots])
        self.basis['h'] = sorted([self.root_to_index[r] for r in self.h_roots])
        self.basis['b'] = sorted(self.basis['n'] + [self.root_to_index[r] for r in self.h_roots])
        self.basis['b+'] = sorted(self.basis['u'] + [self.root_to_index[r] for r in self.h_roots])

        # Make a dictionary mapping a root to its dual
        self.dual_root_dict = dict()
        for root in self.e_roots + self.f_roots:
            self.dual_root_dict[self.root_to_index[-root]] = self.root_to_index[root]
        for root in self.h_roots:
            self.dual_root_dict[self.root_to_index[root]] = self.root_to_index[root]

        # Make a dictionary encoding the associated weight for each basis element.
        # Weight is encoded as a np.array with length self.rank and dtype int.
        self.weight_dic = dict()
        for i, r in enumerate(sorted_basis):
            if r.parent() == self.lattice:  # If root is an e_root or f_root weight is just the monomial_coefficients
                self.weight_dic[i] = self.dic_to_vec(r.monomial_coefficients(), self.rank)
            else:  # If the basis element comes from the Cartan, the weight is zero
                self.weight_dic[i] = np.zeros(self.rank, dtype=INT_PRECISION)

    @staticmethod
    def dic_to_vec(dic, rank):
        """Helper function turning a (sparse) dic of index->coefficient into a dense np.array of given length (rank)"""

        vec = np.zeros(rank, dtype=INT_PRECISION)
        for key, value in dic.items():
            vec[key - 1] = value
        return vec

    def parabolic_p_basis(self, subset=None):
        """Give parabolic p_subalgebra.
         It is spanned by the subalgebra b and the e_roots whose components lie entirely in subset."""

        if subset is None:
            subset = []
        e_roots_in_span = [r for r in self.e_roots if set(r.monomial_coefficients().keys()).issubset(subset)]
        basis = self.basis['b'] + [self.root_to_index[r] for r in e_roots_in_span]
        return sorted(basis)

    def parabolic_n_basis(self, subset=None):
        """Give parabolic n_subalgebra.
         It is spanned by all the f_roots whose components are not entirely contained in subset"""

        if subset is None:
            subset = []
        f_roots_not_in_span = [r for r in self.f_roots if not set(r.monomial_coefficients().keys()).issubset(subset)]
        basis = [self.root_to_index[r] for r in f_roots_not_in_span]
        return sorted(basis)

    def parabolic_u_basis(self, subset=None):
        """Give parabolic u_subalgebra.
         It is spanned by all the e_roots whose components are not entirely contained in subset"""

        if subset is None:
            subset = []
        e_roots_not_in_span = [r for r in self.e_roots if not set(r.monomial_coefficients().keys()).issubset(subset)]
        basis = [self.root_to_index[r] for r in e_roots_not_in_span]
        return sorted(basis)

    def adjoint_action_tensor(self, lie_algebra, module):
        """Compute the action tensor C^k_{i,j} for the adjoint action of a lie_subalgebra on a given module.
        Here lie_algebra and module are a list of indices corresponding a linear subspace of the base Lie algebra.
        Output is a dictionary of indices (i,j) -> {k->C^k_{i,j}} for non-zero C^k_{i,j}"""

        action = defaultdict(dict)
        action_keys = set()
        for i, j in itertools.product(lie_algebra, module):  # for all pairs of indices belonging to either subspace

            # Compute Lie bracket.
            bracket = self.index_to_lie_algebra[i].bracket(self.index_to_lie_algebra[j]).monomial_coefficients()

            # Convert Lie bracket from monomial in roots to dict mapping index -> coefficient
            bracket_in_basis = {self.root_to_index[monomial]: coefficient for monomial, coefficient in bracket.items()}

            if len(bracket_in_basis) > 0:  # Store action only if it is non-trivial
                action[(i, j)] = bracket_in_basis

            # Store keys of non-zero action in a set to check if module is closed under adjoint action.
            for key in bracket_in_basis.keys():
                action_keys.add(key)

        if not action_keys.issubset(set(module)):  # Throw an error if the module is not closed under adjoint action
            raise ValueError("The module is not closed under the adjoint action")

        return action

    def coadjoint_action_tensor(self, lie_algebra, module):
        """Compute the action tensor C^k_{i,j} for the coadjoint action of a lie_subalgebra on a given module.
        Here lie_algebra and module are a list of indices corresponding a linear subspace of the base Lie algebra.
        The output is per definition restricted to the module.
        Output is a dictionary of indices (i,j) -> {k->C^k_{i,j}} for non-zero C^k_{i,j}"""

        action = defaultdict(dict)
        module_set = set(module)
        for i, k in itertools.product(lie_algebra, module):  # for all pairs of indices belonging to either subspace
            # The alpha_k component of coadjoint action on alpha_j is given by (minus)
            # the alpha_j component of the bracket [alpha_i ,alpha_k_dual], using the Killing form.
            # So first we compute bracket [alpha_i,alpha_k_dual]
            k_dual = self.dual_root_dict[k]
            bracket = self.index_to_lie_algebra[i].bracket(self.index_to_lie_algebra[k_dual]).monomial_coefficients()

            # We then iterate of the alpha_j. The alpha_j_dual component of `bracket` is the Killing form
            # <alpha_j, bracket>.
            for monomial, coefficient in bracket.items():
                dual_monomial = self.dual_root_dict[self.root_to_index[monomial]]
                if dual_monomial in module_set:  # We restrict to the module, otherwise action is not well-defined
                    action[(i, dual_monomial)][k] = -coefficient  # Minus sign comes from convention
        return action

    def build_component(self, subalgebra, action_type='ad', subset=None, acting_lie_algebra='n'):
        """Given a subalgebra (either 'g','n','u','b' or 'p'), a type of action (either 'ad' or 'coad'),
        a subset (a list of indices, corresponding to parabolic subalgebras) and an acting lie algebra
        (same type as subalgebra argument), returns a FastModuleComponent corresponding to the
        Lie algebra module of the input."""

        if subset is None:
            subset = []

        module_dic = {'g': self.g_basis,
                      'n': self.parabolic_n_basis(subset),
                      'u': self.parabolic_u_basis(subset),
                      'p': self.parabolic_p_basis(subset),
                      'h': self.basis['h'],
                      'b+': self.basis['b+'],
                      'b': self.parabolic_p_basis(None)}

        # The acting module shouldn't be parabolic. We could make this a feature.
        acting_module_dic = {'g': self.g_basis,
                      'n': self.parabolic_n_basis(None),
                      'u': self.parabolic_u_basis(None),
                      'p': self.parabolic_p_basis(None),
                      'h': self.basis['h'],
                      'b+': self.basis['b+'],
                      'b': self.parabolic_p_basis(None)}
        if subalgebra not in module_dic.keys():
            raise ValueError('Unknown subalgebra \'%s\'' % subalgebra)
        if acting_lie_algebra not in module_dic.keys():
            raise ValueError('Unknown subalgebra \'%s\'' % acting_lie_algebra)

        module = module_dic[subalgebra]
        lie_alg = acting_module_dic[acting_lie_algebra]

        if action_type == 'ad':
            action = self.adjoint_action_tensor(lie_alg, module)
        elif action_type == 'coad':
            action = self.coadjoint_action_tensor(lie_alg, module)
        else:
            raise ValueError('\'%s\' is not a valid type of action' % action_type)
        return FastModuleComponent(module, action, self)


class WeightSet:
    """Class to do simple computations with the weights of a weight module. Elements of the Weyl group
    are wherever possible encoded as a string encoding it as a product of simple reflections.
    The class needs an instance of BGGComplex to instantiate. """

    def __init__(self, BGG):
        self.reduced_words = BGG.reduced_words
        self.weyl_dic = BGG.reduced_word_dic
        self.simple_roots = BGG.simple_roots
        self.rho = BGG.rho
        self.rank = BGG.rank
        self.pos_roots = [self.tuple_to_weight(w) for w in BGG.neg_roots]

        # Matrix of all simple roots, for faster matrix solving
        self.simple_root_matrix = matrix([list(s.to_vector()) for s in self.simple_roots]).transpose()

        self.action_dic, self.rho_action_dic = self.get_action_dic()

    def weight_to_tuple(self, weight):
        """Convert element of weight lattice to a sum of simple roots by solving a matrix equation"""

        b = weight.to_vector()
        b = matrix(b).transpose()
        return tuple(self.simple_root_matrix.solve_right(b).transpose().list())

    def tuple_to_weight(self, t):
        """Inverse of operation above"""
        return sum(int(a) * b for a, b in zip(t, self.simple_roots))

    def get_action_dic(self):
        """Compute dictionary encoding Weyl group action, and dic encoding just the action on rho.
        action_dic constists of dictionary Weyl group -> matrix, and rho_action_dic Weyl group -> vector"""
        action_dic = dict()
        rho_action_dic = dict()
        for s, w in self.weyl_dic.items(): # s is a string, w is a matrix
            # Compute action of w on every simple root, decompose result in simple roots, encode result as matrix.
            action_mat = []
            for mu in self.simple_roots:
                action_mat.append(self.weight_to_tuple(w.action(mu)))
            action_dic[s] = np.array(action_mat, dtype=INT_PRECISION)

            # Encode the dot action of w on rho.
            rho_action_dic[s] = np.array(self.weight_to_tuple(w.action(self.rho) - self.rho), dtype=INT_PRECISION)
        return action_dic, rho_action_dic

    def dot_action(self, w, mu):
        """Compute the dot action of w on mu"""
        # The dot action w.mu = w(mu+rho)-rho = w*mu + (w*rho-rho).
        # The former term is given by action_dic, the latter by rho_action_dic
        return np.matmul(self.action_dic[w].T, np.array(mu, dtype=INT_PRECISION)) + self.rho_action_dic[w]

    def is_dot_regular(self, mu):
        """Check if mu has a non-trivial stabilizer under the dot action"""
        for s in self.reduced_words[1:]:
            if np.all(self.dot_action(s, mu) == mu):
                return False
        else:
            return True

    def compute_weights(self, weights):
        """For each weight, check whether it is dot-regular. If so, compute it's associated dominant and the
        length of the Weyl group element making it dominant.
         Returns a list of triples (dot-regular weight, associated dominant, Weyl group element length)"""

        regular_weights = []
        for mu in weights:
            if self.is_dot_regular(mu):
                mu_prime, w = self.make_dominant(mu)
                regular_weights.append((mu, tuple(mu_prime), len(w)))
        return regular_weights

    def is_dominant(self, mu):
        """Use sagemath built-in function to check if something is dominant"""

        return self.tuple_to_weight(mu).is_dominant()

    def make_dominant(self, mu):
        """For a dot regular weight mu, iterate over all w in W and check if w.mu is dominant. Once found,
         return w.mu and w. It is known that such a w always exists and is unique."""

        for w in self.reduced_words:
            new_mu = self.dot_action(w, mu)
            if self.is_dominant(new_mu):
                return new_mu, w
        else:
            raise ValueError('Could not make weight %s dominant, probably it is not dot-regular.')

    def get_vertex_weights(self, mu):
        """For a given dot-regular mu, return its dot orbit. The orbit is returned as a list of tuples encoding
        weights. This is because an immutable type is required for applications."""

        vertex_weights = dict()
        for w in self.reduced_words:
            vertex_weights[w] = tuple(self.dot_action(w, mu))
        return vertex_weights

    def highest_weight_rep_dim(self, mu):
        """Uses the Weyl dimension formula to obtain the dimension of the highest weight
        representation of a integral dominant weight `mu`."""
        mu_weight = self.tuple_to_weight(mu)
        numerator = 1
        denominator = 1
        for alpha in self.pos_roots:
            numerator *= (mu_weight+self.rho).dot_product(alpha)
            denominator *= (self.rho.dot_product(alpha))
        return numerator//denominator

class BGGCohomology:
    """Class for computing the BGG cohomology of a module. This is a seperate class because it needs to comunicate
    between the module and the BGG complex.
    Input is a BGGComplex instance, and a FastLieAlgebraCompositeModule instance."""

    def __init__(self, BGG, weight_module, coker=None):
        self.BGG = BGG
        self.BGG.compute_signs()  # Make sure BGG signs are computed.
        self.weight_module = weight_module
        if coker is not None:
            self.has_coker = True
            self.coker = coker
        else:
            self.has_coker = False

        self.weights = weight_module.weight_components.keys()
        self.weight_set = WeightSet(BGG)

        self.num_components = len(self.weight_module.components)

        self.regular_weights = self.weight_set.compute_weights(self.weights)  # Find dot-regular weights

    def compute_differential(self, mu, i):
        """Given a dominant weight mu, compute the BGG differential at degree i. The resulting matrix
        does not contain any zero rows, but has the same image as the true differential, and has therefore the
        same rank as the true differential.
        This method outputs a tuple consisting of the matrix of the differential and the dimension of the source
        space (of the true differential)."""

        if not self.weight_set.is_dominant(mu):  # Assert the weight is dominant to prevent further errors.
            raise ValueError('Input weight %s is not dominant' % str(mu))

        vertex_weights = self.weight_set.get_vertex_weights(mu)  # Compute dot orbit of mu

        # Weights need to be in the right format for BGG.compute maps. This will hopefully be fixed in the future.
        maps = self.BGG.compute_maps(self.BGG.weight_to_alpha_sum(self.BGG._tuple_to_weight(mu)),check=True)

        column = self.BGG.column[i]  # Elements of Weyl group in ith column

        # For each w in the ith column, find all arrows w->w' for some w' (of longer length)
        delta_i_arrows = [(w, [arrow for arrow in self.BGG.arrows if arrow[0] == w]) for w in column]

        # Count the total dimension of all the weight components associated to the ith column.
        source_dim = 0
        for w in column:
            initial_vertex = vertex_weights[w]
            if initial_vertex in self.weights:
                source_dim += self.weight_module.dimensions[initial_vertex]

        # For each arrow a: w->w', we compute the action of the PBW element associated to w->w'.
        # We then multiply the result by the sign associated to w->w' in the BGG complex.
        # Finally we concatenate all these in a list.
        #output = []
        diff_keys = []
        diff_coeffs = []
        for w, arrows in delta_i_arrows:
            initial_vertex = vertex_weights[w]  # Get weight w.mu
            if initial_vertex in self.weights:  # Weight component may be empty
                for a in arrows:
                    sign = self.BGG.signs[a]
                    self.weight_module.set_pbw_action_matrix(maps[a])
                    for weight_comp in self.weight_module.weight_components[initial_vertex]:
                        key_pairs, coefficients = self.weight_module.compute_pbw_action(weight_comp)
                        if len(key_pairs) > 0:
                            diff_keys.append(key_pairs)
                            diff_coeffs.append(sign*coefficients)
                            #output.append((key_pairs, sign * coefficients))
        if len(diff_keys)>0:
            diff_keys = np.concatenate(diff_keys)
            diff_coeffs = np.concatenate(diff_coeffs)


            # Make a list of rows in the matrix. Each row is encoded as two numpy vectors.
            # The first vector encodes the hashes of the indices of the image
            # The second vector encodes the coefficients.
            row_list = []
            # for key_pairs, coefficients in output:
            gb = npi.group_by(diff_keys[:, 0])
            row_list = zip(gb.split_array_as_list(diff_keys[:, 1]), gb.split_array_as_list(diff_coeffs))
            # row_list += rows

        else:
            row_list=[]


        # Make a set of all the occurring hashes, enumerate them, and make a dictionary sending hashes to new indices
        all_hashes = []
        for columns, _ in row_list:
            all_hashes += list(columns)
        hash_dic = {h: i for i, h in enumerate(set(all_hashes))}

        # For each row in row_list, replace hashes by new indices, and then sort new indices and reorder coefficients.
        def convert_and_sort(input):
            input_columns, input_data = input
            converted_columns = np.array([hash_dic[c] for c in input_columns], dtype=INT_PRECISION)
            indices = np.argsort(converted_columns)
            return converted_columns[indices], input_data[indices]
        row_list = map(convert_and_sort, row_list)



        # Create a dictionary of all non-zero entries of the matrix of the differential
        # Keys are pairs (row, column), values are the coefficients.
        sparse_dic = dict()
        for row, (columns, data) in enumerate(row_list):
            for column, entry in zip(columns, data):
                sparse_dic[(row, column)] = entry


        # Use the dictionary above to build a sparse matrix
        differential_matrix = matrix(ZZ, len(row_list), len(hash_dic), sparse_dic, sparse=True)

        # Return the sparse matrix as well as the dimension of the source space (because zero rows are omitted).
        return differential_matrix, source_dim

    def cohomology_component(self, mu, i):
        """Compute cohomology BGG_i(mu)"""

        #d_i, chain_dim = self.compute_differential(mu, i)
        #d_i_minus_1, _ = self.compute_differential(mu, i - 1)

        d_i, chain_dim = cohomology.compute_diff(self,mu,i)
        d_i_minus_1, _ = cohomology.compute_diff(self,mu,i-1)

        rank_1 = d_i.rank()
        rank_2 = d_i_minus_1.rank()
        return chain_dim - rank_1 - rank_2

    def cohomology(self, i):
        """Compute full block of cohomology by computing BGG_i(mu) for all dot-regular mu appearing in
        the weight module of length i.
        For a given weight mu, if there are no other weights mu' of length i +/- 1 with the
        same associated dominant, then the weight component mu is isolated and the associated
        cohomology is the entire weight module.
        The cohomology is returned as a list of highest weight vectors appearing in the
        cohomology, together with their multiplicities."""

        # Find all dot-regular weights of lenght i, together with their associated dominants.
        length_i_weights = [triplet for triplet in self.regular_weights if triplet[2] == i]

        # Split the set of dominant weights in those which are isolated, and therefore don't require
        # us to run the BGG machinery, and those which are not isolated and require the BGG machinery.
        dominant_non_trivial = set()
        dominant_trivial = []
        for w, w_dom, _ in length_i_weights:
            for _, w_prime_dom, l in self.regular_weights:
                if w_prime_dom == w_dom and (l == i + 1 or l == i - 1):
                    dominant_non_trivial.add(w_dom)
                    break
            else:
                dominant_trivial.append((w, w_dom))

        cohomology = defaultdict(int)


        # For isolated weights, multiplicity is just the module dimension
        for w, w_dom in dominant_trivial:
            cohom_dim = self.weight_module.dimensions[w]
            if cohom_dim > 0:
                cohomology[w_dom] += cohom_dim

        # For non-isolated weights, multiplicity is computed through the BGG differential
        for w in dominant_non_trivial:
            cohom_dim = self.cohomology_component(w, i)
            if cohom_dim > 0:
                cohomology[w] += cohom_dim

        # Return cohomology as sorted list of highest weight vectors and multiplicities.
        # The sorting order is the same as how we normally sort weights
        return sorted(cohomology.items(), key=lambda t: (sum(t[0]), t[0][::-1]))

    def betti_number(self, cohomology):
        """Given a list of highest weight vectors + multiplicities, compute
        the associated betti numbers by taking a weighted sum of the dimensions of
        the associated highest weight representations.
        """
        betti_num = 0
        for mu, mult in cohomology:
            dim = self.weight_set.highest_weight_rep_dim(mu)
            betti_num+=dim*mult
        return betti_num

    def cohomology_LaTeX(self, i=None, complex_string='', only_non_zero=True, print_betti=False, print_modules= True):
        """In a notebook we can use pretty display of cohomology output.
        Only displays cohomology, does not return anything.
        We have the following options:
        i = None, degree to compute cohomology. If none, compute in all degrees
        complex_string ='', an optional string to cohom as H^i(complex_string) = ...
        only_non_zero = True, a bool indicating whether to print non-zero cohomologies.
        print_betti = False, print the Betti numbers
        print_modules = True, print the decomposition of cohomology into highest weight reps"""

        # If there is a complex_string, insert it between brackets, otherwise no brackets.
        if len(complex_string) > 0:
            display_string = r'(%s)=' % complex_string
        else:
            display_string = r'='

        # compute cohomology. If cohomology is trivial and only_non_zero is true, return nothing.
        if i is None:
            cohoms = [(j, self.cohomology(j)) for j in range(self.BGG.max_word_length+1)]
            max_len = max([len(cohom) for _, cohom in cohoms])
            if max_len==0:
                display(Math(r'\mathrm H^\bullet' + display_string+'0'))
                return None
        else:
            cohoms = [(i, self.cohomology(i))]

        for i, cohom in cohoms:
            if (not only_non_zero) or (len(cohom)>0):
                # Print the decomposition into highest weight modules.
                if print_modules:
                    # Get LaTeX string of the highest weights + multiplicities
                    latex = self.cohom_to_latex(cohom)

                    # Display the cohomology in the notebook using LaTeX rendering
                    display(Math(r'\mathrm H^{%d}' % i + display_string + latex))

                # Print just dimension of cohomology
                if print_betti:
                    betti_num = self.betti_number(cohom)
                    display(Math(r'\mathrm b^{%d}' % i + display_string + str(betti_num)))


    def tuple_to_latex(self, (mu, mult)):
        """LaTeX string representing a tuple of highest weight vector and it's multiplicity"""

        # Each entry mu_i in the tuple mu represents a simple root. We display it as mu_i alpha_i
        alphas = []
        for i, m in enumerate(mu):
            if m == 1:
                alphas.append(r'\alpha_{%d}' % (i + 1))
            elif m != 0:
                alphas.append(r' %d\alpha_{%d}' % (m, i + 1))

        # If all entries are zero, we just display the string '0' to represent zero weight,
        # otherwise we join the mu_i alpha_i together with a + operator in between
        if mult>1:
            if len(alphas) == 0:
                return r'\mathbb{C}^{%d}' % mult
            else:
                alphas_string = r'+'.join(alphas)
                return r'L\left(%s\right)^{%d}' % (alphas_string,mult)
        else:
            if len(alphas) == 0:
                return r'\mathbb{C}'
            else:
                alphas_string = r'+'.join(alphas)
                return r'L\left(%s\right)' % alphas_string

    def cohom_to_latex(self, cohom):
        """String together the tuple_to_latex function multiple times to turn a list of mu, multiplicity
        into one big LaTeX string."""

        # If there is no cohomology just print the string '0'
        if len(cohom) > 0:
            return r'\oplus '.join(map(self.tuple_to_latex, cohom))
        else:
            return r'0'

