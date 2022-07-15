# cython: language_level=2
# cython: profile=True
"""
Module to compute the differentials of the BGG complex. Implemented in Cython for extra speed, since it
is relatively critical for performance.
"""

import numpy as np

from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.rings.rational import Rational
from sage.rings.integer cimport Integer
from sage.matrix.matrix_space import MatrixSpace
from sage.matrix.matrix_integer_dense cimport Matrix_integer_dense
from sage.matrix.matrix_integer_sparse cimport Matrix_integer_sparse
from sage.matrix.special import block_matrix
from time import perf_counter

cpdef compute_action(acting_element, action_source, module, comp_num):
    """Computes action of a single lie algebra element on a list of elements of the module. 
    Outputs a new array where indices and coefficients are replaced as per the action. 
    The output is unsorted, and may contain duplicate entries. """

    # Initialize array for output
    action_image = np.zeros_like(action_source)

    # Get component types. Each type has a different action of the Lie algebra
    type_list = module.type_lists[comp_num]
    cdef image_row = 0 # counter for which row to edit in output
    cdef max_rows = len(action_image)

    cdef int row,j
    for col,mod_type in enumerate(type_list): # compute the action one column at a time
        action_tensor = module.action_tensor_dic[mod_type] # retrieve the structure coefficient tensor
        for row in range(len(action_source)):
            j = action_source[row,col]
            s,k,Cijk = action_tensor[acting_element,j]
            while s!=0: # if s=0, then there are no non-zero structure coeffs
                new_row = action_source[row].copy() # copy row, and change index to k
                new_row[col] = k
                new_row[-1]*=Cijk # multiply coefficient of element by structure coefficient C_ijk
                action_image[image_row] = new_row
                if s==-1: # end of the chain, break out of loop
                    s=0
                else: # still more non-zero C_ijk's to deal with
                    s,k,Cijk = action_tensor[s,j]
                image_row+=1
                if image_row>=max_rows: # double size of image matrix if we run out of space
                    action_image = np.concatenate([action_image,np.zeros_like(action_image)])
                    max_rows = len(action_image)
    return action_image[:image_row] # Only return non-zero rows

cdef check_equal(long [:] row1,long [:] row2,int num_cols):
    """fast check to see if two arrays of given length are equal"""
    cdef int i
    for i in range(num_cols):
        if row1[i]!=row2[i]:
            return False
    else:
        return True

cdef col_nonzero(long [:] col, int num_rows):
    """returns non-zero indices of a column. np.nonzero doesn't seem to work well with memoryviews."""

    indices = np.zeros(num_rows,np.int32)
    cdef int i
    cdef int j = 0
    for i in range(num_rows):
        if col[i]!=0:
            indices[j]=i
            j+=1
    return indices[:j]

cdef merge_sorted_image(long [:,:] action_image):
    """Given sorted array, merges adjacent rows which are equal except for last column, 
    and adds the values of the last column"""
    merged_image = np.zeros_like(action_image)

    cdef long[:] old_row
    cdef long[:] row
    old_row = np.zeros_like(action_image[0])-1

    cdef int row_number = -1
    cdef int num_cols = action_image.shape[1]-1
    cdef int num_rows = action_image.shape[0]

    cdef int i
    for i in range(num_rows):
        row = action_image[i]
        if row[-1]!=0:
            if check_equal(row,old_row,num_cols): # if previous and current row are equal, add last column to result
                merged_image[row_number,-1] += row[-1]
            else:
                row_number+=1
                merged_image[row_number]=row
                old_row = row
    row_number+=1
    non_zero_inds = col_nonzero(merged_image[:row_number,-1],row_number) # Only return rows where column is non-zero
    return merged_image[non_zero_inds,:]

def sort_merge(action_image):
    """Sorts array, ignoring last column and merges rows which are equal, summing in the last column"""
    action_image = action_image[np.lexsort(np.transpose(action_image[:,:-1]))]
    return merge_sorted_image(action_image)

cdef permutation_sign(long [:] row,int num_cols):
    """Computes the sign of a permutation using bubble sort (efficient for extremely short inputs)"""
    cdef int sign = 1
    cdef int i,j
    for i in range(num_cols):
        for j in range(i+1,num_cols):
            if row[i]==row[j]: # if there are duplicate entries then sign is 0 per definition
                return 0
            elif row[i]>row[j]: # For each swap we have to do, the sign changes by -1.
                sign*=-1
    return sign

cdef sort_cols(module, action_image,comp_num):
    """Sort all the entries of each tensor component. If tensor component is a wedge power, then
    mutliply coefficient by sign of permutation sorting the row."""
    cdef int col_min = 0
    cdef int num_rows = len(action_image)
    cdef int i
    cdef long[:] row

    for _,cols,mod_type in module.components[comp_num]:
        if cols>1: # List with one item is always sorted
            if mod_type == 'wedge':
                for i in range(num_rows):
                    row = action_image[i,col_min:col_min+cols]
                    action_image[i,-1]*=permutation_sign(row,cols) # Change coefficient by sign of permutation
            action_image[:,col_min:col_min+cols] = np.sort(action_image[:,col_min:col_min+cols]) # sort the rows
        col_min+=cols

cpdef action_on_basis(pbw_elt,wmbase,module,factory,comp_num):
    """Computes the action of an element of U(n) in PBW order on a basis of the weight component.
    Input is the PBW element, the basis of the weight component,
    the factory that created the module, and the number of the direct sum component"""

    num_cols = wmbase.shape[1]
    action_list = []
    action_source = np.zeros((wmbase.shape[0], num_cols+2),np.int64)
    action_source[:,:num_cols] = wmbase
    action_source[:,num_cols] = np.arange(len(wmbase))
    action_source[:,-1] = 1

    # Compute action for each monomial seperately, and then sum results
    for monomial,coefficient in pbw_elt.monomial_coefficients().items():
        action_image = action_source.copy()
        action_image[:,-1]*=coefficient # mutliply results by coefficient of monomial
        for term in monomial.to_word_list()[::-1]: # Right action, so we take terms of the monomial in inverse order
            index = factory.root_to_index[term] # get the index of the term
            action_image = compute_action(index, action_image, module, comp_num) # compute the action
        action_list.append(action_image)
    action_image = np.concatenate(action_list) # concatenate and merge is equivalent to summing the results.
    if len(action_image)==0: # merging gives errors for empty matrices
        return action_image
    else:
        sort_cols(module,action_image,comp_num) # Sort and merge
        action_image = sort_merge(action_image)

        return action_image

def check_weights(module,action_image):
    weights = set()
    for row in action_image[:,:-2]:
        mu = sum(module.weight_dic[s] for s in row)
        weights.add(tuple(mu))
    if len(weights)>1:
        raise ValueError("Found too many weights :(")

def compute_diff(cohom, mu, i, return_sparse=False):
    """"
    Computes the BGG differential associated to a BGGCohomology object, weight mu and degree i.
    The matrix produced is of the correct rank, but omits some rows consisting entirely of zeros.
    In order to correctly compute kernel, the dimension of the source space is therefore also returned.
    """
    # aliases
    BGG = cohom.BGG
    module = cohom.weight_module
    factory =  module.factory

    # weights associated to each vertex of the Bruhat graph
    vertex_weights = cohom.weight_set.get_vertex_weights(mu)

    # maps of the BGG complex
    maps = BGG._maps[mu]

    # for each vertex, get the arrows in the Bruhat graph going out of it.
    column = BGG.column[i]
    delta_i_arrows = [(w, [arrow for arrow in BGG.arrows if arrow[0] == w]) for w in column]

    # Look up vertex weights for the target column
    target_column = BGG.column[i+1]
    target_col_dic = {w:vertex_weights[w] for w in target_column}

    # To give the weights in the target column a unique index, we compute
    # an offset for each weight component in the target column
    offset = 0
    for w,mu in target_col_dic.items():
        target_col_dic[w] = offset
        if cohom.has_coker and (mu in cohom.coker):
            offset+=cohom.coker[mu].nrows() # Dimension of quotient is number of rows
        else:
            if mu in module.dimensions:
                offset+=module.dimensions[mu]

    # Compute dimension of source space by adding dimensions of weight components in the column
    source_dim = 0
    for w in column:
        initial_vertex = vertex_weights[w]
        if initial_vertex in cohom.weights:
            if cohom.has_coker and (initial_vertex in cohom.coker):
                source_dim += cohom.coker[initial_vertex].nrows() # dimension of quotient is number of rows
            else:
                source_dim += cohom.weight_module.dimensions[initial_vertex]


    offset = 0
    diff_dict = {}
    for w, arrows in delta_i_arrows:
        initial_vertex = vertex_weights[w]  # weight of vertex
        if initial_vertex in cohom.weights:  # Ensure weight component isn't empty
            for a in arrows: # Compute image for each arrow
                final_vertex = vertex_weights[a[1]]

                sign = BGG.signs[a] # Multiply everything by the sign of the map in BGG complex

                comp_offset_s = 0

                for comp_num,weight_comp in module.weight_components[initial_vertex]:
                    # compute the action of the PBW element
                    # map is multiplied by sign. Need to convert sign to Rational to avoid errors in newer sage version
                    basis_action = action_on_basis(maps[a]*Rational(sign),weight_comp,module,factory,comp_num)

                    basis_action[:,-2] += comp_offset_s # update source
                    initial_dimension = module.dimensions_components[comp_num][initial_vertex]
                    comp_offset_s += initial_dimension

                    try:
                        initial_dim = module.dimensions[initial_vertex]
                        final_dim = module.dimensions[final_vertex]
                    except KeyError:  # One of the modules is empty
                        continue
                    # If there is a cokernel, we have reduce the image of the action
                    # to the basis of the quotient module
                    if cohom.has_coker:
                        bas2 = coker_reduce(
                            cohom.weight_module,cohom.coker, basis_action,
                            initial_vertex, final_vertex,
                            initial_dim, final_dim,
                            component=comp_num
                        )
                        bas = bas2



                    # The cokernel reduction automatically converts to sparse format
                    # In the non-cokernel case we still have to do this manually
                    if len(basis_action)>0:
                        if not cohom.has_coker:
                            target_basis_dic = module.weight_comp_index_numbers[final_vertex]
                            basis_action = multiindex_to_index(basis_action,target_basis_dic,comp_num)
                            bas = dok_to_sage_sparse(basis_action, initial_dim, final_dim)
                    key = (initial_vertex,final_vertex,comp_num)
                    if key in diff_dict:
                        raise ValueError("Found duplicate key in diff_dict")
                    diff_dict[key] = bas


    d_sparse = assemble_block_diff(diff_dict)

    return d_sparse, source_dim

cdef multiindex_to_index(long [:,:] action_image, target_basis_dic, long component):
    cdef size_t num_rows = action_image.shape[0]
    cdef long[:,:] lookup_map = np.concatenate(
        [action_image[:,:-2], np.ones((num_rows,1), dtype=np.int_) * component], axis=1
    )
    cdef long[:] target_inds = np.zeros(num_rows, dtype=np.int_)
    cdef size_t j
    for j in range(num_rows):
        target_inds[j] = target_basis_dic[tuple(lookup_map[j])]
    cdef long[:,:] result = np.stack([target_inds,action_image[:,-2], action_image[:,-1]],axis=1)

    return result

# def dok_to_sage_sparse(dok_matrix):
#     return matrix(ZZ, {(a,b): c for a, b, c in np.array(dok_matrix)}, sparse=True)

def dok_to_sage_sparse(dok_matrix, initial_dim, final_dim):
    result = matrix(ZZ, final_dim, initial_dim, sparse=True)
    for a, b, c in np.array(dok_matrix):
        result[a,b] += c
    return result

def dok_to_sage_dense(dok_matrix):
    return matrix(ZZ, {(a,b): c for a, b, c in np.array(dok_matrix)}, sparse=False)

def assemble_block_diff(diff_dict):
    initial_vertices = set()
    final_vertices = set()
    pairs_sums = dict()
    for (init,final,comp_num),val in diff_dict.items():
        initial_vertices.add(init)
        final_vertices.add(final)
        try:
            pairs_sums[(init,final)] += val
        except KeyError:
            pairs_sums[(init,final)] = val
        
    block_rows = []
    for final in final_vertices:
        block_row= []
        for init in initial_vertices:
            if (init,final) in pairs_sums:
                block_row.append(pairs_sums[(init,final)])
            else:
                block_row.append(Integer(0))
        block_rows.append(block_row)
    sparse_new_block= block_matrix(block_rows, sparse=True)
    return sparse_new_block

def coker_reduce(target_module, coker, long[:,:] action_image, mu0, mu1,initial_dim,final_dim, component=0):
    """Projects source and target of an action in the coker quotient coker(f), f:M->N.
    Returns action in the basis of the cokernel.
    `source_module` is the module M
    `target_module` is the module N
    `coker` is a dictionary encoding a basis of the coker in each weight component of N
    `action_image` is what action_on_basis returns.
    `component` is the index of the direct sum component
    We assume we acted with a map `mu0`->`mu1` in action_on_basis.
    """
    # the input is always of shape [i1,i2,..,ik,j,c] where i denotes the indices
    # of the target, j the source index, and c the coefficient.
    # then `num_cols` denotes this number k.
    cdef size_t num_cols = action_image.shape[1]-2

    # If mu1 is not in the module, then it has to be zero
    if mu1 not in target_module.weight_components:
        return []

    # Convert the sets of indices [i1,...,ik] into a single index i for the whole weight component
    # We do this by looking the index i up in a dictionary.
    target_basis_dic = target_module.weight_comp_index_numbers[mu1]
    cdef long[:,:] action_image_target

    action_image_target = multiindex_to_index(
        action_image, target_basis_dic, component
    )
    cdef Matrix_integer_sparse res = dok_to_sage_sparse(action_image_target, initial_dim, final_dim)
    cdef size_t max_s = res.ncols()
    cdef size_t max_t = res.nrows()

    # If mu0 is in the cokernel dictionary, express the action in the basis of the quotient
    # If not, then the basis of the quotient is equal to the basis of the module, so there's nothing to do
    if mu0 in coker:
        # res has shape max_t, max_s
        res = res*(coker[mu0][:,:max_s].T)
    

    # If mu1 is in the cokernel dictionary, then reduce the image to the quotient
    # We do this by multiplying by the matrix encoding the basis of the cokernel
    # If it's not in the dictionary, no reduction is necessary.
    if mu1 in coker:
        res = coker[mu1][:,:max_t]*res
    return res
