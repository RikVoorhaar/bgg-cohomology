#cython: language_level=2
"""
Module to compute the differentials of the BGG complex. Implemented in Cython for extra speed, since it
is relatively critical for performance.
"""

import numpy as np

from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.rings.rational import Rational

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

def compute_diff(cohom, mu, i):
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
    maps = BGG.compute_maps(BGG.weight_to_alpha_sum(BGG._tuple_to_weight(mu)),check=True)

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
    total_diff=[]
    for w, arrows in delta_i_arrows:
        initial_vertex = vertex_weights[w]  # weight of vertex
        if initial_vertex in cohom.weights:  # Ensure weight component isn't empty
            action_images = []
            for a in arrows: # Compute image for each arrow
                final_vertex = vertex_weights[a[1]]

                sign = BGG.signs[a] # Multiply everything by the sign of the map in BGG complex

                comp_offset_s = 0

                for comp_num,weight_comp in module.weight_components[initial_vertex]:
                    # compute the action of the PBW element
                    # map is multiplied by sign. Need to convert sign to Rational to avoid errors in newer sage version
                    basis_action = action_on_basis(maps[a]*Rational(sign),weight_comp,module,factory,comp_num)

                    basis_action[:,-2] += comp_offset_s # update source
                    comp_offset_s += module.dimensions_components[comp_num][initial_vertex]

                    # If there is a cokernel, we have reduce the image of the action
                    # to the basis of the quotient module
                    if cohom.has_coker:
                        basis_action = coker_reduce(cohom.weight_module,cohom.coker, basis_action,
                                                    initial_vertex, final_vertex,
                                                    component=comp_num)
                        if len(basis_action)>0:
                            basis_action[:,0]+=target_col_dic[a[1]] # offset for weight module
                            action_images.append(basis_action)

                    # The cokernel reduction automatically inserts appropriate offsets for indices
                    # In the non-cokernel case we still have to do this manually
                    if len(basis_action)>0:
                        if not cohom.has_coker:
                            new_basis_action = np.zeros(shape=(basis_action.shape[0],3),dtype=basis_action.dtype)



                            # Convert the sets of indices [i1,...,ik] into a single index for the whole weight component
                            # We do this by looking the index i up in a dictionary.
                            target_basis_dic = module.weight_comp_index_numbers[final_vertex]
                            num_cols = basis_action.shape[1]-2
                            for i,row in enumerate(basis_action):
                                j = target_basis_dic[tuple(list(row[:num_cols])+[comp_num])]
                                new_basis_action[i][0]=j
                                new_basis_action[i][1:] = row[num_cols:]
                            new_basis_action[:,0]+=target_col_dic[a[1]]

                            action_images.append(new_basis_action)

            if len(action_images)>0:
                # Concatenate images for each arrow to get total image
                sub_diff = np.concatenate(action_images)

                # Each basis element of weight component gets index
                # Because we have multiple weight components, we need to add a number to this index
                # So that index remains unique across multiple components

                sub_diff[:,-2]+=offset

                offset+=module.dimensions[initial_vertex]
                total_diff.append(sub_diff)


    if len(total_diff)>0: # Sometimes action is trivial, would otherwise raise errors
        total_diff = np.concatenate(total_diff)
        total_diff = sort_merge(total_diff) # for cokernels we can get duplicate entries. We need to merge them.
        total_diff = total_diff[np.lexsort(np.transpose(total_diff[:,:-2]))]  # Sort by source indices

    if len(total_diff) ==0: # Trivial differential
        return matrix(ZZ,0,0),source_dim

     # encode as sparse matrix. each entry is triple of two indices and the value at the two indices
    diff_entries = np.zeros((len(total_diff),3),np.int64)
    j = -1

    # If two entries represent the same element in source column, put them in same row j.
    # This loop merges this an populates a sparse matrix with correct row numbers.

    prev_row = np.zeros_like(total_diff[0,:-2])-1  # every row is different from this one
    for i in range(len(total_diff)):
        row_num = i
        row = total_diff[i,:-2]
        if np.any(np.not_equal(row,prev_row)): # if row is different, it will have different index
            j+=1
            prev_row = row
        diff_entries[row_num,0] = j # populate sparse matrix
        diff_entries[row_num,1:] = total_diff[i,-2:]
    j+=1

    # turn sparse differential matrix into dense one.
    d_dense = matrix(ZZ,j,max(diff_entries[:,1])+1)
    for i in range(len(diff_entries)):
        d_dense[diff_entries[i,0],diff_entries[i,1]] = diff_entries[i,2]

    return d_dense, source_dim

def coker_reduce(target_module, coker, action_image, mu0, mu1, component=0):
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
    num_cols = action_image.shape[1]-2

    # If mu1 is not in the module, then it has to be zero
    if mu1 not in target_module.weight_components:
        return []

    # Convert the sets of indices [i1,...,ik] into a single index i for the whole weight component
    # We do this by looking the index i up in a dictionary.
    target_basis_dic = target_module.weight_comp_index_numbers[mu1]
    action_image_target = np.zeros((action_image.shape[0],3),dtype=action_image.dtype)
    for i,row in enumerate(action_image):
        j = target_basis_dic[tuple(list(row[:num_cols])+[component])]
        action_image_target[i][0]=j
        action_image_target[i][1:] = row[num_cols:]

    # If mu0 is in the cokernel dictionary, express the action in the basis of the quotient
    # If not, then the basis of the quotient is equal to the basis of the module, so there's nothing to do
    if mu0 in coker:
        new_images = np.zeros((action_image.shape[0]*coker[mu0].ncols(),3),dtype=action_image.dtype)
        current_row = 0

        for action_row in action_image_target:
            target, source,coeff = action_row
            for i,c in enumerate(coker[mu0][:,source]):
                if c!=0:
                    new_images[current_row] = [target, i, coeff*c]
                    current_row+=1
        new_action_image = new_images[:current_row]
    else:
        new_action_image = action_image_target

    # If mu1 is in the cokernel dictionary, then reduce the image to the quotient
    # We do this by multiplying by the matrix encoding the basis of the cokernel
    # If it's not in the dictionary, no reduction is necessary.
    if mu1 in coker:
        new_image_coker = np.zeros((new_action_image.shape[0]*coker[mu1].ncols(),3),dtype=action_image.dtype)
        current_row = 0

        for action_row in new_action_image:
            j = action_row[0]
            for i,c in enumerate(coker[mu1][:,j]):
                if c!=0:
                    new_image_coker[current_row]=action_row
                    new_image_coker[current_row][0]=i
                    new_image_coker[current_row][2]*=c
                    current_row+=1
    else:
        new_image_coker = new_action_image
        current_row = len(new_image_coker)

    # At the end of the day, sort the result and sum coefficients of identical (source, target) tuples.
    # If the final matrix is empty, instead we just return an empty array to avoid errors.
    if current_row>0:
        return sort_merge(new_image_coker[:current_row])
    else:
        return np.array([])