import numpy as np
cimport numpy as np

from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ

cdef compute_action(acting_element, action_source, module):
    action_image = np.zeros_like(action_source)
    type_list = module.type_lists[0]
    cdef image_row = 0
    cdef max_rows = len(action_image)

    cdef int row,j
    for col,mod_type in enumerate(type_list):
        action_tensor = module.action_tensor_dic[mod_type]
        for row in range(len(action_source)):
            j = action_source[row,col]
            s,k,Cijk = action_tensor[acting_element,j]
            while s!=0:
                new_row = action_source[row].copy()
                new_row[col] = k
                new_row[-1]*=Cijk
                action_image[image_row] = new_row
                if s==-1:
                    s=0
                else:
                    s,k,Cijk = action_tensor[s,j]
                image_row+=1
                if image_row>=max_rows: # double size of image matrix if we run out of space
                    action_image = np.concatenate([action_image,np.zeros_like(action_image)])
                    max_rows = len(action_image)
    return action_image[:image_row]

cdef check_equal(long [:] row1,long [:] row2,int num_cols):
    cdef int i
    for i in range(num_cols):
        if row1[i]!=row2[i]:
            return False
    else:
        return True

cdef col_nonzero(long [:] col, int num_rows):
    """return non-zero indices of a column. np.nonzero doesn't seem to work well with memoryviews."""

    indices = np.zeros(num_rows,np.int32)
    cdef int i
    cdef int j = 0
    for i in range(num_rows):
        if col[i]!=0:
            indices[j]=i
            j+=1
    return indices[:j]

cdef merge_sorted_image(long [:,:] action_image):
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
            if check_equal(row,old_row,num_cols):
                merged_image[row_number,-1] += row[-1]
            else:
                row_number+=1
                merged_image[row_number]=row
                old_row = row
    row_number+=1
    non_zero_inds = col_nonzero(merged_image[:row_number,-1],row_number)
    return merged_image[non_zero_inds,:]

def sort_merge(action_image):
    action_image = action_image[np.lexsort(np.transpose(action_image[:,:-1]))]
    return merge_sorted_image(action_image)

cdef permutation_sign(long [:] row,int num_cols):
    cdef int sign = 1
    cdef int i,j
    for i in range(num_cols):
        for j in range(i+1,num_cols):
            if row[i]==row[j]:
                return 0
            elif row[i]>row[j]:
                sign*=-1
    return sign

cdef sort_cols(module, action_image,comp_num):
    cdef int col_min = 0
    cdef int num_rows = len(action_image)
    cdef int i
    cdef long[:] row

    for _,cols,mod_type in module.components[comp_num]:
        if cols>1: # List with one item is always sorted
            if mod_type == 'wedge':
                for i in range(num_rows):
                    row = action_image[i,col_min:col_min+cols]
                    action_image[i,-1]*=permutation_sign(row,cols)
            action_image[:,col_min:col_min+cols] = np.sort(action_image[:,col_min:col_min+cols])
        col_min+=cols

cpdef action_on_basis(pbw_elt,wmbase,module,factory,comp_num):
    num_cols = wmbase.shape[1]
    action_list = []
    action_source = np.zeros((wmbase.shape[0], num_cols+2),np.int64)
    action_source[:,:num_cols] = wmbase
    action_source[:,num_cols] = np.arange(len(wmbase))
    action_source[:,-1] = 1
    for monomial,coefficient in pbw_elt.monomial_coefficients().items():
        action_image = action_source.copy()
        action_image[:,-1]*=coefficient
        for term in monomial.to_word_list()[::-1]:
            index = factory.root_to_index[term]
            action_image = compute_action(index, action_image,module)
        action_list.append(action_image)
    action_image = np.concatenate(action_list)
    sort_cols(module,action_image,comp_num)
    return sort_merge(action_image)



def compute_diff(cohom,mu,i):
    BGG = cohom.BGG
    module = cohom.weight_module
    factory =  module.component_dic.values()[0].factory

    vertex_weights = cohom.weight_set.get_vertex_weights(mu)
    maps = BGG.compute_maps(BGG.weight_to_alpha_sum(BGG._tuple_to_weight(mu)),check=True)
    column = BGG.column[i]
    delta_i_arrows = [(w, [arrow for arrow in BGG.arrows if arrow[0] == w]) for w in column]

    source_dim = 0
    for w in column:
        initial_vertex = vertex_weights[w]
        if initial_vertex in cohom.weights:
            source_dim += cohom.weight_module.dimensions[initial_vertex]

    offset = 0
    total_diff_list = []
    for comp_num in range(len(module.components)):
        total_diff=[]
        for w, arrows in delta_i_arrows:
            initial_vertex = vertex_weights[w]
            if initial_vertex in cohom.weights: # Ensure weight component isn't empty
                action_images = []
                max_ind = 0
                for a in arrows:
                    sign = BGG.signs[a]
                    weight_comp = module.weight_components[initial_vertex][comp_num][-1]
                    action_images.append(action_on_basis(maps[a]*sign,weight_comp,module,factory,comp_num))
                    max_ind = max(max_ind,action_images[-1][-1,-2])
                sub_diff = np.concatenate(action_images)
                sub_diff[:,-2]+=offset
                offset+=max_ind+1
                total_diff.append(sub_diff)
        total_diff = np.concatenate(total_diff)
        total_diff = total_diff[np.lexsort(np.transpose(total_diff[:,:-2]))]
        total_diff_list.append(total_diff)

    total_length = sum(len(diff) for diff in total_diff_list)
    diff_entries = np.zeros((total_length,3),np.int64)
    j = -1
    offset=0
    for total_diff in total_diff_list:
        prev_row = np.zeros_like(total_diff[0,:-2])-1
        for i in range(len(total_diff)):
            row_num = i+offset
            row = total_diff[i,:-2]
            if np.any(np.not_equal(row,prev_row)):
                j+=1
                prev_row = row
            diff_entries[row_num,0] = j
            diff_entries[row_num,1:] = total_diff[i,-2:]
        offset+=len(total_diff)
    j+=1

    d_dense = matrix(ZZ,j,max(diff_entries[:,1])+1)
    for i in range(len(diff_entries)):
        d_dense[diff_entries[i,0],diff_entries[i,1]] = diff_entries[i,2]

    return d_dense