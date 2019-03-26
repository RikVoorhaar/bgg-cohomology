from numpy.random import randint
from itertools import chain

def compute_signs(BGG):
    """Computes signs for all the edges so that the product of signs around any admissible cycle is -1"""
    edges = BGG.arrows
    BGG.find_cycles()

    #number all the edges and save all the cycles by the index of the edges contained in them
    edge_numbers = range(len(edges))
    edge_dic = {a:i for i,a in enumerate(edges)}
    cycles= [[edge_dic[(c[0],c[1])],edge_dic[(c[1],c[2])],edge_dic[(c[3],c[2])],edge_dic[(c[4],c[3])]] for c in BGG.cycles]

    #make a dictionary with keys edge indices and values a list of the cycles the edge is contained in
    edge_to_cycle = {i:[] for i,_ in enumerate(edges)}
    for i,c in enumerate(cycles):
        for e in c:
            edge_to_cycle[e]+=[i]

    #the total sign of a cycle is the product of signs of constituent edges. We aim these to all be -1
    def _total_sign(c):
        return signs[c[0]]*signs[c[1]]*signs[c[2]]*signs[c[3]]
    
    #flipping an edge reduces the total number of bad cycles if the sum of total signs of cycles it is contained in is positive
    def _check_choice(edge):
        edge_cycles = [cycles[i] for i in edge_to_cycle[edge]]
        sign_sum= sum(map(_total_sign,edge_cycles))
        return sign_sum > 0
    
    #count the number of bad cycles
    def _count_bad_cycles():
        counter=0
        for cycle in cycles:
            if _total_sign(cycle)==1:
                counter+=1
        return counter

    #inintialize
    signs=list((-1)**randint(2,size=len(edges)))
    bad_cycle_counter = _count_bad_cycles()
    edges_to_be_randomized = 1
    step_counter=0

    while bad_cycle_counter>0:
        #keep a list with True/False for all the edges depending on whether flipping them would reduce the number of bad cycles
        edge_status = list(map(_check_choice,edge_numbers))
        while True:
            try:
                #find the first edge which would be benificial to flip and flip it. if there are none, break out of the loop
                flipped_edge=next(i for i in edge_numbers if edge_status[i])
                signs[flipped_edge]*=-1
            except StopIteration:
                break
            
            #find the edges whose status potentially changed by flipping the edge and recompute their status
            check_queue=set(chain.from_iterable([cycles[c] for c in edge_to_cycle[flipped_edge]]))
            for e in check_queue:
                edge_status[e] = _check_choice(e)
            step_counter+=1

        #if there was no improvement over last time we ran out of edges to flip, we flip a bunch of edges at random
        #this number of edges to be flipped at random doubles every time, and resets whenever we did get an improvement
        current_bad_cycles = _count_bad_cycles()
        if current_bad_cycles<bad_cycle_counter:
            bad_cycle_counter=current_bad_cycles
            edges_to_be_randomized=1
        else:
            for i in np.choice(edge_numbers,size=edges_to_be_randomized):
                signs[i]*=-1
            edges_to_be_randomized*=2
            edges_to_be_randomized=max(edges_to_be_randomized,len(signs))

    return {e:s for e,s in zip(edges,signs)}