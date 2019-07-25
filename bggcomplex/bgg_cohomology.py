"""
Deprecated and replaced by fast_module.py as soon as quotients are implemented into fast_module.py
"""

from collections import defaultdict
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from itertools import chain
from time import time


class BGGCohomologyComputer(object):
    def __init__(self, BGG, weight_module):
        self.BGG = BGG
        self.BGG.compute_signs() # we definitely need the signs for this computation
        self.weight_module = weight_module
        self.timer = defaultdict(int)
        timer = time()
        self.all_weights, self.regular_weights = self.BGG.compute_weights(self.weight_module)
        self.timer['BGG_weights']+=time()-timer

        self.maps = dict()


    def get_vertex_weights(self,weight):
        vertex_weights = dict()
        for w, reflection in self.BGG.reduced_word_dic.items():
            new_weight = self.BGG.weight_to_alpha_sum(reflection.action(weight + self.BGG.rho) - self.BGG.rho)
            vertex_weights[w] = new_weight
        return vertex_weights

    def bgg_differential(self, weight, i):
        """Compute the BGG differential delta_i: E_i\to E_{i+1} on the quotient weight module."""
        timer = time()
        vertex_weights = self.get_vertex_weights(weight)
        self.timer['vertex_weights']+=time()-timer

        timer = time()
        maps = self.BGG.compute_maps(weight)
        self.timer['compute_maps']+=time()-timer

        # Get vertices of the ith column
        column = self.BGG.column[i]

        # Find all the arrows going from ith column to (i+1)th column in form (w, [w->w',w->w'',...])
        delta_i_arrows = [(w, [arrow for arrow in self.BGG.arrows if arrow[0] == w]) for w in column]

        # Initialize output. Since there could be two arrows with the same target, we use counter to sum results.
        output_dic = dict()

        for initial_vertex, arrows in delta_i_arrows:
            # Get the section for the initial weight w
            timer=time()
            initial_section = self.weight_module.get_section(vertex_weights[initial_vertex])
            self.timer['get_section']+=time()-timer

            # Initialize output dic
            for row_index, _ in enumerate(initial_section):
                output_dic[(initial_vertex,row_index)] = defaultdict(int)

            # Compute $s_w'^\top \sigma(a) F(a) \,s_w$ for each a:w->w'
            for a in arrows:
                sign = self.BGG.signs[a]
                pbw = maps[a]

                timer=time()
                final_section = self.weight_module.get_section(vertex_weights[a[1]])
                self.timer['get_section']+=time()-timer

                timer=time()
                # Compute image for each row of the initial_section, add results together
                for index, row in enumerate(initial_section):

                    # First the image of the row without applying the transpose section
                    image_before_section = defaultdict(int)

                    # Compute the action of the PBW element on each of the basis components
                    # of the row of the section and add the result together

                    timer2 = time()
                    for key, coeff in row.items():
                        action_on_element = self.weight_module.pbw_action(pbw, self.weight_module.basis()[key])

                        action_on_element *= coeff * sign
                        monomial_coeffs = action_on_element.monomial_coefficients()

                        for mon_key, mon_coeff in monomial_coeffs.items():
                            image_before_section[mon_key] += mon_coeff
                    self.timer['image_before_section']+=time()-timer2

                    # Compute the image of the row under the transpose section, add the result to the output
                    if len(image_before_section) > 0:
                        final_image = self.section_transpose_image(final_section, image_before_section, a[1])
                        for mon_key, mon_coeff in final_image.items():
                            output_dic[(initial_vertex, index)][mon_key] += mon_coeff
                self.timer['compute_image']+=time()-timer

        timer=time()
        output_keys = [(w, row_index)for w in self.BGG.column[i+1]
                       for row_index, _ in enumerate(self.weight_module.get_section(vertex_weights[w]))
                       ]
        delta_i_matrix = self.vectorize_dictionaries(output_dic, key_list=output_keys)

        self.timer['vectorizer']+=time()-timer
        return delta_i_matrix

    def compute_cohomology(self,weight,i):
        diff_i = self.bgg_differential(weight, i)
        diff_i_1 = self.bgg_differential(weight, i - 1)
        timer = time()
        cohom_dim= diff_i.dimensions()[0] - diff_i.rank() - diff_i_1.rank()
        self.timer['matrix_rank']=time()-timer
        return cohom_dim

    def compute_full_cohomology(self,i):
        length_i_weights = [triplet for triplet in self.regular_weights if triplet[2] == i]

        dominant_non_trivial = set()
        dominant_trivial = []
        for w, w_dom, _ in length_i_weights:
            for _, w_prime_dom, l in self.regular_weights:
                if w_prime_dom == w_dom and (l == i + 1 or l == i - 1):
                    dominant_non_trivial.add(w_dom)
                    break
            else:
                dominant_trivial.append((w,w_dom))
        cohomology = defaultdict(int)

        for w,w_dom in dominant_trivial:
            cohom_dim = len(self.weight_module.get_section(w))
            if cohom_dim>0:
                cohomology[w_dom]+= cohom_dim
        for w in dominant_non_trivial:
            cohom_dim = self.compute_cohomology(w, i)
            if cohom_dim>0:
                cohomology[w]+= cohom_dim
        return sorted(cohomology.items(),key=lambda t:t[-1])

    @staticmethod
    def section_transpose_image(section, monomial_coeffs, section_index):
        """computes the image of the transpose of the section, reports the coefficients as (index,row_number) where
        index is supplied and should be the w' vertex of BGG, and row_number is the number of the row of the section"""
        output = defaultdict(int)
        for row_index, row in enumerate(section):
            for monomial_key, monomial_coeff in monomial_coeffs.items():
                if monomial_key in row:
                    output[(section_index, row_index)] += row[monomial_key] * monomial_coeff
        return output

    @staticmethod
    def vectorize_dictionaries(list_of_dics, key_list=None):
        """Turn a list of dictionaries (or a dictionary of dictionaries) into a dense integer matrix"""

        if isinstance(list_of_dics, dict):
            list_of_dics = list_of_dics.values()

        if key_list is None:
            keys = set(chain.from_iterable(dic.keys() for dic in list_of_dics))
            key_map = {key: i for i, key in enumerate(keys)}
        else:
            key_map = {key: i for i, key in enumerate(key_list)}

        output = matrix(ZZ, len(list_of_dics), len(key_map))
        for row_number, row in enumerate(list_of_dics):
            for key, value in row.items():
                output[row_number, key_map[key]] = value
        return output
