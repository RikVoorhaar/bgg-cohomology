from collections import defaultdict, Counter
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from itertools import chain


class BGGCohomologyComputer(object):
    def __init__(self, BGG, weight_module, degree):
        self.BGG = BGG
        self.BGG.compute_signs() # we definitely need the signs for this computation
        self.weight_module = weight_module
        self.degree = degree
        self.all_weights, self.regular_weights = self.BGG.compute_weights(self.weight_module)

        self.maps = dict()

    def bgg_differential(self, weight, i):
        """Compute the BGG differential delta_i: E_i\to E_{i+1} on the quotient weight module."""
        vertex_weights = dict()
        for w, reflection in self.BGG.reduced_word_dic.items():
            new_weight = self.BGG.weight_to_alpha_sum(reflection.action(weight + self.BGG.rho) - self.BGG.rho)
            vertex_weights[w] = new_weight

        if weight in self.maps:
            maps = self.maps[weight]
        else:
            self.maps[weight] = self.BGG.compute_maps(weight)
            maps = self.maps[weight]

        # Get vertices of the ith column
        column = self.BGG.column[i]

        # Find all the arrows going from ith column to (i+1)th column in form (w, [w->w',w->w'',...])
        delta_i_arrows = [(w, [arrow for arrow in self.BGG.arrows if arrow[0] == w]) for w in column]

        # Initialize output. Since there could be two arrows with the same target, we use counter to sum results.
        output_dic = defaultdict(Counter)

        for initial_vertex, arrows in delta_i_arrows:
            # Get the section for the initial weight w
            initial_section = self.weight_module.get_section(vertex_weights[initial_vertex])

            # Compute $s_w'^\top \sigma(a) F(a) \,s_w$ for each a:w->w'
            for a in arrows:
                sign = self.BGG.signs[a]
                pbw = maps[a]
                final_section = self.weight_module.get_section(vertex_weights[a[1]])

                # Compute image for each row of the initial_section, add results together
                for index, row in enumerate(initial_section):

                    # First the image of the row without applying the transpose section
                    image_before_section = Counter()

                    # Compute the action of the PBW element on each of the basis components
                    # of the row of the section and add the result together
                    for key, coeff in row.items():
                        action_on_element = self.weight_module.pbw_action(pbw, self.weight_module.basis()[key])

                        action_on_element *= coeff * sign
                        monomial_coeffs = action_on_element.monomial_coefficients()

                        for mon_key, mon_coeff in monomial_coeffs.items():
                            image_before_section[mon_key] += mon_coeff

                    # Compute the image of the row under the transpose section, add the result to the output
                    if len(image_before_section) > 0:
                        final_image = self.section_transpose_image(final_section, image_before_section, a[1])
                        #output_dic[(initial_vertex, index)] = output_dic[(initial_vertex, index)] + final_image
                        for mon_key, mon_coeff in final_image.items():
                            output_dic[(initial_vertex, index)][mon_key] += mon_coeff
        delta_i_matrix = self.vectorize_dictionaries(output_dic)
        return delta_i_matrix

    @staticmethod
    def section_transpose_image(section, monomial_coeffs, section_index):
        """computes the image of the transpose of the section, reports the coefficients as (index,row_number) where
        index is supplied and should be the w' vertex of BGG, and row_number is the number of the row of the section"""
        output = Counter()
        for row_index, row in enumerate(section):
            for monomial_key, monomial_coeff in monomial_coeffs.items():
                if monomial_key in row:
                    output[(section_index, row_index)] += row[monomial_key] * monomial_coeff
        return output

    @staticmethod
    def vectorize_dictionaries(list_of_dics, key_map=None):
        """Turn a list of dictionaries (or a dictionary of dictionaries) into a dense integer matrix"""

        if isinstance(list_of_dics, dict):
            list_of_dics = list_of_dics.values()

        if key_map is None:
            keys = set(chain.from_iterable(dic.keys() for dic in list_of_dics))
            key_map = {key: i for i, key in enumerate(keys)}

        output = matrix(ZZ, len(list_of_dics), len(key_map))
        for row_number, row in enumerate(list_of_dics):
            for key, value in row.items():
                output[row_number, key_map[key]] = value
        return output
