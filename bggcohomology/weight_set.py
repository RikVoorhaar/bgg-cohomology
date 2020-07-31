"""Module for doing computations with the weights of a weight module and action of Weyl group."""

import numpy as np

from sage.combinat.root_system.weyl_group import WeylGroup
from sage.matrix.constructor import matrix


class WeightSet:
    """Class to do simple computations with the weights of a weight module.

    Parameters
    ----------
    root_system : str
        String representing the root system (e.g. 'A2')

    Attributes
    ----------
    root_system : str
        String representing the root system (e.g. 'A2')
    W : WeylGroup
        Object encoding the Weyl group.
    weyl_dic : dict(str, WeylGroup.element_class)
        Dictionary mapping strings representing a Weyl group element as reduced word
        in simple reflections, to the Weyl group element.
    reduced_words : list[str]
        Sorted list of all strings representing Weyl group elements as reduced word in
        simple reflections.
    simple_roots : list[RootSpace.element_class]
        List of the simple roots
    rank : int
        Rank of root system
    rho : RootSpace.element_class
        Half the sum of all positive roots
    pos_roots : List[RootSpace.element_class]
        List of all the positive roots
    action_dic : Dict[str, np.array(np.int32, np.int32)]
        dictionary mapping each string representing an
        element of the Weyl group to a matrix expressing the action on the simple roots.
    rho_action_dic : Dict[str, np.array(np.int32)]
        dictionary mapping each string representing an
        element of the Weyl group to a vector representing the image
        of the dot action on rho.
    """

    @classmethod
    def from_bgg(cls, BGG):
        """Initialize from an instance of BGGComplex.

        Some data can be reused, and this gives roughly 3x faster initialization.
        
        Parameters
        ----------
        BGG : BGGComplex
            The BGGComplex to initialize from.
        """
        hot_start = {"W": BGG.W, "weyl_dic": BGG.reduced_word_dic}
        return cls(BGG.root_system, hot_start=hot_start)

    def __init__(self, root_system, hot_start=None):
        self.root_system = root_system

        if hot_start is None:
            self.W = WeylGroup(root_system)
            self.weyl_dic = self._compute_weyl_dictionary()
        else:
            self.W = hot_start["W"]
            self.weyl_dic = hot_start["weyl_dic"]

        self.domain = self.W.domain()

        self.reduced_words = sorted(self.weyl_dic.keys(), key=len)

        self.simple_roots = self.domain.simple_roots().values()
        self.rank = len(self.simple_roots)
        self.rho = self.domain.rho()

        self.pos_roots = self.domain.positive_roots()

        # Matrix of all simple roots, for faster matrix solving
        self.simple_root_matrix = matrix(
            [list(s.to_vector()) for s in self.simple_roots]
        ).transpose()

        self.action_dic, self.rho_action_dic = self.get_action_dic()

    def _compute_weyl_dictionary(self):
        """Construct a dictionary enumerating all of the elements of the Weyl group.

        The keys are reduced words of the elements
        """
        reduced_word_dic = {
            "".join([str(s) for s in g.reduced_word()]): g for g in self.W
        }
        return reduced_word_dic

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
        """Inverse of `weight_to_tuple`.
        
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
            action_dic[s] = np.array(action_mat, dtype=np.int32)

            # Encode the dot action of w on rho.
            rho_action_dic[s] = np.array(
                self.weight_to_tuple(w.action(self.rho) - self.rho), dtype=np.int32
            )
        return action_dic, rho_action_dic

    def dot_action(self, w, mu):
        """Compute the dot action of w on mu.
        
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
            np.matmul(self.action_dic[w].T, np.array(mu, dtype=np.int32))
            + self.rho_action_dic[w]
        )

    def dot_orbit(self, mu):
        """Compute the orbit of the Weyl group action on a weight.

        Parameters
        ----------
        mu : iterable(int)
            A weight

        Returns
        -------
        dict(str, np.array[np.int32])
            Dictionary mapping Weyl group elements to weights encoded as numpy vectors.
        """
        return {w: self.dot_action(w, mu) for w in self.reduced_words}

    def is_dot_regular(self, mu):
        """Check if mu has a non-trivial stabilizer under the dot action.
        
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
        """Find dot-regular weights and associated dominant weights of a set of weights.

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
        """Use sagemath built-in function to check if weight is dominant.
        
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
        """Give dimension of highest weight representation of integral dominant weight.

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
