import ecosim
import scipy.integrate
import numpy as np


""" Classes for solving dmft equations
"""


class dmftModel(ecosim.base.Storable):
    def __init__(self, S):
        self._S = S
        self._interaction_network = None

    def full_solve(self):
        raise RuntimeError("full_solve has not been implemented for this model.")

    @property
    def K(self):
        return self._interaction_network.K

    @K.setter
    def K(self, val):
        self._interaction_network.K = val


class EqdmftModel(dmftModel):
    def __init__(self, S):
        super().__init__(S)
        self._SAD = None
        self._SAD_bounds = (0, np.inf)
        self._self_avg_quantities = {}

    @property
    def interaction_network(self):
        return self._interaction_network

    @interaction_network.setter
    def interaction_network(self, val):
        self._interaction_network = val

    @property
    def SAD(self):
        return self._SAD

    def equilibrium_average(self, fun):
        """Given a function f(x) where x are abundances, returns its average value at equilibrium"""
        return scipy.integrate.quad(
            lambda x: fun(x) * self._SAD(x), self._SAD_bounds[0], self._SAD_bounds[1]
        )[0]

    def full_solve(self):
        raise RuntimeError("full_solve has not been implemented for this model.")
