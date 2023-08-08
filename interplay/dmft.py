import ecosim
import numpy as np
import scipy.integrate

""" Classes for solving dmft equations
"""


class dmftModel(ecosim.base.Storable):
    def __init__(self, S):
        self._S = S
        self._interaction_network = None

    def full_solve(self):
        raise RuntimeError("full_solve has not been implemented for this model.")
    
    def test_data(self):
        raise RuntimeError("test_data has not been implemented for this model.")

    @property
    def K(self):
        return self._interaction_network.K

    @K.setter
    def K(self, val):
        self._interaction_network.K = val
        
    @staticmethod
    def test_data(data,*args):
        ''' The purpose of test_data is to provide a way to test how far from a model the provided data is.
        It is implemented by each model, and should return a number that is small if the data is close to the model.
        '''
        return RuntimeError("test_data has not been implemented for this model.")

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
    
    @staticmethod
    def test_data(data,*args):
        return RuntimeError("test_data has not been implemented for this model.")
