""" Defines a basic interface for handling interaction network
    as sums of deterministic components and random components
    
    We use the Storable class from ecosim to pickle
"""

import copy

import ecosim
import numpy as np


class Function(ecosim.base.Storable):
    @classmethod
    def mean_biomass(cls, S):
        b_vec = np.ones(S)
        return cls(b_vec, b_vec / S)

    def __init__(self, yields, sources):
        if len(yields) != len(sources):
            raise ValueError(
                f"Lengths of yields and sources do not coincide ({len(yields)} != {len(sources)})."
            )

        self._yields = yields
        self._sources = sources

    def randomize(self, n_shuffle, yields=True, sources=True):
        """Randomly shuffle the roles of the species within  the function"""
        for n in range(n_shuffle):
            i, j = np.random.choice(range(self.S), size=2, replace=False)
            if yields:
                yields[i], yields[j] = yields[j], yields[i]
            if sources:
                sources[i], sources[j] = sources[j], sources[i]

    @property
    def yields(self):
        return self._yields

    @yields.setter
    def yields(self, val):
        self._yields = val

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, val):
        self._sources = val

    @property
    def S(self):
        return len(self._yields)

    def __neg__(self):
        return Function(-self._yields, self._sources)

    def __mul__(self, val):
        return Function(val * self._yields, self._sources)

    def __rmul__(self, val):
        return Function(val * self._yields, self._sources)

    def __truediv__(self, val):
        return Function(self._yields / val, self._sources)

    def __rtruediv__(self, val):
        return Function(self._yields / val, self._sources)


class InteractionNetwork(ecosim.base.Storable):
    def __init__(self, S):
        self._S = S
        self._random_component = ecosim.glv.RandomLotkaVolterra.InteractionModel(
            None, {}, None
        )
        self._structure_component = None
        self._functions = []
        self._interaction_matrix = None
        self._direct_interactions = np.zeros((S, S))
        self._K = np.ones(S)

    def add_function(self, function):
        if function.S != self._S:
            raise ValueError(
                f"Number of species in Function does not match number of species in InteractionNetwork ({function.S} != {self._S})"
            )

        self._functions.append(function)

    def functions(self):
        return self._functions

    def quench(self, quench_functions=False):
        """Quench the random component and incorporate it into the direct interactions"""
        self.set_random_component("none")
        self._direct_interactions += self._last_random_sample
        self._last_random_sample = None
        if quench_functions:
            for fun in self._functions:
                self._direct_interactions += fun.yields[:, None] @ fun.sources[None, :]
            self._functions = []

    def set_random_component(self, name, **kwargs):
        """
        This function was taken from ecosim.glv.RandomLotkaVolterra.interactions as of 04/07/23
        """

        """
        Set the interaction model to one among a set of pre-defined options.
        `name` is the alias of the interaction model and `**kwarg` all its statistical 
        parameters and their desired values. Omitted statistical parameters acquire 
        default values. Will immediately sample a new matrix. 
        """

        # Pre-defined interaction models
        # These static variables are defined and loaded on class creation by the rlv_Metaclass meta-class

        if name in ecosim.glv.RandomLotkaVolterra._available_interaction_models:
            int_mod = ecosim.glv.RandomLotkaVolterra.InteractionModel(
                name,
                ecosim.glv.RandomLotkaVolterra._available_interaction_models_defaults[
                    name
                ],
                ecosim.glv.RandomLotkaVolterra._available_interaction_models[name],
            )
        else:
            raise ValueError(f"No interaction model '{name}' is implemented")

        # Check that correct arguments where given for interaction model
        if not int_mod.check_arguments(**kwargs):
            raise ValueError(f"Unknown arguments given for matrix model '{name}'")

        # First set the interaction model with its default values
        self.random_component = int_mod

        # Then update the overridden values
        for p, v in kwargs.items():
            setattr(self, p, v)

    def default_random_component(self):
        """
        Resets the interaction parameters to defaults of the interaction model
        """

        if self._random_component is None:
            self.set_random_component("gaussian")
        else:
            self.random_component = self._random_component

    @property
    def random_component(self):
        return self._random_component

    @random_component.setter
    def random_component(self, int_mod):
        # Remove attributes associated with previous interaction model
        if self._random_component is not None:
            for attr in self._random_component.params:
                delattr(self, attr)

        # Link the new interaction model
        self._random_component = int_mod

        # Add attributes of the new model (if int_mod is created through set_random_component this will add the defaults)
        for p, v in int_mod.params.items():
            setattr(self, p, v)

    @property
    def random_component_params(self):
        """A dictionary of the random model parameters and their current values"""
        return {p: getattr(self, p) for p in self._random_component.params.keys()}

    def sample_matrix(
        self, include_functions=True, include_direct_interactions=True, seed=None
    ):
        if self._random_component is not None:
            self._last_random_sample = self._random_component.sampler(
                self.S, **self.random_component_params, seed=seed
            )
            self._interaction_matrix = self._last_random_sample.copy()
        else:
            self._interaction_matrix = np.zeros((self.S, self.S))
        self._structure_component = np.zeros((self.S, self.S))
        if include_functions:
            for fun in self._functions:
                self._structure_component += fun.yields[:, None] @ fun.sources[None, :]
        if include_direct_interactions:
            self._structure_component += self._direct_interactions

        self._interaction_matrix += self._structure_component

    def __getitem__(self, val):
        if not isinstance(val, tuple):
            raise TypeError(f"Argument must be tuple of len 2, not {type(val)}.")
        elif len(val) != 2:
            TypeError(f"Argument must be of len 2, not {len(val)}.")

        if self._structure_component == None:
            raise RuntimeError("Interaction matrix has not been sampled yet.")

        return self._structure_component[val]

    def __setitem__(self, key, val):
        if not isinstance(val, tuple):
            raise TypeError(f"Argument must be tuple of len 2, not {type(val)}.")
        elif len(val) != 2:
            TypeError(f"Argument must be of len 2, not {len(val)}.")

        self._direct_interactions[key] = val

    @property
    def interaction_matrix(self):
        return self._interaction_matrix

    @property
    def structure_component(self):
        return self._structure_component

    @property
    def direct_interactions(self):
        return self._direct_interactions

    @direct_interactions.setter
    def direct_interactions(self, val):
        self._direct_interactions = val

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, val):
        self._S = val

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, val):
        if isinstance(val, (int, float)):
            self._K = val * np.ones(self.S)
        else:
            self._K = val


"""A wrapper around the ecosim.LotkaVolterra class to handle the InteractionNetwork class
This class has been written so that ecosim.scan.ModelScan can directly act on it as if it were an ecosim.base.Model
"""


class DynamicModel(ecosim.base.Storable):
    def __init__(self, S):
        self._interaction_network = InteractionNetwork(
            S
        )  # it is important that this is the first created object, otherwise __setattr__ will fail
        self._simulator = ecosim.glv.LotkaVolterra(S)

    def run(self, time, record_interval=None, **kwargs):
        self.resample_interactions(**kwargs)
        self._simulator.mat_alpha = self._interaction_network.interaction_matrix
        return self._simulator.run(time, record_interval)

    def functions_from_trajectory(self, trajectory, functions=None):
        if functions == None:
            functions = self._interaction_network.functions()
        f_mat = np.array([f.sources for f in functions])

        new_traj_mat = np.zeros((len(trajectory.time_points), len(functions)))
        for i, t in enumerate(trajectory.time_points):
            new_traj_mat[i] = f_mat @ trajectory[t]

        return ecosim.base.Trajectory(trajectory.vec_t, new_traj_mat, parent_sim=self)

    def set_current_state(self, vec_x):
        self._simulator.vec_x = vec_x

    """From now on, the idea is that the properties of the InteractionNetwork class should only be modified
    throught he wrapper functions of the DynamicModel class. We don't provide in place access functions for the _interaction_network
    but we provide copy access functions so that a copy of it can be used somewhere else
    Similarly, properties of the LotkaVolterra class, namely the parameters K, lambda, are accesed through this UI.
    """

    def default_random_component(self):
        """
        Resets the interaction parameters to defaults of the interaction model
        """
        self._interaction_network.default_random_component()

    def set_random_component(self, name, **kwargs):
        """Delete InteractionModel attributes from the DynamicModel"""
        for key in self._interaction_network._random_component.params:
            if hasattr(self, key):
                delattr(self, key)
        self._interaction_network.set_random_component(name, **kwargs)
        """Add the new InteractionModel attributes to the DynamicModel scope"""
        for key, val in self._interaction_network.random_component_params.items():
            setattr(self, key, val)

    @property
    def random_component_params(self):
        """A dictionary of the random model parameters and their current values"""
        return self._interaction_network.random_component_params

    """__setattr__ override: if the name of the attribute coincides with a parameter of the _random_component InteractionModel
    then we consider that the user is changing that attribute as well, and so we update the value of its corresponding parameter
    in the InteractionNetwork class. Attributes defined only in the InteractionNetwork class are not affected.
    """

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name != "_interaction_network"
            and name in self._interaction_network._random_component.params
        ):
            setattr(self._interaction_network, name, value)

    def add_function(self, function):
        self._interaction_network.add_function(function)

    def functions(self):
        return self._interaction_network.functions()

    @property
    def S(self):
        return self._S

    def transform_network(self, tr):
        tr(self._interaction_network)

    """No in-place access to the interaction network
    """

    @property
    def interaction_network(self):
        raise RuntimeError(
            "In place access to InteractionNetwork is not implemented. Use the wrapper functions of the DynamicModel class to modify it instead."
        )

    def interaction_network_copy(self):
        return copy.deepcopy(self._interaction_network)

    """To avoid mismatches this function creates a copy of the provided interaction network so that the two become independent.
    """

    @interaction_network.setter
    def interaction_network(self, i_network):
        """Delete InteractionModel attributes from the DynamicModel"""
        for key in self._interaction_network._random_component.params:
            if hasattr(self, key):
                delattr(self, key)
        self._interaction_network = copy.deepcopy(i_network)
        """Add the new InteractionModel attributes to the DynamicModel scope"""
        for key, val in self._interaction_network.random_component_params.items():
            setattr(self, key, val)

    @property
    def interaction_matrix(self):
        return self._interaction_network.interaction_matrix

    @property
    def structure_component(self):
        return self._interaction_network.structure_component

    @property
    def direct_interactions(self):
        return self._interaction_network.direct_interactions

    @direct_interactions.setter
    def direct_interactions(self, val):
        self._interaction_network.direct_interactions = val

    @property
    def r(self):
        return self._simulator.r

    @r.setter
    def r(self, val):
        self._simulator.r = val

    @property
    def K(self):
        return self._simulator.K

    @K.setter
    def K(self, val):
        self._simulator.K = val

    @property
    def lam(self):
        return self._simulator.lam

    @lam.setter
    def lam(self, val):
        self._simulator.lam = val

    def resample_interactions(self, **kwargs):
        self._interaction_network.sample_matrix(**kwargs)

    def set_integrator(self, name, *, log_transform=False, **kwargs):
        self._simulator.set_integrator(name, log_transform=log_transform, **kwargs)
