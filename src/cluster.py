"""A class for clustering a Lotka-Volterra interaction network
into 'functional' groups
"""

import ecosim
from .base import InteractionNetwork
import cluster_methods
from inspect import getmembers, isfunction
import numpy as np


class icm_metaclass(type):
    def __new__(cls, name, bases, dict):
        icm = super().__new__(cls, name, bases, dict)

        # Available functions should be defined in cluster_methods and have the syntax cluster_*1_optimize_*2 where
        # *1 and *2 stand for the clustering method and the optimization methods respectively.

        # Optimization methods should have signature fun(InteractionNetwork,n_clusters,**kwargs) -> (list,list,float,dict)
        # The first list has len(n_clusters) and list[i] contains elements in cluster i
        # The second list has len S and list[i] contains the cluster of species i
        # The float is a meaure of clustering accuracy
        # The dict contains any other useful data

        icm._available_clustering_functions = {}
        for key, adress in getmembers(cluster_methods, isfunction):
            split = key.split("_")
            key_c_method = split[1]
            key_o_method = split[3]
            icm._available_clustering_functions[(key_c_method, key_o_method)] = adress

        return icm


class InteractionClusteringModel(ecosim.base.Storable, metaclass=icm_metaclass):
    def __init__(self):
        self._clustering_method = "spectral"
        self._optimization_method = "metropolis"

        self._list_of_clusters = None
        self._list_of_species = None
        self._accuracy = None

        self._clustered_model = None

    @property
    def clustering_method(self):
        return self._clustering_method

    @clustering_method.setter
    def clustering_method(self, val):
        self._clustering_method = val

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def optimization_method(self):
        return self._optimization_method

    @optimization_method.setter
    def optimization_method(self, val):
        self._optimization_method = val

    @property
    def clustered_network(self):
        return self._clustered_model

    def default_methods(self):
        self.clustering_method = "spectral"
        self.optimization_method = "metropolis"

    def get_method_params(self):
        if (
            not (self._clustering_method, self._optimization_method)
            in InteractionClusteringModel._available_clustering_functions
        ):
            raise ValueError(
                f"The pair {(self._clustering_method,self._optimization_method)} does not match any clustering function."
            )

        code_alias = InteractionClusteringModel._available_clustering_functions[
            (self.clustering_method, self.optimization_method)
        ].__code__
        return code_alias.co_varnames[2 : code_alias.co_argcount]

    def __call__(self, interaction_network, n_clusters, **kwargs):
        if (
            not (self._clustering_method, self._optimization_method)
            in InteractionClusteringModel._available_clustering_functions
        ):
            raise ValueError(
                f"The pair {(self._clustering_method,self._optimization_method)} does not match any clustering function."
            )

        (
            cluster_sp_map,
            sp_cluster_map,
            accuracy,
            info_dict,
        ) = InteractionClusteringModel._available_clustering_functions[
            (self._clustering_method, self._optimization_method)
        ](
            interaction_network, n_clusters, **kwargs
        )

        self._list_of_clusters = cluster_sp_map
        self._list_of_species = sp_cluster_map
        self._accuracy = accuracy
        for key, val in info_dict:
            setattr(self, key, val)

        # Generate a clustered interaction network
        S = interaction_network.S
        self._clustered_model = InteractionNetwork(S)

        new_mat = np.zeros((S, S))
        old_mat = interaction_network.structure_component

        for c1, c2 in self:
            new_mat[np._ix(c1, c2)] = (
                old_mat[np._ix(c1, c2)] @ np.ones(len(c1), len(c2)) / len(c2)
            )

        self._clustered_model.structure_component = new_mat

    def __iter__(self):
        if self._list_of_clusters == None:
            raise RuntimeError("No clustering has been performed yet.")
        return iter(self._list_of_clusters)

    def __getitem__(self, i):
        if self._list_of_clusters == None:
            raise RuntimeError("No clustering has been performed yet.")

        return self._list_of_clusters[i]

    def species_clusters(self):
        if self._list_of_clusters == None:
            raise RuntimeError("No clustering has been performed yet.")

        return self._list_of_species
