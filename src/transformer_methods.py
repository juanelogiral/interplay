import numpy as np
import scipy
from math import sqrt
from scipy.linalg import expm

default_args = {}

default_args["orthogonal_mixing"] = {}
default_args["mode_mixing"] = {}
default_args["infinitesimal_rotation"] = {"dt": 0.1}


def orthogonal_mixing(network):
    U = scipy.stats.ortho_group.rvs(network.S)
    for f in network.functions():
        f.yields = U @ f.yields
        f.sources = U @ f.sources
    network.direct_interactions = U @ network.direct_interactions @ U.T


def mode_mixing(network):
    for f in network.functions():
        A = np.transpose([f.yields, f.sources])
        np.random.shuffle(A)
        f.yields, f.sources = A.T

    u, eig, vh = np.linalg.svd(network.direct_interactions)
    for i in range(network.S):
        A = np.transpose([u[:, i], vh[i]])
        np.random.shuffle(A)
        u[:, i], vh[i] = A.T
    network.direct_interactions = u @ np.diag(eig) @ vh

def infinitesimal_rotation(network,dt):
    M_rand = np.random.normal(0,1,(network.S,network.S))
    U = expm(sqrt(dt/network.S)*(M_rand - M_rand.T)/sqrt(2))
    
    for f in network.functions():
        f.yields = U @ f.yields
        f.sources = U @ f.sources
    network.direct_interactions = U @ network.direct_interactions @ U.T
    