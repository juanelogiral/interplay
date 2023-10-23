import numpy as np

def block(block_sizes,block_interactions):
    S = np.sum(block_sizes)
    n = len(block_sizes)
    A = np.zeros((S,S))
    for i in range(n):
        for j in range(n):
            A[i,j] = block_interactions[i,j]
    return A

def circulant(S,f):
    A = np.zeros((S,S))
    for i in range(S):
        for j in range(S):
            A[i,j] = f(i-j)
    return A

def hierarchical(S,f):
    A = np.zeros((S,S))
    for i in range(S):
        for j in range(S):
            A[i,j] = f(i,j)
    return A

def hierarchical_uniform(S,mu,dmu):
    A = np.zeros((S,S)) 
    for i in range(S):
        for j in range(S):
            A[i,j] = mu + dmu*np.sign(i-j)
    return A

def nested(S,f):
    A = np.zeros((S,S))
    for i in range(S):
        for j in range(S):
            A[i,j] = f(i+j)
    return A