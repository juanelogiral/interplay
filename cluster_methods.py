'''Function callables for clustering within the InteractionClustering class
'''

import ecosim
from base import InteractionNetwork
import tqdm
import numpy as np

# This finds k clusters by applying a Metropolis algorithm similar to the one in lv_optimize_clusters but with the projection
# norm as target function

def cluster_spectral_optimize_metropolis(i_network,k,T,max_iter):
    
    N = i_network.S
    A = i_network.structure_component
    
    A2 = np.dot(A.T,A)
    sval,svec = np.linalg.eigh(A2)
    #truncate svec to keep only k vectors
    for i in range(N-k):
        svec[:,i] = 0
    
    grp_idx = np.random.randint(low=0,high=k,size = N)
    grp_idx_list = [grp_idx]
    
    err = np.inf
    err_list = [np.inf]
    
    for i in tqdm(range(max_iter),total=max_iter):
        # at each iteration, swap ONE individual from one group to another
        new_grp = grp_idx.copy()
        new_grp[np.random.randint(low=0,high=N)] = np.random.randint(low=0,high=k)
        #group vectors
        group_vec = np.zeros((k,N))
        for i in range(N):
            group_vec[new_grp[i],i] = 1
        for gp in group_vec:
            if np.linalg.norm(gp)>0:
                gp /= np.linalg.norm(gp)
        #projection matrix
        p_grp = np.dot(group_vec.T,group_vec)
        
        err_2 = np.linalg.norm(np.dot(np.identity(N)-p_grp,svec),ord='fro') 
        
        if err_2 < err or (T > 0 and np.random.random() < np.exp(- (err_2-err)/T)):
            grp_idx = new_grp
            err = err_2
            grp_idx_list.append(grp_idx)
            err_list.append(err)
                
    best = np.argmin(err_list)
    grp_idx = grp_idx_list[best]
    
    additional_dict = {'singular_values' : np.flip(sval), 'error_list' : err_list}
    
    return [np.where(grp_idx == l)[0] for l in np.unique(grp_idx)],grp_idx,np.min(err_list),additional_dict
