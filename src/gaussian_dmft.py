from .base import InteractionNetwork
from .dmft import EqdmftModel
import numpy as np
from scipy.special import erf
from math import sqrt,exp,pi

''' A specific class for solving dmft models where the interaction matrix reads a_ij = mu_ij + sig * z_ij for z_ij gaussian
variables
'''

class Eqdmft_gaussian(EqdmftModel):
    
    def __init__(self,S):
        super().__init__(S)
        self._S =S
        self._interaction_network = InteractionNetwork(S)
        self._interaction_network.set_random_component('gaussian')
    
    @property
    def S(self):
        return self._S
    
    @property
    def sig(self):
        return self._interaction_network.sig
    @sig.setter
    def sig(self,val):
        self._interaction_network.sig=val
    
    @property
    def gam(self):
        return self._interaction_network.gam
    @gam.setter
    def gam(self,val):
        self._interaction_network.gam=val
    
    @property
    def interaction_network(self):
        return self._interaction_network
    @interaction_network.setter
    def interaction_network(self,val):
        val.set_random_component('gaussian')
        self._interaction_network = val
        
    
    def full_solve(self,verbose = True,n_iter=200,r=.25):
        omega0 = lambda delta : (1+erf(delta/sqrt(2)))/2
        omega1 = lambda delta : delta*omega0(delta) + exp(-delta**2 / 2)/sqrt(2*pi)
        omega2 = lambda delta : omega0(delta) * (1+delta**2) + delta* exp(-delta**2 / 2) / sqrt(2*pi)
        
        omega0 = np.vectorize(omega0)
        omega1 = np.vectorize(omega1)
        omega2 = np.vectorize(omega2)
        
        self._interaction_network.sample_matrix()
        mu = self._interaction_network.structure_component
        
        S = self.S
        K = self.K
        sig = self.sig
        gam = self.gam
        
        op1 = lambda y,q,chi : np.array([np.sum(mu[i] * omega1(y+K/sqrt(q*sig**2)))/(1-gam*sig**2*chi) for i in range(S)])
        op2 = lambda y,q,chi : np.mean(omega0(y+K/sqrt(q*sig**2)))/(1-gam*sig**2*chi)
        op3 = lambda y,q,chi : sig**2 *q * np.mean(omega2(y+K/sqrt(q*sig**2)))/(1-gam*sig**2*chi)**2
        
        q0 = 1
        chi0 = 1
        y0 = np.ones(S)
        
        if verbose:
            for i in range(n_iter):
                y0 = y0*(1-r) + r * op1(y0,q0,chi0)
                chi0 = chi0*(1-r) + r * op2(y0,q0,chi0)
                q0 = q0*(1-r) + r * op3(y0,q0,chi0)
            print("Errors are (%f , %f , %f)"%(np.linalg.norm(y0-op1(y0,q0,chi0)),chi0-op2(y0,q0,chi0),q0-op3(y0,q0,chi0)))
        else:
            for i in range(n_iter):
                y0 = y0*(1-r) + r * op1(y0,q0,chi0)
                chi0 = chi0*(1-r) + r * op2(y0,q0,chi0)
                q0 = q0*(1-r) + r * op3(y0,q0,chi0)
        
        
        self._SAD = lambda x : (1-gam*sig**2*chi0)*np.mean([exp(-(x*(1-gam*sig**2*chi0)/sqrt(q0*sig**2)- y0[i] - K[i]/sqrt(q0*sig**2))**2 /2) for i in range(S)]) / sqrt(2*pi * sig**2 *q0)
        self._SAD_bounds = (0,(4 + np.max(y0) - K/sqrt(q0*sig**2)) * sqrt(q0*sig**2) /(1-gam*sig**2*chi0))
        self._self_avg_quantities['q']= q0
        self._self_avg_quantities['chi']= chi0
        self._self_avg_quantities['y'] =y0


''' A specific class for solving dmft models where the interaction matrix reads a_ij = mu + sig * z_ij for z_ij gaussian
variables
'''

class Eqdmft_uniform_gaussian(EqdmftModel):
    
    def __init__(self,S):
        super().__init__(S)
        self._S =S
        self._interaction_network = InteractionNetwork(S)
        self._interaction_network.set_random_component('gaussian')
    
    @property
    def S(self):
        return self._S
    
    @property
    def sig(self):
        return self._interaction_network.sig
    @sig.setter
    def sig(self,val):
        self._interaction_network.sig=val
    
    @property
    def gam(self):
        return self._interaction_network.gam
    @gam.setter
    def gam(self,val):
        self._interaction_network.gam=val
    
    @property
    def mu(self):
        return self._interaction_network.mu
    @mu.setter
    def mu(self,val):
        self._interaction_network.mu=val
    
    @property
    def interaction_network(self):
        return self._interaction_network
    @interaction_network.setter
    def interaction_network(self,val):
        val.set_random_component('gaussian')
        self._interaction_network = val
        
    
    def full_solve(self,verbose = True,n_iter=200,r=.25):
        omega0 = lambda delta : (1+erf(delta/sqrt(2)))/2
        omega1 = lambda delta : delta*omega0(delta) + exp(-delta**2 / 2)/sqrt(2*pi)
        omega2 = lambda delta : omega0(delta) * (1+delta**2) + delta* exp(-delta**2 / 2) / sqrt(2*pi)
        
        self._interaction_network.sample_matrix()
        mu = self._interaction_network.structure_component
        
        S = self.S
        K = self.K[0]
        sig = self.sig
        mu = self.mu
        gam = self.gam
        
        op1 = lambda y,q,chi : mu * omega1(y+K/sqrt(q*sig**2))/(1-gam*sig**2*chi)
        op2 = lambda y,q,chi : omega0(y+K/sqrt(q*sig**2))/(1-gam*sig**2*chi)
        op3 = lambda y,q,chi : sig**2 *q * omega2(y+K/sqrt(q*sig**2))/(1-gam*sig**2*chi)**2
                
        q0 = 1
        chi0 = 1
        y0 = 1

        if verbose:
            for i in range(n_iter):
                y0 = y0*(1-r) + r * op1(y0,q0,chi0)
                chi0 = chi0*(1-r) + r * op2(y0,q0,chi0)
                q0 = q0*(1-r) + r * op3(y0,q0,chi0)
            print("Errors are (%f , %f , %f)"%(y0-op1(y0,q0,chi0),chi0-op2(y0,q0,chi0),q0-op3(y0,q0,chi0)))
        else:
            for i in range(n_iter):
                y0 = y0*(1-r) + r * op1(y0,q0,chi0)
                chi0 = chi0*(1-r) + r * op2(y0,q0,chi0)
                q0 = q0*(1-r) + r * op3(y0,q0,chi0)
        
        
        d_mean = (K + sig*sqrt(q0)*y0)/(1-gam*sig**2 * chi0)
        d_std = sig*sqrt(q0)/(1-gam*sig**2 * chi0)
        
        self._SAD = lambda x : exp(-.5*(x-d_mean)**2 / d_std**2) / sqrt(2*pi * d_std**2)
        self._SAD_bounds = (0,(4 + np.max(y0) - K/sqrt(q0*sig**2)) * sqrt(q0*sig**2) /(1-gam*sig**2*chi0))
        self._self_avg_quantities['q']= q0
        self._self_avg_quantities['chi']= chi0
        self._self_avg_quantities['y'] =y0
        