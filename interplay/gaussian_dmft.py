from math import exp, pi, sqrt

import numpy as np
from scipy.special import erf

from .base import InteractionNetwork
from .dmft import EqdmftModel
from .utils import KL_divergence,gaussian_KL_divergence
from scipy.stats import norm
from scipy.optimize import fsolve

""" A specific class for solving dmft models where the interaction matrix reads a_ij = mu_ij + sig * z_ij for z_ij gaussian
variables
"""


class Eqdmft_gaussian(EqdmftModel):
    def __init__(self, S):
        super().__init__(S)
        self._S = S
        self._interaction_network = InteractionNetwork(S)
        self._interaction_network.set_random_component("gaussian")

    @property
    def S(self):
        return self._S

    @property
    def sig(self):
        return self._interaction_network.sig

    @sig.setter
    def sig(self, val):
        self._interaction_network.sig = val

    @property
    def gam(self):
        return self._interaction_network.gam

    @gam.setter
    def gam(self, val):
        self._interaction_network.gam = val

    @property
    def interaction_network(self):
        return self._interaction_network

    @interaction_network.setter
    def interaction_network(self, val):
        val.set_random_component("gaussian")
        self._interaction_network = val

    def _full_solve_species(self, verbose=True, n_iter=200, r=0.25):
        omega0 = lambda delta: (1 + erf(delta / sqrt(2))) / 2
        omega1 = lambda delta: delta * omega0(delta) + exp(-(delta**2) / 2) / sqrt(2 * pi)
        omega2 = lambda delta: omega0(delta) * (1 + delta**2) + delta * exp(-(delta**2) / 2) / sqrt(2 * pi)

        omega0 = np.vectorize(omega0)
        omega1 = np.vectorize(omega1)
        omega2 = np.vectorize(omega2)

        self._interaction_network.sample_matrix()
        mu = self._interaction_network.structure_component

        S = self.S
        K = self.K
        try:
            iter(K)
        except:
            K = K*np.ones(S)
        sig = self.sig
        gam = self.gam

        op1 = lambda y, q, chi: np.array([
                np.sum(mu[i] * omega1(y + K / sqrt(q * sig**2)))/ (1 - gam * sig**2 * chi)
                for i in range(S)
            ])
        op2 = lambda y, q, chi: np.mean(omega0(y + K / sqrt(q * sig**2))) / (1 - gam * sig**2 * chi)
        op3 = lambda y, q, chi: sig**2* q * np.mean(omega2(y + K / sqrt(q * sig**2))) / (1 - gam * sig**2 * chi) ** 2

        q0 = np.random.rand()
        chi0 = np.random.rand()
        y0 = np.random.uniform(-1,1,S)

        if verbose:
            for i in range(n_iter):
                y0 = y0 * (1 - r) + r * op1(y0, q0, chi0)
                chi0 = chi0 * (1 - r) + r * op2(y0, q0, chi0)
                q0 = q0 * (1 - r) + r * op3(y0, q0, chi0)
                print("Iteration %i: (%f,%f,%f)"%(i,
                    np.linalg.norm(m0 - op1(q0, chi0,o1)),
                    chi0 - op2(q0, chi0,o0),
                    q0 - op3(q0, chi0,o2),
                ),end="\r")
            print(
                "Errors are (%f , %f , %f)"
                % (
                    np.linalg.norm(y0 - op1(y0, q0, chi0)),
                    chi0 - op2(y0, q0, chi0),
                    q0 - op3(y0, q0, chi0),
                )
            )
        else:
            for i in range(n_iter):
                y0 = y0 * (1 - r) + r * op1(y0, q0, chi0)
                chi0 = chi0 * (1 - r) + r * op2(y0, q0, chi0)
                q0 = q0 * (1 - r) + r * op3(y0, q0, chi0)

        d_means = (K + sig * sqrt(q0) * y0) / (1 - gam * sig**2 * chi0)
        d_std = sig * sqrt(q0) / (1 - gam * sig**2 * chi0)

        self._SAD = lambda x : np.mean(np.exp(-(x-d_means)**2 / (2 * d_std**2)))/sqrt(2 * pi * d_std**2)
        self._SAD_bounds = (
            0,
            np.max(d_means) + 4 * d_std,
        )
        self._self_avg_quantities["q"] = q0
        self._self_avg_quantities["chi"] = chi0
        self._self_avg_quantities["y"] = y0
        self._self_avg_quantities["phi"] = np.mean(omega0(y0 + K/sqrt(q0*sig**2)))
    
    def _full_solve_eigen(self, verbose=True, n_iter=200, r=0.25):
        omega0 = lambda delta: (1 + erf(delta / sqrt(2))) / 2
        omega0 = np.vectorize(omega0)
    
        self._interaction_network.sample_matrix()
        mu = self._interaction_network.structure_component
        
        S = self.S
        K = self.K
        try:
            iter(K)
        except:
            K = K*np.ones(S)
        sig = self.sig
        gam = self.gam
        
        # We compute the eigenvalues and eigenvectors of the interaction matrix
        # and filter them
        thr = 1e-2
        u,eig,v = np.linalg.svd(mu)
        n_eig = len(eig[eig>thr])
        u = u[:,:n_eig] * eig[:n_eig] * sqrt(S)
        v = v[:n_eig,:] / sqrt(S)
        print("Found {n_eig} eigenvectors".format(n_eig=n_eig))
        #beta is a species index and i is an eigenvalue index
        delta = lambda beta,m,q : (K[beta] + m @ u[beta,:])/sqrt(q*sig**2)
        op1 = lambda q,chi,o1: sqrt(q)*sig *np.array([np.sum(v[i,:] * o1) for i in range(n_eig)])/(1-gam*sig**2*chi)
        op2 = lambda q,chi,o0 : np.mean(o0)/(1-gam*sig**2*chi)
        op3 = lambda q,chi,o2 : sig**2 *q * np.mean(o2)/(1-gam*sig**2*chi)**2
        
        q0 = np.random.rand()
        chi0 = np.random.rand()
        m0 = np.random.uniform(-1,1,n_eig)

        if verbose:
            for i in range(n_iter):
                delta_vec = np.array([delta(beta,m0,q0) for beta in range(S)])
                o0 = omega0(delta_vec)
                o1 = delta_vec * o0 + np.exp(-(delta_vec**2) / 2) / sqrt(2 * pi)
                o2 = o0 * (1 + delta_vec**2) + delta_vec * np.exp( -(delta_vec**2) / 2) / sqrt(2 * pi)
  
                m0 = m0 * (1 - r) + r * op1(q0, chi0,o1)
                chi0 = chi0 * (1 - r) + r * op2(q0, chi0,o0)
                q0 = q0 * (1 - r) + r * op3(q0, chi0,o2)
                print("Iteration %i: (%f,%f,%f)"%(i,
                    np.linalg.norm(m0 - op1(q0, chi0,o1)),
                    chi0 - op2(q0, chi0,o0),
                    q0 - op3(q0, chi0,o2),
                ),end="\r")
             
            delta_vec = np.array([delta(beta,m0,q0) for beta in range(S)])
            o0 = omega0(delta_vec)
            o1 = delta_vec * o0 + np.exp(-(delta_vec**2) / 2) / sqrt(2 * pi)
            o2 = o0 * (1 + delta_vec**2) + delta_vec * np.exp( -(delta_vec**2) / 2) / sqrt(2 * pi)   
            print(
                "Errors are (%f , %f , %f)"
                % (
                    np.linalg.norm(m0 - op1(q0, chi0,o1)),
                    chi0 - op2(q0, chi0,o0),
                    q0 - op3(q0, chi0,o2),
                )
            )
        else:
            for i in range(n_iter):
                delta_vec = np.array([delta(beta,m0,q0) for beta in range(S)])
                o0 = omega0(delta_vec)
                o1 = delta_vec * o0 + np.exp(-(delta_vec**2) / 2) / sqrt(2 * pi)
                o2 = o0 * (1 + delta_vec**2) + delta_vec * np.exp( -(delta_vec**2) / 2) / sqrt(2 * pi)
                
                m0 = m0 * (1 - r) + r * op1(q0, chi0,o1)
                chi0 = chi0 * (1 - r) + r * op2(q0, chi0,o0)
                q0 = q0 * (1 - r) + r * op3(q0, chi0,o2)

        d_means = sqrt(q0)*sig * np.array([delta(beta,m0,q0) for beta in range(S)]) / (1 - gam * sig**2 * chi0)
        d_std = sig * sqrt(q0) / (1 - gam * sig**2 * chi0)

        self._SAD = lambda x : np.mean([np.exp(-(x-d_means)**2 / (2 * d_std**2))])/sqrt(2 * pi * d_std**2)
        self._SAD_bounds = (
            0,
            np.max(d_means) + 4 * d_std,
        )
        self._self_avg_quantities["q"] = q0
        self._self_avg_quantities["chi"] = chi0
        self._self_avg_quantities["m"] = m0
        self._self_avg_quantities["phi"] = np.mean(omega0(delta_vec))

    def full_solve(self, verbose=True, n_iter=200, r=0.25,reduce_dimension=True):
        if not reduce_dimension:
            self._full_solve_species(verbose, n_iter, r)
        else:
            self._full_solve_eigen(verbose, n_iter, r)

    @staticmethod
    def test_data(data,i_network):
        pass

""" A specific class for solving dmft models where the interaction matrix reads a_ij = mu + sig * z_ij for z_ij gaussian
variables
"""

class Eqdmft_uniform_gaussian(EqdmftModel):
    def __init__(self, S):
        super().__init__(S)
        self._S = S
        self._interaction_network = InteractionNetwork(S)
        self._interaction_network.set_random_component("gaussian")

    @property
    def S(self):
        return self._S

    @property
    def sig(self):
        return self._interaction_network.sig

    @sig.setter
    def sig(self, val):
        self._interaction_network.sig = val

    @property
    def gam(self):
        return self._interaction_network.gam

    @gam.setter
    def gam(self, val):
        self._interaction_network.gam = val

    @property
    def mu(self):
        return self._interaction_network.mu

    @mu.setter
    def mu(self, val):
        self._interaction_network.mu = val

    @property
    def interaction_network(self):
        return self._interaction_network

    @interaction_network.setter
    def interaction_network(self, val):
        val.set_random_component("gaussian")
        self._interaction_network = val

    def full_solve(self, verbose=True, n_iter=200, r=0.25):
        omega0 = lambda delta: (1 + erf(delta / sqrt(2))) / 2
        omega1 = lambda delta: delta * omega0(delta) + exp(-(delta**2) / 2) / sqrt( 2 * pi)
        omega2 = lambda delta: omega0(delta) * (1 + delta**2) + delta * exp(-(delta**2) / 2) / sqrt(2 * pi)

        S = self.S
        K = self.K[0]
        sig = self.sig
        mu = self.mu
        gam = self.gam

        op1 = lambda y, q, chi: mu* omega1(y + K / sqrt(q * sig**2))/ (1 - gam * sig**2 * chi)
        op2 = lambda y, q, chi: omega0(y + K / sqrt(q * sig**2)) / (1 - gam * sig**2 * chi)
        op3 = lambda y, q, chi: sig**2* q* omega2(y + K / sqrt(q * sig**2))/ (1 - gam * sig**2 * chi) ** 2

        q0 = 1
        chi0 = 1
        y0 = 1

        if verbose:
            for i in range(n_iter):
                y0 = y0 * (1 - r) + r * op1(y0, q0, chi0)
                chi0 = chi0 * (1 - r) + r * op2(y0, q0, chi0)
                q0 = q0 * (1 - r) + r * op3(y0, q0, chi0)
            print(
                "Errors are (%f , %f , %f)"
                % (
                    y0 - op1(y0, q0, chi0),
                    chi0 - op2(y0, q0, chi0),
                    q0 - op3(y0, q0, chi0),
                )
            )
        else:
            for i in range(n_iter):
                y0 = y0 * (1 - r) + r * op1(y0, q0, chi0)
                chi0 = chi0 * (1 - r) + r * op2(y0, q0, chi0)
                q0 = q0 * (1 - r) + r * op3(y0, q0, chi0)

        d_mean = (K + sig * sqrt(q0) * y0) / (1 - gam * sig**2 * chi0)
        d_std = sig * sqrt(q0) / (1 - gam * sig**2 * chi0)

        self._SAD = lambda x: exp(-0.5 * (x - d_mean) ** 2 / d_std**2) / sqrt(2 * pi * d_std**2)
        self._SAD_bounds = (
            0,
            d_mean + 4 * d_std,
        )
        self._self_avg_quantities["q"] = q0
        self._self_avg_quantities["chi"] = chi0
        self._self_avg_quantities["y"] = y0
    
    @staticmethod
    def test_data(data,mu, sig,gam):
        
        abundance_thr = 1e-3
        extant = np.where(data >= abundance_thr)
        
        gaussian_mean_emp = lambda _mu,_sig,_gam : (1 + _mu * np.mean(data)) / (1 - _gam * _sig**2 * np.mean(data**2))
        gaussian_std_emp = lambda _mu,_sig,_gam : _sig * np.sqrt(np.mean(data**2)) / (1 - _gam * _sig**2 * np.mean(data**2))
        
        Z = (data[extant] - gaussian_mean_emp(mu,sig,gam)) / gaussian_std_emp(mu,sig,gam) #under the null model Z is a centered unit gaussian
        KL_div = KL_divergence(Z, norm.pdf)
        
        #Now check how far parameters should be from (mu,sig,gam) to get a KL_divergence as high as the one we got and use that
        #as a measure of distance. Right now we do this with gradient descent, but stochastic search might give a different answer
        
        out = fsolve(lambda _mu,_sig,_gam : gaussian_KL_divergence(
            (gaussian_mean_emp(mu,sig,gam),gaussian_std_emp(mu,sig,gam)),
            (gaussian_mean_emp(_mu,_sig,_gam),gaussian_std_emp(_mu,_sig,_gam))
            ),(mu,sig,gam),full_output=True)
        
        x,status = out[0],out[2]
        
        if status != 1:
            return np.inf
        else:
            return np.linalg.norm(x/np.array([mu,sig,gam])-1)
        
