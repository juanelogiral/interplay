from math import exp, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

from .base import InteractionNetwork
from .dmft import EqdmftModel
from .utils import Interpolator, KL_divergence, gaussian_KL_divergence,cdf_distance
from scipy.stats import norm
from scipy.optimize import fsolve

""" A specific class for solving dmft models where the interaction matrix reads a_ij = z_ij for z_ij orthogonally invariant
variables

ATTENTION: This method is valid if the full interaction matrix (with self regulation) is Id + Z, however in simulation 
we generally set the diagonal to z_ii = 0. This doesn't have an impact as long as the spectrum of Z is centered, otherwise
the offset in self-interaction z_ii is O(1) and cannot be neglected. Therefore this solver will be accurate for simulations
in which z_ii=0 as long as mean(spectrum)=0.
"""


class Eqdmft_orthogonal(EqdmftModel):
    def __init__(self, S):
        super().__init__(S)
        self._S = S
        self._interaction_network = InteractionNetwork(S)
        self._interaction_network.set_random_component(
            "orthogonally_invariant", spectrum=np.linspace(0, 1, S)
        )
        self._lambda_max = None  # A theoretical estimate of the bound of the spectrum is useful to estimate the range of the Stieltjes
        self._lambda_min = None  # In case the Stieltjes is provided as a callable
        self._spectrum = None
        self._gam = 1  # The calculations only work for gam=1 but we extrapolate to gam < 1 and compare to numerics

    @property
    def gam(self):
        return self._gam

    @gam.setter
    def gam(self, val):
        self._gam = val

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, val):
        """
        The spectrum to be used by the solver.

        Parameters
        ----------
        val : TYPE
            Either a sequence of length S representing the deterministic eigenvalues of the matrix or a function giving
            the pdf of the probability distribution. In the latter case, S eigenvalues will be sampled from that distribution.

        Raises
        ------
        ValueError
            If the provided spectrum is neither a sequence of eigenvalues nor a callable representing the pdf of the eigenvalue
            distribution.

        Returns
        -------
        None.

        """
        if len(val) == 3 and callable(val[0]):
            self._spectrum = val[0]
            # InteractionModel 'orthogonally_invariant' admits a callable as spectrum, but it has to be the inv CDF, not the PDF
            # We build the CDF as an interpolation
            self._lambda_min = val[1]
            self._lambda_max = val[2]

            CDF = Interpolator(val[0], (val[1], val[2]), 0.01).integrate()
            PPF = CDF.inverse()
            self._interaction_network.spectrum = PPF
        elif "__iter__" in dir(val):
            self._spectrum = np.sort(val)
            self._interaction_network.spectrum = np.sort(val)
        else:
            raise ValueError(
                "Unrecognized format. Provide (callable,numeric,numeric) or iterable"
            )

    @property
    def lambda_max(self):
        return self._lambda_max

    @lambda_max.setter
    def lambda_max(self, val):
        self._lambda_max = val

    @property
    def lambda_min(self):
        return self._lambda_min

    @lambda_min.setter
    def lambda_min(self, val):
        self._lambda_min = val

    @property
    def S(self):
        return self._S

    @property
    def interaction_network(self):
        return self._interaction_network

    @interaction_network.setter
    def interaction_network(self, val):
        val.set_random_component("orthogonal")
        self._interaction_network = val

    def full_solve(self, verbose=True, n_iter=200, r=0.25):
        # Tabulate the R transform of the eigenvalue distribution and its derivative
        print("Interpolating...")
        D = self._spectrum

        # When discretizing, the Stieltjes transform cannot be computed raw since g(lambda_max) yields inf even in cases where
        # the Stieltjes is well defined at lambda_max. To go around this we systematically erase the last eigenvalue. Assuming
        # that the distribution has no atoms this doesn't change the result and gives a good estimate of the real range of the
        # Stieltjes.

        if self._lambda_max == None:
            if callable(D):
                raise ValueError("If spectrum is callable, please provide lambda_max.")
            lmax = D[-1]
        else:
            lmax = self._lambda_max

        if callable(D) and self._lambda_min == None:
            raise ValueError("If spectrum is callable, please provide lambda_min.")

        if callable(D):
            eps = 0.0000001  # in case the Stieltjes is divergent at lambda_max
            g_tr = Interpolator(
                lambda x: quad(
                    lambda y: D(y) / (x - y),
                    self._lambda_min,
                    self._lambda_max,
                    points=[x],
                )[0],
                (lmax + eps, lmax + 100),
                0.01,
            )
            dg_tr = g_tr.differentiate()
        else:
            g_tr = Interpolator(
                lambda x: np.mean(1 / (x - D[:-1])), (lmax, lmax + 100), 0.01
            )
            dg_tr = Interpolator(
                lambda x: -np.mean(1 / (x - D[:-1]) ** 2), (lmax, lmax + 100), 0.01
            )

        R_tr = g_tr.inverse() - (lambda x: 1 / x)
        dR_tr = 1 / (dg_tr @ g_tr.inverse()) + (lambda x: 1 / x**2)

        self._interaction_network.sample_matrix()
        mu = self._interaction_network.structure_component
        S = mu.shape[0]
        # We compute the eigenvalues and eigenvectors of the interaction matrix
        # and filter them
        thr = 1e-3
        u,eig,v = np.linalg.svd(mu)
        n_eig = len(eig[eig>thr])
        u = u[:,:n_eig] * eig[:n_eig] * sqrt(S)
        v = v[:n_eig,:] / sqrt(S)
        print("Found {n_eig} eigenvectors".format(n_eig=n_eig))
        
        omega0 = lambda delta: (1 + erf(delta / sqrt(2))) / 2
        omega0 = np.vectorize(omega0)
        
        delta = lambda beta,m,z : (1 + m @ u[beta,:])/np.sqrt(z)

        op1 = lambda q, dz, z, o0: R_tr(np.mean(o0) / (1 - dz))
        op2 = lambda q, dz, z, o0: q * dR_tr(np.mean(o0) / (1 - dz))
        op3 = lambda q, dz, z, o2: z * np.mean(o2) / (1 - dz) ** 2
        op4 = lambda q, dz, z ,o1: np.sqrt(z) *np.array([np.sum(v[i,:] * o1) for i in range(n_eig)])/(1-dz)

        q0 = 1
        dz0 = 0.5
        z0 = 1
        m0 = np.random.uniform(-1,1,n_eig)

        if verbose:
            for i in range(n_iter):
                delta_vec = np.array([delta(beta,m0,z0) for beta in range(S)])
                o0 = omega0(delta_vec)
                o1 = delta_vec * o0 + np.exp(-(delta_vec**2) / 2) / sqrt(2 * pi)
                o2 = o0 * (1 + delta_vec**2) + delta_vec * np.exp( -(delta_vec**2) / 2) / sqrt(2 * pi)
                
                dz0 = dz0 * (1 - r) + r * op1(q0, dz0, z0,o0)
                z0 = z0 * (1 - r) + r * op2(q0, dz0, z0,o0)
                q0 = q0 * (1 - r) + r * op3(q0, dz0, z0,o2)
                m0 = m0 * (1-r) + r* op4(q0,dz0,z0,o1)
                print(
                "\n Iteration %i : (%f , %f , %f, %f)"
                % (
                    i,
                    np.linalg.norm(m0 - op4(q0, dz0, z0,o1)),
                    dz0 - op1(q0, dz0, z0,o0),
                    z0 - op2(q0, dz0, z0,o0),
                    q0 - op3(q0, dz0, z0,o2),
                ))
            
            delta_vec = np.array([delta(beta,m0,z0) for beta in range(S)])
            o0 = omega0(delta_vec)
            o1 = delta_vec * o0 + np.exp(-(delta_vec**2) / 2) / sqrt(2 * pi)
            o2 = o0 * (1 + delta_vec**2) + delta_vec * np.exp( -(delta_vec**2) / 2) / sqrt(2 * pi)
                
            print(
                "\n Errors are (%f , %f , %f, %f)"
                % (
                    np.linalg.norm(m0 - op4(q0, dz0, z0,o1)),
                    dz0 - op1(q0, dz0, z0,o0),
                    z0 - op2(q0, dz0, z0,o0),
                    q0 - op3(q0, dz0, z0,o2),
                )
            )
        else:
            for i in range(n_iter):
                delta_vec = np.array([delta(beta,m0,z0) for beta in range(S)])
                o0 = omega0(delta_vec)
                o1 = delta_vec * o0 + np.exp(-(delta_vec**2) / 2) / sqrt(2 * pi)
                o2 = o0 * (1 + delta_vec**2) + delta_vec * np.exp( -(delta_vec**2) / 2) / sqrt(2 * pi)
                
                dz0 = dz0 * (1 - r) + r * op1(q0, dz0, z0,o0)
                z0 = z0 * (1 - r) + r * op2(q0, dz0, z0,o0)
                q0 = q0 * (1 - r) + r * op3(q0, dz0, z0,o2)
                m0 = m0 * (1-r) + r* op4(q0,dz0,z0,o1)

        d_means = sqrt(z0) * np.array([delta(beta,m0,z0) for beta in range(S)]) / (1 - dz0)
        d_std = sqrt(z0) / (1 - dz0)

        self._SAD = lambda x : np.mean([np.exp(-(x-d_means)**2 / (2 * d_std**2))])/sqrt(2 * pi * d_std**2)
        self._SAD_bounds = (
            0,
            np.max(d_means) + 4 * d_std,
        )
        
        self._self_avg_quantities["q"] = q0
        self._self_avg_quantities["dz"] = dz0
        self._self_avg_quantities["z"] = z0
        self._self_avg_quantities["m"] = m0
        
    @classmethod
    def test_data(self, data,spectrum,r=.25,n_iter=200,**kwargs):
        '''Spectrum should be provided as list, not callable
        '''
        
        abundance_thr = 1e-3
        extant = np.where(data >= abundance_thr)
        
        # A solver that given spectrum and q computes z and dz and returns the expected std

        def orthogonal_std(_spectrum,*args):
            if not len(args):
                lmax = _spectrum[-1]
            else:
                lmax = args[0]
            
            g_tr = Interpolator(
                lambda x: np.mean(1 / (x - _spectrum[:-1])), (lmax, lmax + 100), 0.01)
            dg_tr = Interpolator(
                lambda x: -np.mean(1 / (x - _spectrum[:-1]) ** 2), (lmax, lmax + 100), 0.01)

            R_tr = g_tr.inverse() - (lambda x: 1 / x)
            dR_tr = 1 / (dg_tr @ g_tr.inverse()) + (lambda x: 1 / x**2)
            
            #Using the empirical value for q, solve for dz and z
            q = np.mean(data**2)
            
            omega0 = lambda delta: (1 + erf(delta / sqrt(2))) / 2
            op1 = lambda dz, z: self._gam * (R_tr(omega0(1 / sqrt(z)) / (1 - dz)))
            op2 = lambda dz, z: q * dR_tr(omega0(1 / sqrt(z)) / (1 - dz))

            dz0 = 0.5
            z0 = 1
            for i in range(n_iter):
                dz0 = dz0 * (1 - r) + r * op1(q, dz0, z0)
                z0 = z0 * (1 - r) + r * op2(q, dz0, z0)
        
        
        Z = (data[extant] - 1) / orthogonal_std(spectrum,**kwargs) #under the null model Z is a centered unit gaussian
        KL_div = KL_divergence(Z, norm.pdf)
        
        #Now check how far parameters should be from (mu,sig,gam) to get a KL_divergence as high as the one we got and use that
        #as a measure of distance. Right now we do this with gradient descent, but stochastic search might give a different answer
        
        def _minimizer(_spectrum,*args):
            return gaussian_KL_divergence((1,orthogonal_std(_spectrum,*args)),(1,orthogonal_std(spectrum,*args)))
        
        out = fsolve(lambda _ : _minimizer,spectrum,full_output=True,args=(kwargs['lambda_max'],) if 'lambda_max' in kwargs else ())
        x,status = out[0],out[2]
        
        if status != 1:
            return np.inf
        else:
            #the distance is defined as the supremum of the difference between the CDF of the two distributions
            return cdf_distance(x,spectrum)
        

