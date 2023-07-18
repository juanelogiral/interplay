from .base import InteractionNetwork
from .utils import Interpolator
from .dmft import EqdmftModel
import numpy as np
from scipy.special import erf
from math import sqrt,exp,pi 
from scipy.integrate import quad

''' A specific class for solving dmft models where the interaction matrix reads a_ij = z_ij for z_ij orthogonally invariant
variables

ATTENTION: This method is valid if the full interaction matrix (with self regulation) is Id + Z, however in simulation 
we generally set the diagonal to z_ii = 0. This doesn't have an impact as long as the spectrum of Z is centered, otherwise
the offset in self-interaction z_ii is O(1) and cannot be neglected. Therefore this solver will be accurate for simulations
in which z_ii=0 as long as mean(spectrum)=0.
'''

class Eqdmft_orthogonal(EqdmftModel):
    
    def __init__(self,S):
        super().__init__(S)
        self._S = S
        self._interaction_network = InteractionNetwork(S)
        self._interaction_network.set_random_component('orthogonally_invariant',spectrum=np.linspace(0,1,S))
        self._lambda_max = None #A theoretical estimate of the bound of the spectrum is useful to estimate the range of the Stieltjes
        self._lambda_min = None #In case the Stieltjes is provided as a callable
        self._spectrum = None
        self._gam = 1 #The calculations only work for gam=1 but we extrapolate to gam < 1 and compare to numerics
    
    @property
    def gam(self):
        return self._gam
    @gam.setter
    def gam(self,val):
        self._gam=val
    
    @property
    def spectrum(self):
        return self._spectrum
    @spectrum.setter
    def spectrum(self,val):
        '''
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

        '''
        if len(val) == 3 and callable(val[0]):
            self._spectrum = val[0]
            #InteractionModel 'orthogonally_invariant' admits a callable as spectrum, but it has to be the inv CDF, not the PDF
            #We build the CDF as an interpolation
            self._lambda_min = val[1]
            self._lambda_max = val[2]
            
            CDF = Interpolator(val[0], (val[1],val[2]), .01).integrate()
            PPF =CDF.inverse()
            self._interaction_network.spectrum=PPF
        elif '__iter__' in dir(val):
            self._spectrum = np.sort(val)
            self._interaction_network.spectrum=np.sort(val)
        else:
            raise ValueError("Unrecognized format. Provide (callable,numeric,numeric) or iterable")

    @property
    def lambda_max(self):
        return self._lambda_max
    @lambda_max.setter
    def lambda_max(self,val):
        self._lambda_max = val

    @property
    def lambda_min(self):
        return self._lambda_min
    @lambda_min.setter
    def lambda_min(self,val):
        self._lambda_min = val


    @property
    def S(self):
        return self._S

    @property
    def interaction_network(self):
        return self._interaction_network
    @interaction_network.setter
    def interaction_network(self,val):
        val.set_random_component('orthogonal')
        self._interaction_network = val
    
    def full_solve(self,verbose = True,n_iter=200,r=.25):
        
        #Tabulate the R transform of the eigenvalue distribution and its derivative
        print("Interpolating...")
        D = self._spectrum
        
        
        #When discretizing, the Stieltjes transform cannot be computed raw since g(lambda_max) yields inf even in cases where
        #the Stieltjes is well defined at lambda_max. To go around this we systematically erase the last eigenvalue. Assuming
        #that the distribution has no atoms this doesn't change the result and gives a good estimate of the real range of the
        #Stieltjes.
        
        if self._lambda_max == None:
            if callable(D):
                raise ValueError("If spectrum is callable, please provide lambda_max.")
            lmax = D[-1]
        else:
            lmax= self._lambda_max
        
        if callable(D) and self._lambda_min==None:
            raise ValueError("If spectrum is callable, please provide lambda_min.")
        
        if callable(D):
            eps = .0000001 #in case the Stieltjes is divergent at lambda_max
            g_tr = Interpolator(lambda x : quad(lambda y: D(y)/(x-y),self._lambda_min,self._lambda_max,points=[x])[0],(lmax+eps,lmax+100),.01)
            dg_tr = g_tr.differentiate()
        else:
            g_tr = Interpolator(lambda x : np.mean(1/(x-D[:-1])),(lmax,lmax+100),.01)
            dg_tr = Interpolator(lambda x : -np.mean(1/(x-D[:-1])**2),(lmax,lmax+100),.01)

        R_tr = g_tr.inverse() - (lambda x : 1/x)
        dR_tr = 1 / (dg_tr @ g_tr.inverse()) + (lambda x : 1/x**2)

        omega0 = lambda delta : (1+erf(delta/sqrt(2)))/2
        omega2 = lambda delta : omega0(delta) * (1+delta**2) + delta* exp(-delta**2 / 2) / sqrt(2*pi)

        op1 = lambda q,dz,z: self._gam*(R_tr(omega0(1/sqrt(z))/(1-dz)))
        op2 = lambda q,dz,z : q*dR_tr(omega0(1/sqrt(z))/(1-dz))
        op3 = lambda q,dz,z : z*omega2(1/sqrt(z))/(1-dz)**2
        
        q0 = 1
        dz0 = .5
        z0 = 1
        
        if verbose:
            for i in range(n_iter):
                dz0 = dz0*(1-r) + r * op1(q0,dz0,z0)
                z0= z0*(1-r) + r * op2(q0,dz0,z0)
                q0 = q0*(1-r) + r * op3(q0,dz0,z0)
            print("\n Errors are (%f , %f , %f)"%(np.linalg.norm(dz0-op1(q0,dz0,z0)),z0-op2(q0,dz0,z0),q0-op3(q0,dz0,z0)))
        else:
            for i in range(n_iter):
                dz0 = dz0*(1-r) + r * op1(q0,dz0,z0)
                z0= z0*(1-r) + r * op2(q0,dz0,z0)
                q0 = q0*(1-r) + r * op3(q0,dz0,z0)
        
        
        d_mean =1/(1-dz0)
        d_std = sqrt(z0)/(1-dz0)
        
        self._SAD = lambda x : exp(-.5*(x-d_mean)**2 /d_std**2)/sqrt(2*pi*d_std**2)
        self._SAD_bounds = (0,(1+5*z0)/(1-dz0))
        self._self_avg_quantities['q']= q0
        self._self_avg_quantities['dz']= dz0
        self._self_avg_quantities['z']= z0
        