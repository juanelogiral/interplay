""" Some miscellaneous code that is useful at some places
"""

from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
from math import log
from scipy.integrate import cumtrapz
import sys

def KL_divergence(model, data):
    """Given a set of empirical data, computes the KL divergence between the model (given as a PDF) and the data (given as a set of values)
    """
    counts,edges = np.histogram(data, density=True)
    model_counts = np.array([model((edges[i]+edges[i+1])/2) for i in len(edges)])
    return np.sum(counts*np.log(counts/model_counts))

def gaussian_KL_divergence(model,data):
    mu1,sig1 = model
    mu0,sig0 = data
    return log(sig1/sig0) + .5*(sig0**2 + (mu1-mu0)**2)/sig1**2 - .5

def cdf_distance(data1,data2):
    """Given two sets of data, computes the supremum norm between their cdfs
    """
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    i,j=0,0
    maxi=0
    while i<len(data1) and j<len(data2):
        if data1[i]<data2[j]:
            i+=1
            maxi = max(maxi,abs(i/len(data2) - j/len(data1)))
        elif data1[i]>data2[j]:
            j+=1
            maxi = max(maxi,abs(i/len(data1) - j/len(data2)))
        else:
            i+=1
            j+=1
    return maxi

""" Given a function, this class tabulates its values and then defines an interpolated version of the original function
    This is intended to speed-up evaluation of complicated functions, such as those defined by integrals
"""

class GeneralInterpolator:
    def __init__(self):
        self._min_bound = None
        self._max_bound = None
        self._vec_x = None
        self._vec_y = None
        self._domain = None

    def _interpolate(self, x):
        def take_closest(myList, myNumber):
            ## https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
            """
            Assumes myList is sorted. Returns closest index to myNumber from below
            """
            if myNumber < myList[0] or myNumber > myList[-1]:
                raise ValueError(
                    f"{myNumber} is outside the range of the interpolator {(myList[0],myList[-1])}"
                )
            pos = bisect_left(myList, myNumber)
            if pos == 0:  # this can only happen if myNumber == myList[0]
                return 0
            return pos - 1

        idx = take_closest(self._vec_x, x)
        return self._vec_y[idx] + (self._vec_y[idx + 1] - self._vec_y[idx]) * (
            x - self._vec_x[idx]
        ) / (self._vec_x[idx + 1] - self._vec_x[idx])

    @property
    def domain(self):
        return self._domain

    @property
    def interpolation_bounds(self):
        return (self._min_bound, self._max_bound)

    def inverse(self):
        """Generates another interpolator with the inverse of the interpolated function.
        This only works if the function is monotonic in the interpolation domain
        """
        seq_type = ""
        if all(x < y for x, y in zip(self._vec_y, self._vec_y[1:])):
            seq_type = "increasing"
        elif all(x > y for x, y in zip(self._vec_y, self._vec_y[1:])):
            seq_type = "decreasing"
        else:
            raise ValueError("Cannot invert a non-monotonic function")

        if seq_type == "increasing":
            return StaticInterpolator(self._vec_y, self._vec_x)
        elif seq_type == "decreasing":
            return StaticInterpolator(np.flip(self._vec_y), np.flip(self._vec_x))

    def differentiate(self):
        """Generates another interpolator with the derivative of the interpolated function."""
        new_vec_y = np.diff(self._vec_y) / np.diff(self._vec_x)
        new_vec_y = np.append(new_vec_y, new_vec_y[-1])
        return StaticInterpolator(self._vec_x, new_vec_y)

    def integrate(self):
        """Generates another interpolator with the primitive of the interpolated function vanishing at the starting point"""
        new_vec_y = cumtrapz(self._vec_y, x=self._vec_x, initial=0)
        return StaticInterpolator(self._vec_x, new_vec_y)

    def __add__(self, addend):
        if isinstance(addend, GeneralInterpolator):
            if (
                self._min_bound != addend._min_bound
                or self._max_bound != addend._max_bound
            ):
                raise ValueError(
                    "Cannot add Interpolators with different interpolation ranges."
                )
            pass
        # TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(
                self._vec_x, self._vec_y + np.vectorize(addend)(self._vec_x)
            )
            return new_interpolator
        elif isinstance(addend, (float, int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y + addend)
        else:
            raise ValueError(
                f"Addend must be GeneralInterpolator or callable, not {type(addend)}."
            )

    def __radd__(self, addend):
        return self.__add__(addend)

    def __sub__(self, addend):
        if isinstance(addend, GeneralInterpolator):
            if (
                self._min_bound != addend._min_bound
                or self._max_bound != addend._max_bound
            ):
                raise ValueError(
                    "Cannot add Interpolators with different interpolation ranges."
                )
            pass
        # TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(
                self._vec_x, self._vec_y - np.vectorize(addend)(self._vec_x)
            )
            return new_interpolator
        elif isinstance(addend, (float, int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y - addend)
        else:
            raise ValueError(
                f"Addend must be GeneralInterpolator or callable, not {type(addend)}."
            )

    def __rsub__(self, addend):
        if isinstance(addend, GeneralInterpolator):
            if (
                self._min_bound != addend._min_bound
                or self._max_bound != addend._max_bound
            ):
                raise ValueError(
                    "Cannot add Interpolators with different interpolation ranges."
                )
            pass
        # TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(
                self._vec_x, np.vectorize(addend)(self._vec_x) - self._vec_y
            )
            return new_interpolator
        elif isinstance(addend, (float, int)):
            new_interpolator = StaticInterpolator(self._vec_x, addend - self._vec_y)
        else:
            raise ValueError(
                f"Addend must be GeneralInterpolator or callable, not {type(addend)}."
            )

    def __mul__(self, addend):
        if isinstance(addend, GeneralInterpolator):
            if (
                self._min_bound != addend._min_bound
                or self._max_bound != addend._max_bound
            ):
                raise ValueError(
                    "Cannot add Interpolators with different interpolation ranges."
                )
            pass
        # TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(
                self._vec_x, self._vec_y * np.vectorize(addend)(self._vec_x)
            )
            return new_interpolator
        elif isinstance(addend, (float, int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y * addend)
            return new_interpolator
        else:
            raise ValueError(
                f"Addend must be GeneralInterpolator, callable or numeric, not {type(addend)}."
            )

    def __rmul__(self, addend):
        return self.__mul__(addend)

    def __truediv__(self, addend):
        if isinstance(addend, GeneralInterpolator):
            if (
                self._min_bound != addend._min_bound
                or self._max_bound != addend._max_bound
            ):
                raise ValueError(
                    "Cannot add Interpolators with different interpolation ranges."
                )
            pass
        # TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(
                self._vec_x, self._vec_y / np.vectorize(addend)(self._vec_x)
            )
            return new_interpolator
        elif isinstance(addend, (float, int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y / addend)
            return new_interpolator
        else:
            raise ValueError(
                f"Addend must be GeneralInterpolator, callable or numeric, not {type(addend)}."
            )

    def __rtruediv__(self, addend):
        if isinstance(addend, GeneralInterpolator):
            if (
                self._min_bound != addend._min_bound
                or self._max_bound != addend._max_bound
            ):
                raise ValueError(
                    "Cannot add Interpolators with different interpolation ranges."
                )
            pass
        # TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(
                self._vec_x, np.vectorize(addend)(self._vec_x) / self._vec_y
            )
            return new_interpolator
        elif isinstance(addend, (float, int)):
            new_interpolator = StaticInterpolator(self._vec_x, addend / self._vec_y)
            return new_interpolator
        else:
            raise ValueError(
                f"Addend must be GeneralInterpolator, callable or numeric, not {type(addend)}."
            )

    def __matmul__(self, inner):
        if isinstance(inner, GeneralInterpolator):
            new_vec_y = list(map(self._interpolate, inner._vec_y))
            return StaticInterpolator(inner._vec_x, new_vec_y)
        else:
            raise ValueError(f"Inner must be GeneralInterpolator, not {type(inner)}.")

    def __rmatmul__(self, inner):
        if callable(inner):
            new_vec_y = list(map(inner, self._vec_y))
            return StaticInterpolator(self._vec_x, new_vec_y)
        else:
            raise ValueError(f"Inner must be callable, not {type(inner)}.")

    def plot(self,ax=None,label=None):
        if ax is None:
            ax = plt.gca()
        if label is None:
            ax.plot(self._vec_x, self._vec_y, linewidth=1.5)
        else:
            ax.plot(self._vec_x, self._vec_y, linewidth=1.5, label=label)
        return ax


class StaticInterpolator(GeneralInterpolator):
    def __init__(self, vec_x, vec_y):
        if len(vec_x) != len(vec_y):
            raise ValueError(
                f"Impossible to match sequence of len {len(vec_x)} with sequence of len {len(vec_y)}."
            )

        self._vec_x = np.array(vec_x)
        self._vec_y = np.array(vec_y)
        self._domain = (np.min(vec_x), np.max(vec_x))
        self._min_bound, self._max_bound = self._domain

    def __call__(self, x):
        if "__iter__" in dir(x):
            x = np.array(x)
            if np.any(x > self._domain[1]) or np.any(x < self._domain[0]):
                raise ValueError(
                    f"{x} is outside the domain of the function {self._domain}."
                )
            print(x)
            return list(map(super()._interpolate, x))
        else:
            if x > self._domain[1] or x < self._domain[0]:
                raise ValueError(
                    f"{x} is outside the domain of the function {self._domain}."
                )

            return super()._interpolate(x)


class Interpolator(GeneralInterpolator):
    def __init__(self, fun, bounds, max_dx, domain=(-np.inf, np.inf)):
        if not callable(fun):
            raise TypeError(f"fun must be a callable object, {type(fun)} is not.")

        self._fun = np.vectorize(fun)
        self._domain = domain
        self._vec_x = self._create_range(bounds, max_dx)
        self._vec_y = self._fun(self._vec_x)
        self._min_bound = bounds[0]
        self._max_bound = bounds[1]

    def _create_range(self, bounds, max_dx):
        if bounds == None:
            return []
        # for now we create a uniform range but this could be optimized
        new_range = np.arange(bounds[0], bounds[1] + max_dx, max_dx)
        return new_range

    def expand(self, bounds, max_dx):
        expand_up = (
            (self._max_bound, bounds[1]) if bounds[1] > self._max_bound else None
        )
        expand_down = (
            (bounds[0], self._min_bound) if bounds[0] < self._min_bound else None
        )

        range_up = self._create_range(expand_up, max_dx)
        range_down = self._create_range(expand_down, max_dx)

        # Generate a time array
        self._vec_x = np.concatenate([range_down, self._vec_x, range_up])
        # Sample values
        self._vec_y = np.concatenate(
            [self._fun(range_down), self._vec_y, self._fun(range_up)]
        )

        self._min_bound = np.min(self._vec_x)
        self._max_bound = np.max(self._vec_x)

    def __call__(self, x):
        if x > self._domain[1] or x < self._domain[0]:
            raise ValueError(
                f"{x} is outside the domain of the function {self._domain}."
            )

        if x > self._max_bound:
            self.expand(self._min_bound, x)
        elif x < self._min_bound:
            self.expand(x, self._max_bound)

        return super()._interpolate(x)
    
## Fast Fourier transforms
# Codes adapted from Emil Mallmin

def fft(mat, axis=0, target='cpu'):
    '''
    Calculate the FFT of a matrix along a given axis on cpu or gpu.

    Parameters
    ----------
    mat : ndarray
        matrix over which to compute FFT
    axis : int
        axis along which to compute FFT
    target : {'cpu', 'gpu'}
        platform to run FFT on

    Returns
    -------
    mat_fft : ndarray
        transformed array, where the ``axis`` dimension indexes frequency in range ``0:mat.shape[axis]//2+1``
    '''
    
    if target == 'gpu':
        if 'cupy' not in sys.modules:
             import cupy
        xp = sys.modules['cupy']
        arr = xp.asarray(arr, dtype='float32')
    elif target == 'cpu':
        xp = sys.modules['numpy']
    else:
        raise ValueError('target must be either cpu or gpu')
    
    return xp.fft.rfft(mat,axis=axis)


def autocorr_fft(traj,species='all', lags=200, standardize=True, real_input=True, target='cpu'):
    '''
    Calculate the autocorrelation function using FFT on cpu or gpu.

    Parameters
    ----------
    arr : Trajectory
        trajectory over which to compute autocorrelation. Trajectory anchors will be used as the discrete time points.
    species : string or np.array slicer
        species for which autocorrelation is to be computed. Either 'all' or some type able to slice an np.array
    lags : int
        number of lags to include (size of return vector)
    standardize : bool
        to subtract mean and normalize by variance
    target : {'cpu', 'gpu'}
        platform to run FFT on

    Returns
    -------
    vec_acf : ndarray
        transformed array, where the ``axis`` dimension indexes time lags in range ``0:lags``    
    '''
    
    # construct array from trajectory
    if species == 'all':
        sp_idx = slice(None)
    else:
        sp_idx=species
    arr = traj.anchors[1][:,sp_idx]
    
    axis = 0 #correlation computed over first axis
    
    # Zero padding for FFT
    pow2 = int(2**np.ceil(np.log2(arr.shape[axis])))

    if standardize:
        arr = arr - np.mean(arr,axis=axis)

    
    if target == 'gpu':
        if 'cupy' not in sys.modules:
             import cupy
        xp = sys.modules['cupy']
        arr = xp.asarray(arr, dtype='float32')
    else:
        xp = sys.modules['numpy']

    fft = xp.fft.rfft if real_input else xp.fft.fft
    ifft = xp.fft.irfft if real_input else xp.fft.ifft

    arr_fft = fft(arr, axis=axis, n=pow2)
    arr_acf = ifft( arr_fft * xp.conjugate(arr_fft), axis=axis).real / pow2
    
    # Normalize
    if standardize:
        ax = arr.ndim - 1 if axis == -1 else axis
        ids = (slice(None),)*ax + (0,) + (slice(None),)*(arr.ndim - ax - 1)
        arr_acf = arr_acf[:lags] / arr_acf[ids]
   
    if target == 'gpu':
        arr_acf = xp.asnumpy(arr_acf)

    return arr_acf



def crosscorr_fft(traj, species, lags=200, standardize=True, real_input=True, target='cpu'):
    '''
    Compute all crosscorrelation functions between species

    Parameters
    ----------
    traj : Trajectory
        trajectory over which to compute autocorrelation. Trajectory anchors will be used as the discrete time points.
    species : 'all' or np.array slicer
        species for which autocorrelation is to be computed. Either 'all' or some type able to slice an np.array    
    lags : int
        number of lags to include (size of return vector)
    standardize : bool
        to subtract mean and normalize by variance
    target : {'cpu', 'gpu'}
        platform to run FFT on
        
    Returns
    -------
    ten_ccf : 3darray
        transformed array, where the dimension 0 indexes time lags in range ``0:lags``,
        and dimension 1,2 the indices of columns of the input
    '''

    if species == 'all':
        sp_idx = slice(None)
    else:
        sp_idx=species

    mat = traj.anchor[1][:,sp_idx]
    
    # Zero padding for FFT
    pow2 = int(2**np.ceil(np.log2(mat.shape[0])))
    S = mat.shape[1]  
   
    if standardize:
        mat = mat - np.mean(mat,axis=0)
  
    if target == 'cpu':
        dtype = np.float64
        xp = sys.modules['numpy']        
    elif target=='gpu':
        if 'cupy' not in sys.modules:
            import cupy
        dtype = np.float32
        xp = sys.modules['cupy']
        mat = xp.asarray(mat, dtype='float32')
    
    ten_ccf = xp.zeros((lags,S,S), dtype=dtype)

    fft = xp.fft.rfft if real_input else xp.fft.fft
    ifft = xp.fft.irfft if real_input else xp.fft.ifft

    mat_fft = fft(mat, axis=0, n=pow2)

    for i in range(S):

        vec_fft_i = xp.conjugate(mat_fft[:,i])[:,None]
        
        #column j now represents the (i,i+j) correlator
        mat_ccf = ifft(mat_fft[:,i:] * vec_fft_i, axis=0).real / pow2
                
        ten_ccf[:,i,i:] = mat_ccf[:lags,:]
        

    if standardize:
        vec_std = xp.diagonal(ten_ccf[0,:,:])
        mat_var = xp.sqrt(xp.outer(vec_std, vec_std))

        # could give silent division by zero => nan
        ten_ccf /= mat_var

    # fill in lower triangle
    ten_ccf += xp.transpose(ten_ccf, axes=(0,2,1)) #will double the diagonal
    
    # Halve the diagonal. Maybe a smarter way exists?
    mat_halve_diag = xp.ones((S,S),dtype=dtype)
    xp.fill_diagonal(mat_halve_diag, 0.5)
    ten_ccf *= mat_halve_diag

    if target == 'gpu':
        ten_ccf = xp.asnumpy(ten_ccf)

    return ten_ccf

