''' Some miscellaneous code that is useful at some places
'''

import numpy as np
from bisect import bisect_left
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
''' Given a function, this class tabulates its values and then defines an interpolated version of the original function
    This is intended to speed-up evaluation of complicated functions, such as those defined by integrals
'''

class GeneralInterpolator():
    
    def __init__(self):
        
        self._min_bound = None
        self._max_bound = None
        self._vec_x = None
        self._vec_y = None
        self._domain = None
    
    def _interpolate(self,x):
        
        def take_closest(myList, myNumber):
            ## https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
            """
            Assumes myList is sorted. Returns closest index to myNumber from below
            """
            if myNumber < myList[0] or myNumber > myList[-1]:
                raise ValueError(f"{myNumber} is outside the range of the interpolator {(myList[0],myList[-1])}")
            pos = bisect_left(myList, myNumber)
            if pos == 0: # this can only happen if myNumber == myList[0]
                return 0
            return pos-1
        
        idx = take_closest(self._vec_x, x)
        return self._vec_y[idx] + (self._vec_y[idx+1]-self._vec_y[idx]) * (x - self._vec_x[idx])/(self._vec_x[idx+1] - self._vec_x[idx])
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def interpolation_bounds(self):
        return (self._min_bound,self._max_bound)
    
    def inverse(self):
        '''Generates another interpolator with the inverse of the interpolated function.
        This only works if the function is monotonic in the interpolation domain
        '''
        seq_type = ''
        if all(x < y for x, y in zip(self._vec_y, self._vec_y[1:])):
            seq_type = 'increasing'
        elif all(x > y for x, y in zip(self._vec_y, self._vec_y[1:])):
            seq_type = 'decreasing'
        else:
            raise ValueError("Cannot invert a non-monotonic function")
            
        if seq_type == 'increasing':
             return StaticInterpolator(self._vec_y,self._vec_x)
        elif seq_type == 'decreasing':
             return StaticInterpolator(np.flip(self._vec_y),np.flip(self._vec_x))
    
    def differentiate(self):
        '''Generates another interpolator with the derivative of the interpolated function.
        '''
        new_vec_y =np.diff(self._vec_y)/np.diff(self._vec_x)
        new_vec_y = np.append(new_vec_y,new_vec_y[-1])
        return StaticInterpolator(self._vec_x,new_vec_y)
    
    def integrate(self):
        '''Generates another interpolator with the primitive of the interpolated function vanishing at the starting point
        '''
        new_vec_y = cumtrapz(self._vec_y,x=self._vec_x,initial=0)
        return StaticInterpolator(self._vec_x,new_vec_y)
    
    def __add__(self,addend):
        
        if isinstance(addend,GeneralInterpolator):
            
            if self._min_bound != addend._min_bound or self._max_bound != addend._max_bound:
                raise ValueError("Cannot add Interpolators with different interpolation ranges.")
            pass
        #TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y + np.vectorize(addend)(self._vec_x))
            return new_interpolator
        elif isinstance(addend,(float,int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y + addend)
        else:
            raise ValueError(f"Addend must be GeneralInterpolator or callable, not {type(addend)}.")
    
    def __radd__(self,addend):
        return self.__add__(addend)
    
    def __sub__(self,addend):
        
        if isinstance(addend,GeneralInterpolator):
            
            if self._min_bound != addend._min_bound or self._max_bound != addend._max_bound:
                raise ValueError("Cannot add Interpolators with different interpolation ranges.")
            pass
        #TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y - np.vectorize(addend)(self._vec_x))
            return new_interpolator
        elif isinstance(addend,(float,int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y - addend)
        else:
            raise ValueError(f"Addend must be GeneralInterpolator or callable, not {type(addend)}.")
            
    def __rsub__(self,addend):
        
        if isinstance(addend,GeneralInterpolator):
            
            if self._min_bound != addend._min_bound or self._max_bound != addend._max_bound:
                raise ValueError("Cannot add Interpolators with different interpolation ranges.")
            pass
        #TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(self._vec_x, np.vectorize(addend)(self._vec_x)-self._vec_y)
            return new_interpolator
        elif isinstance(addend,(float,int)):
            new_interpolator = StaticInterpolator(self._vec_x, addend - self._vec_y)
        else:
            raise ValueError(f"Addend must be GeneralInterpolator or callable, not {type(addend)}.")
                
    def __mul__(self,addend):
        
        if isinstance(addend,GeneralInterpolator):
            
            if self._min_bound != addend._min_bound or self._max_bound != addend._max_bound:
                raise ValueError("Cannot add Interpolators with different interpolation ranges.")
            pass
        #TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y * np.vectorize(addend)(self._vec_x))
            return new_interpolator
        elif isinstance(addend,(float,int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y * addend)
            return new_interpolator
        else:
            raise ValueError(f"Addend must be GeneralInterpolator, callable or numeric, not {type(addend)}.")
            
    def __rmul__(self,addend):
        return self.__mul__(addend)
    
    def __truediv__(self,addend):
        
        if isinstance(addend,GeneralInterpolator):
            
            if self._min_bound != addend._min_bound or self._max_bound != addend._max_bound:
                raise ValueError("Cannot add Interpolators with different interpolation ranges.")
            pass
        #TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y / np.vectorize(addend)(self._vec_x))
            return new_interpolator
        elif isinstance(addend,(float,int)):
            new_interpolator = StaticInterpolator(self._vec_x, self._vec_y / addend)
            return new_interpolator
        else:
            raise ValueError(f"Addend must be GeneralInterpolator, callable or numeric, not {type(addend)}.")
            
    def __rtruediv__(self,addend):
        
        if isinstance(addend,GeneralInterpolator):
            
            if self._min_bound != addend._min_bound or self._max_bound != addend._max_bound:
                raise ValueError("Cannot add Interpolators with different interpolation ranges.")
            pass
        #TODO
        elif callable(addend):
            new_interpolator = StaticInterpolator(self._vec_x, np.vectorize(addend)(self._vec_x) / self._vec_y)
            return new_interpolator
        elif isinstance(addend,(float,int)):
            new_interpolator = StaticInterpolator(self._vec_x, addend / self._vec_y)
            return new_interpolator
        else:
            raise ValueError(f"Addend must be GeneralInterpolator, callable or numeric, not {type(addend)}.")
    
    def __matmul__(self,inner):            
        
        if isinstance(inner,GeneralInterpolator):
            new_vec_y = list(map(self._interpolate,inner._vec_y))
            return StaticInterpolator(inner._vec_x,new_vec_y)
        else:
            raise ValueError(f"Inner must be GeneralInterpolator, not {type(inner)}.")
        
    def __rmatmul__(self,inner):            
        
        if callable(inner):
            new_vec_y = list(map(inner,self._vec_y))
            return StaticInterpolator(self._vec_x,new_vec_y)
        else:
            raise ValueError(f"Inner must be callable, not {type(inner)}.")

    def plot(self):
        sns.set_style('darkgrid')
        pl = sns.lineplot(self._vec_x,self._vec_y,linewidth=1.5)
        pl.legend([],[], frameon=False)
        plt.show()

class StaticInterpolator(GeneralInterpolator):
    
    def __init__(self,vec_x,vec_y):
        if len(vec_x) != len(vec_y):
            raise ValueError(f"Impossible to match sequence of len {len(vec_x)} with sequence of len {len(vec_y)}.")
            
        self._vec_x = np.array(vec_x)
        self._vec_y = np.array(vec_y)
        self._domain = (np.min(vec_x),np.max(vec_x))
        self._min_bound,self._max_bound = self._domain
        
    def __call__(self,x):
        
        if '__iter__' in dir(x):
            x = np.array(x)
            if np.any(x > self._domain[1]) or np.any(x < self._domain[0]):
                raise ValueError(f"{x} is outside the domain of the function {self._domain}.")
            print(x)
            return list(map(super()._interpolate,x))
        else:
            if x > self._domain[1] or x < self._domain[0]:
                raise ValueError(f"{x} is outside the domain of the function {self._domain}.")
                    
            return super()._interpolate(x)
            
        

class Interpolator(GeneralInterpolator):
    
    def __init__(self,fun,bounds,max_dx,domain=(-np.inf,np.inf)):
        
        if not callable(fun):
            raise TypeError(f"fun must be a callable object, {type(fun)} is not.")
            
        self._fun = np.vectorize(fun)
        self._domain = domain
        self._vec_x = self._create_range(bounds,max_dx)
        self._vec_y = self._fun(self._vec_x)
        self._min_bound = bounds[0]
        self._max_bound = bounds[1]
    
    def _create_range(self,bounds,max_dx):
        if bounds == None:
            return []
        #for now we create a uniform range but this could be optimized
        new_range = np.arange(bounds[0], bounds[1]+max_dx,max_dx)
        return new_range
        
    def expand(self,bounds,max_dx):
        
        expand_up = (self._max_bound, bounds[1]) if bounds[1] > self._max_bound else None
        expand_down = (bounds[0],self._min_bound) if bounds[0] < self._min_bound else None
        
        range_up = self._create_range(expand_up,max_dx)
        range_down = self._create_range(expand_down,max_dx)
        
        #Generate a time array
        self._vec_x = np.concatenate([range_down,self._vec_x,range_up])
        #Sample values
        self._vec_y = np.concatenate([self._fun(range_down),self._vec_y,self._fun(range_up)])
        
        self._min_bound = np.min(self._vec_x)
        self._max_bound = np.max(self._vec_x)
        
    def __call__(self,x):
        
        if x > self._domain[1] or x < self._domain[0]:
            raise ValueError(f"{x} is outside the domain of the function {self._domain}.")
        
        if x > self._max_bound:
            self.expand(self._min_bound,x)
        elif x < self._min_bound:
            self.expand(x,self._max_bound)
        
        return super()._interpolate(x)        