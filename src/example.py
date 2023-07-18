from .base import DynamicModel
from .orthogonal_dmft import Eqdmft_orthogonal
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from ecosim.plot import lineplot
from scipy import stats


S = 1000
T=25
dt=.1
gam=-1
mul=.5
spectrum = mul*stats.beta.rvs(20,1,size=S//2)
spectrum = np.concatenate([-spectrum,spectrum])

dyn = DynamicModel(S)
dyn.set_current_state(np.ones(S))
dyn.set_random_component('orthogonally_invariant',spectrum=spectrum,gam=gam)
traj = dyn.run(T,dt)

sol = Eqdmft_orthogonal(S)
sol.spectrum = spectrum
sol.gam = gam
sol.lambda_max=mul
sol.lambda_min=-mul
sol.full_solve()


x = np.linspace(0,np.max(traj['last']),1000)
y = list(map(sol.SAD,x))

plt.hist(traj['last'],density=True,bins=15)
plt.plot(x,y)
plt.show()