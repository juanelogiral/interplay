import unittest
import ecosim
from interplay.base import InteractionNetwork,DynamicModel,Function
import numpy as np

class TestDynamicModel(unittest.TestCase):
    
    def test_init_and_set_interaction_network(self):
        dyn = DynamicModel(100)
        i_net = InteractionNetwork(100)
        
    def test_returns_functions_from_trajectory(self):
        dyn = DynamicModel(100)
        i_net = InteractionNetwork(100)
        i_net.add_function(Function.mean_biomass(100))
        traj = dyn.run(100,1)
        
        traj_fun = dyn.functions_from_trajectory(traj)
        
        self.assertTrue(np.array_equal(traj_fun.time_points,traj.time_points))
        