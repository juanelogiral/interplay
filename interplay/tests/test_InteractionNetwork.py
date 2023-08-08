import unittest
import ecosim
from interplay.base import InteractionNetwork,Function
import numpy as np

class TestInteractionNetwork(unittest.TestCase):
    
    def test_init(self):
        ic = InteractionNetwork(100)
        
    def test_add_function(self):
        ic = InteractionNetwork(100)
        f = Function.mean_biomass(100)
        ic.add_function(f)
        self.assertTrue(len(ic._functions)==1 and ic._functions[0]==f)
        
        self.assertRaises(TypeError, ic.add_function,1)