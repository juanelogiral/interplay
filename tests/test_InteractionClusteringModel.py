import unittest
import ecosim
from interplay.base import InteractionNetwork
from interplay.cluster import InteractionClusteringModel
import interplay.cluster_methods
import mock

class TestInteractionClusteringModel(unittest.TestCase):
    
    def test_methods_loaded_on_class_definition(self):
        
        # Create mock objects for randmat functions and a mock default arguments
        patch_funs = {'dummyfun_1':mock.DEFAULT,'dummyfun_2':mock.DEFAULT,'dummyfun_3':mock.DEFAULT}
        for fun in patch_funs:
            setattr(interplay.cluster_methods,fun,lambda x : x)  
    
        # In case glv.RandomLotkaVolterra has already been defined, by undo the definition
        if 'InteractionClusteringModel' in dir(interplay.cluster):
            icm_copy = interplay.cluster.InteractionClusteringModel.copy()
            del interplay.cluster.InteractionClusteringModel
       
        # This redefines the class
        interplay.cluster.InteractionClusteringModel = interplay.cluster.icm_metaclass('InteractionClusteringModel',(ecosim.base.Storable,),{})
        
        # If properly working, RandomLotkaVolterra now has static variables for the available models and these
        # variables reference the mock functions created at the beginning
        
        self.assertTrue('_available_clustering_functions' in dir(interplay.cluster.InteractionClusteringModel))

        self.assertTrue(set(interplay.cluster.InteractionClusteringModel._available_clustering_functions.keys()).issuperset(set(patch_funs.keys())))
        
       
        for fun in patch_funs:
            delattr(interplay.cluster_methods,fun)
            
        del interplay.cluster.InteractionClusteringModel
        interplay.cluster.InteractionClusteringModel = icm_copy

    def test_init(self):
        icm = InteractionClusteringModel()
        
        self.assertEqual(icm.clustering_method, 'spectral')
        self.assertEqual(icm.optimization_method, 'metropolis')