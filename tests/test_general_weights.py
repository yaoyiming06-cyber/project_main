import unittest
from cxr_project.general_weights import weight_source

class GeneralWeightTests(unittest.TestCase):
    def test_only_supported_nonmedical_sources(self):
        self.assertEqual(weight_source('vit_base_patch16_224.augreg_in21k')['repo_id'], 'timm/vit_base_patch16_224.augreg_in21k')
        with self.assertRaises(ValueError): weight_source('eva_medical')

if __name__=='__main__': unittest.main()
