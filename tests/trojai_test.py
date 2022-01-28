import unittest
from trojai_submission import data_tools
import os
import datasets

class TestTriggerDataset(unittest.TestCase):

    def setUp(self):
        self.current_script_dirname = os.path.dirname(os.path.abspath(__file__))
        self.model_filepath = os.path.join(self.current_script_dirname, 'data/round8_sample_dataset/models/id-00000000/model.pt')
        self.round_training_dataset_dirpath = os.path.join(self.current_script_dirname, 'data/round8_sample_dataset')
        self.scratch_filepath = '.tmp'

    def test_load_config(self):
        config = data_tools.load_config(self.model_filepath)
        self.assertTrue('model_architecture' in config)
    
    def test_get_clean_model_filepaths(self):
        config = data_tools.load_config(self.model_filepath)
        clean_model_filepaths = data_tools.get_clean_model_filepaths(config, self.round_training_dataset_dirpath)
        self.assertTrue(len(clean_model_filepaths))
        self.assertTrue(clean_model_filepaths[0].split('/')[-2] == 'id-00000018')

    def test_load_examples(self):
        dataset = data_tools.load_examples(self.model_filepath, self.scratch_filepath)
        self.assertTrue(isinstance(dataset, datasets.arrow_dataset.Dataset))
        self.assertTrue(len(dataset)>0)

    def test_get_taskname(self):
        config = data_tools.load_config(self.model_filepath)
        task = data_tools.get_taskname(self.round_training_dataset_dirpath, config)
        print("here")



if __name__ == '__main__':
    unittest.main(verbosity=3)