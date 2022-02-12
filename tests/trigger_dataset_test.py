import unittest
import os
import torch
from trigger_attack.trigger_dataset import TriggerDataset
from trigger_attack.trigger_models import TriggerModels
from trojai_submission import data_tools


class TestTriggerDataset(unittest.TestCase):
    
    def setUp(self):
        current_script_dirname = os.path.dirname(os.path.abspath(__file__))
        model_filepath = os.path.join(current_script_dirname, 'data/round8_sample_dataset/models/id-00000000/model.pt')
        round_training_dataset_dirpath = os.path.join(current_script_dirname, 'data/round8_sample_dataset')
        tokenizer_filepath = os.path.join(current_script_dirname, 'data/round8_sample_dataset/tokenizers/tokenizer-roberta-base.pt')
        config = data_tools.load_config(model_filepath)
        clean_model_filepaths = data_tools.get_clean_model_filepaths(config, round_training_dataset_dirpath)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        models = TriggerModels(model_filepath, clean_model_filepaths, tokenizer_filepath, device=device)
        dataset = data_tools.load_examples(model_filepath, scratch_dirpath  ='.tmp')

        trigger_length = 10
        trigger_init_fn = 'embed_ch'
        trigger_loc = 'both'
        task = 'qa'
        self.trigger_dataset = TriggerDataset(dataset, models, trigger_length, trigger_init_fn, trigger_loc, task)

    def test_init(self):
        '''
        '''
        return NotImplementedError

    def test_insert_new_trigger(self):
        return NotImplementedError

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)