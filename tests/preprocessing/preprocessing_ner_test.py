import unittest
import os
import torch
from trigger_attack.trigger_models import TriggerModels
from trigger_attack.preprocessing import ner, tools
from trojai_submission import data_tools
import warnings
warnings.filterwarnings("ignore")


class TestSCPreprocessing(unittest.TestCase):

    def setUp(self):
        self.dataset = self._load_dataset()
        self.models = self._load_models()

    def _load_dataset(self):
        model_filepath = self._prepend_current_script_path('../data/round9_sample_dataset/models/id-00000068/model.pt')
        scratch_filepath = '.tmp'
        return data_tools.load_examples(model_filepath, scratch_filepath)

    def _load_models(self):
        suspicious_model_filepath = self._prepend_current_script_path('../data/round9_sample_dataset/models/id-00000068/model.pt')
        clean_model_filepaths = [self._prepend_current_script_path('../data/round9_sample_dataset/models/id-00000086/model.pt')]
        tokenizer_filepath = self._prepend_current_script_path('../data/round9_sample_dataset/tokenizers/google-electra-small-discriminator.pt')
        return TriggerModels(suspicious_model_filepath, clean_model_filepaths, tokenizer_filepath, device=torch.device('cuda'))

    @staticmethod
    def _prepend_current_script_path(path):
        current_script_dirname = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_script_dirname, path)

    def test_columns_tokenize_for_ner_columns(self):
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        required_cols = set(['input_ids', 'attention_mask', 'token_type_ids', 'label'])
        existing_cols = set(dataset.column_names)
        self.assertTrue(required_cols.issubset(existing_cols))

    def test_length_tokenize_for_ner(self):
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        expected_length = 81
        self.assertTrue(len(dataset) == expected_length)

    def test_width_tokenize_for_ner(self):
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        expected_length = 384
        input_ids = torch.tensor(dataset['input_ids'])
        self.assertTrue(input_ids.shape[1] <= expected_length)

    def test_select_unique_input(self):
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        expected_length = 80
        self.assertTrue(len(dataset) == expected_length)

    def test_start_initialize_dummy_trigger(self):
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_source_labels = [4]
        dummy = self.models.tokenizer.pad_token_id
        dataset = ner._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_source_labels, dummy=dummy)
        first_input_ids_with_dummy = [101, 1023, 1012, 5003, 9496, 12426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13545, 5134, 2102, 1006, 3304, 1007, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        first_input_ids_without_dummy = dataset['input_ids'][0]
        self.assertTrue(torch.equal(first_input_ids_without_dummy, torch.tensor(first_input_ids_with_dummy)))        
    
    def test_baseline_probabilities_no_grad(self):
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_source_labels = [4]
        dummy = self.models.tokenizer.pad_token_id
        dataset = ner._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_source_labels, dummy=dummy)
        agg_function = torch.mean
        dataset = ner._add_baseline_probabilities(dataset, self.models)
        expected = torch.tensor([0.0051, 0.0022, 0.0013, 0.8853, 0.0795, 0.0093, 0.0065, 0.0078, 0.0029])
        actual = dataset['baseline_probabilities'][0].to('cpu')
        self.assertTrue(torch.allclose(expected, actual, atol=1e-02))

    def test_TriggerDataset(self):
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_source_labels = [4]
        dummy = self.models.tokenizer.pad_token_id
        dataset = ner._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_source_labels, dummy=dummy)
        agg_function = torch.mean
        dataset = ner._add_baseline_probabilities(dataset, self.models)
        dataset = ner.NERTriggeredDataset(dataset)
        self.assertTrue(True)

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)