import unittest
import torch
import tools
from trigger_attack.trigger import Trigger
from trigger_attack.preprocessing import ner
import warnings
from datasets.utils import set_progress_bar_enabled


class TestSCPreprocessing(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore")
        set_progress_bar_enabled(False)

        suspicious_model_filepath = (
            '../data/round9_sample_dataset/models/id-00000068/model.pt')
        clean_models_filepaths = [
            '../data/round9_sample_dataset/models/id-00000086/model.pt'
        ]
        tokenizer_filepath = (
            '../data/round9_sample_dataset/tokenizers'
            '/google-electra-small-discriminator.pt')

        dataset = tools.load_dataset(suspicious_model_filepath)
        trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        tokenizer = tools.load_tokenizer(tokenizer_filepath)
        trigger = Trigger(torch.tensor([1]*10), 'None', source_labels=[4])
        self.preprocessor = ner.NERDatasetPreprocessor(dataset,
                                                       trigger,
                                                       trigger_models,
                                                       tokenizer)

    def test_columns_after_tokenize(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        required_cols = set(['input_ids', 'attention_mask', 'token_type_ids', 'label'])
        existing_cols = set(tokenized_dataset.column_names)
        self.assertTrue(required_cols.issubset(existing_cols))

    def test_length_after_tokenize(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        expected_length = 81
        self.assertTrue(len(tokenized_dataset) == expected_length)

    def test_width_after_tokenize(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        expected_length = 384
        input_ids = torch.tensor(tokenized_dataset['input_ids'])
        self.assertTrue(input_ids.shape[1] <= expected_length)

    def test_select_unique_input(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        expected_length = 80
        self.assertTrue(len(unique_inputs_dataset) == expected_length)

    def test_initialize_dummy_trigger(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        dataset_with_dummy = self.preprocessor._insert_dummy(unique_inputs_dataset)
        first_input_ids_with_dummy = [101, 1023, 1012, 5003, 9496, 12426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13545, 5134, 2102, 1006, 3304, 1007, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        first_input_ids_without_dummy = dataset_with_dummy['input_ids'][0]
        self.assertTrue(torch.equal(first_input_ids_without_dummy, torch.tensor(first_input_ids_with_dummy)))        

    def test_baseline_probabilities_no_grad(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        dataset_with_dummy = self.preprocessor._insert_dummy(unique_inputs_dataset)
        dataset = self.preprocessor._add_baseline_probabilities(dataset_with_dummy)
        expected = torch.tensor([0.0051, 0.0022, 0.0013, 0.8853, 0.0795, 0.0093, 0.0065, 0.0078, 0.0029])
        actual = dataset['baseline_probabilities'][0].to('cpu')
        self.assertTrue(torch.allclose(expected, actual, atol=1e-02))

    def test_TriggerDataset(self):
        dataset = self.preprocessor.preprocess_data()
        self.assertTrue(True)

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)