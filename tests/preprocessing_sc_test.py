import unittest
import os
from cv2 import threshold
import torch
from trigger_attack.trigger_models import TriggerModels
from trigger_attack.preprocessing import sc, tools
from trojai_submission import data_tools


class TestSCPreprocessing(unittest.TestCase):

    def setUp(self):
        self.dataset = self._load_dataset()
        self.models = self._load_models()

    def _load_dataset(self):
        model_filepath = self._prepend_current_script_path('data/round9_sample_dataset/models/id-00000014/model.pt')
        scratch_filepath = '.tmp'
        return data_tools.load_examples(model_filepath, scratch_filepath)

    def _load_models(self):
        suspicious_model_filepath = self._prepend_current_script_path('data/round9_sample_dataset/models/id-00000014/model.pt')
        clean_model_filepaths = [self._prepend_current_script_path('data/round9_sample_dataset/models/id-00000002/model.pt')]
        tokenizer_filepath = self._prepend_current_script_path('data/round9_sample_dataset/tokenizers/roberta-base.pt')
        return TriggerModels(suspicious_model_filepath, clean_model_filepaths, tokenizer_filepath, device=torch.device('cuda'))

    @staticmethod
    def _prepend_current_script_path(path):
        current_script_dirname = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_script_dirname, path)

    def test_columns_tokenize_for_sc_columns(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        required_cols = set(['input_ids', 'attention_mask', 'token_type_ids'])
        existing_cols = set(dataset.column_names)
        self.assertTrue(required_cols.issubset(existing_cols))

    def test_length_tokenize_for_sc(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        expected_length = 58
        self.assertTrue(len(dataset) == expected_length)

    def test_width_tokenize_for_sc(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        expected_length = 384
        input_ids = torch.tensor(dataset['input_ids'])
        self.assertTrue(input_ids.shape[1] == expected_length)

    def test_select_unique_input(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        expected_length = 56
        self.assertTrue(len(dataset) == expected_length)

    def test_start_initialize_dummy_trigger(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_loc = 'start'
        dummy = self.models.tokenizer.pad_token_id
        dataset = sc._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc, dummy=dummy)
        first_input_ids_with_dummy = [0] + [dummy]*trigger_length + [1817, 19041, 139, 34, 393, 626, 10, 822, 36, 2527, 444, 25, 38, 216, 6, 8, 42, 1171, 1980, 3660, 30, 69, 308, 1150, 6, 211, 7718, 43, 147, 79, 10578, 7, 311, 70, 9, 69, 38272, 23, 103, 477, 4, 9136, 615, 6, 11, 5, 144, 15305, 1273, 6054, 1310, 655, 6, 2129, 4145, 7305, 34, 69, 567, 88, 39, 558, 8, 235, 89, 6, 19327, 201, 14, 190, 600, 69, 2549, 16, 62, 6, 79, 64, 202, 4757, 69, 10802, 11, 69, 42613, 23, 143, 576, 200, 36, 5488, 79, 473, 53, 11, 215, 10, 9624, 22, 12196, 116, 524, 38, 269, 1782, 14, 1917, 761, 9, 169, 322, 20, 19888, 2233, 6, 9805, 6, 817, 42, 356, 101, 10, 24295, 1794, 5886, 1627, 822, 98, 47, 489, 2445, 7, 192, 69, 1004, 88, 10, 19474, 8, 33414, 7844, 7586, 117, 215, 6620, 4, 264, 18, 33760, 3876, 6, 10601, 62, 15, 103, 10662, 34, 57, 8, 190, 3007, 6, 34, 117, 10280, 2417, 11, 2166, 28924, 97, 87, 36, 12557, 22627, 13278, 602, 160, 12389, 5, 5418, 77, 143, 1310, 2386, 24, 4, 1405, 7678, 9663, 2370, 74, 28, 3035, 114, 129, 79, 399, 75, 860, 7, 146, 24, 2369, 98, 2132, 8, 860, 7, 1067, 101, 10, 316, 76, 793, 4, 1336, 59, 42, 6197, 116, 30856, 12, 11261, 53, 11586, 295, 36935, 1075, 1543, 1043, 337, 693, 16, 6017, 3185, 7, 3549, 150, 79, 416, 34, 277, 7226, 15, 5, 213, 8, 16, 878, 103, 6162, 1262, 432, 1666, 41437, 116, 99, 116, 473, 1268, 33, 10563, 11, 42, 1569, 7, 109, 932, 97, 87, 907, 10, 11402, 3121, 116, 20, 26735, 1302, 7, 28, 28094, 30, 6901, 961, 1067, 10, 319, 59, 5, 276, 383, 81, 8, 81, 36, 100, 21, 4804, 7, 192, 5, 2373, 3501, 15, 42, 2082, 25, 10, 3436, 54, 12961, 418, 11, 24, 95, 7, 33, 103, 2441, 86, 43, 111, 89, 16, 1085, 164, 15, 2849, 5, 4678, 16855, 9, 1817, 18, 2048, 14, 23, 143, 477, 11, 5, 1569, 8806, 99, 5, 19888, 1719, 5343, 4, 264, 18, 3953, 734, 53, 79, 2215, 141, 7, 3549, 4, 79, 2342, 11791, 83, 37567, 4, 264, 2342, 11791, 12343, 6, 79, 2342, 11791, 457, 7001, 6, 79, 190, 2342, 11791, 11, 2]
        first_input_ids_without_dummy = dataset['input_ids'][0]
        self.assertTrue(torch.equal(torch.tensor(first_input_ids_without_dummy), torch.tensor(first_input_ids_with_dummy)))        

    def test_end_initialize_dummy_trigger(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_loc = 'end'
        dummy = self.models.tokenizer.pad_token_id
        dataset = sc._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc, dummy=dummy)
        first_input_ids_with_dummy = [0, 1817, 19041, 139, 34, 393, 626, 10, 822, 36, 2527, 444, 25, 38, 216, 6, 8, 42, 1171, 1980, 3660, 30, 69, 308, 1150, 6, 211, 7718, 43, 147, 79, 10578, 7, 311, 70, 9, 69, 38272, 23, 103, 477, 4, 9136, 615, 6, 11, 5, 144, 15305, 1273, 6054, 1310, 655, 6, 2129, 4145, 7305, 34, 69, 567, 88, 39, 558, 8, 235, 89, 6, 19327, 201, 14, 190, 600, 69, 2549, 16, 62, 6, 79, 64, 202, 4757, 69, 10802, 11, 69, 42613, 23, 143, 576, 200, 36, 5488, 79, 473, 53, 11, 215, 10, 9624, 22, 12196, 116, 524, 38, 269, 1782, 14, 1917, 761, 9, 169, 322, 20, 19888, 2233, 6, 9805, 6, 817, 42, 356, 101, 10, 24295, 1794, 5886, 1627, 822, 98, 47, 489, 2445, 7, 192, 69, 1004, 88, 10, 19474, 8, 33414, 7844, 7586, 117, 215, 6620, 4, 264, 18, 33760, 3876, 6, 10601, 62, 15, 103, 10662, 34, 57, 8, 190, 3007, 6, 34, 117, 10280, 2417, 11, 2166, 28924, 97, 87, 36, 12557, 22627, 13278, 602, 160, 12389, 5, 5418, 77, 143, 1310, 2386, 24, 4, 1405, 7678, 9663, 2370, 74, 28, 3035, 114, 129, 79, 399, 75, 860, 7, 146, 24, 2369, 98, 2132, 8, 860, 7, 1067, 101, 10, 316, 76, 793, 4, 1336, 59, 42, 6197, 116, 30856, 12, 11261, 53, 11586, 295, 36935, 1075, 1543, 1043, 337, 693, 16, 6017, 3185, 7, 3549, 150, 79, 416, 34, 277, 7226, 15, 5, 213, 8, 16, 878, 103, 6162, 1262, 432, 1666, 41437, 116, 99, 116, 473, 1268, 33, 10563, 11, 42, 1569, 7, 109, 932, 97, 87, 907, 10, 11402, 3121, 116, 20, 26735, 1302, 7, 28, 28094, 30, 6901, 961, 1067, 10, 319, 59, 5, 276, 383, 81, 8, 81, 36, 100, 21, 4804, 7, 192, 5, 2373, 3501, 15, 42, 2082, 25, 10, 3436, 54, 12961, 418, 11, 24, 95, 7, 33, 103, 2441, 86, 43, 111, 89, 16, 1085, 164, 15, 2849, 5, 4678, 16855, 9, 1817, 18, 2048, 14, 23, 143, 477, 11, 5, 1569, 8806, 99, 5, 19888, 1719, 5343, 4, 264, 18, 3953, 734, 53, 79, 2215, 141, 7, 3549, 4, 79, 2342, 11791, 83, 37567, 4, 264, 2342, 11791, 12343, 6, 79, 2342, 11791, 457, 7001, 6, 79, 190, 2342, 11791, 11] + [dummy]*trigger_length + [2]
        first_input_ids_without_dummy = dataset['input_ids'][0]
        self.assertTrue(torch.equal(torch.tensor(first_input_ids_without_dummy), torch.tensor(first_input_ids_with_dummy)))

    def test_middle_initialize_dummy_trigger(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_loc = 'middle'
        dummy = self.models.tokenizer.pad_token_id
        dataset = sc._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc, dummy=dummy)
        first_input_ids_with_dummy = [0, 1817, 19041, 139, 34, 393, 626, 10, 822, 36, 2527, 444, 25, 38, 216, 6, 8, 42, 1171, 1980, 3660, 30, 69, 308, 1150, 6, 211, 7718, 43, 147, 79, 10578, 7, 311, 70, 9, 69, 38272, 23, 103, 477, 4, 9136, 615, 6, 11, 5, 144, 15305, 1273, 6054, 1310, 655, 6, 2129, 4145, 7305, 34, 69, 567, 88, 39, 558, 8, 235, 89, 6, 19327, 201, 14, 190, 600, 69, 2549, 16, 62, 6, 79, 64, 202, 4757, 69, 10802, 11, 69, 42613, 23, 143, 576, 200, 36, 5488, 79, 473, 53, 11, 215, 10, 9624, 22, 12196, 116, 524, 38, 269, 1782, 14, 1917, 761, 9, 169, 322, 20, 19888, 2233, 6, 9805, 6, 817, 42, 356, 101, 10, 24295, 1794, 5886, 1627, 822, 98, 47, 489, 2445, 7, 192, 69, 1004, 88, 10, 19474, 8, 33414, 7844, 7586, 117, 215, 6620, 4, 264, 18, 33760, 3876, 6, 10601, 62, 15, 103, 10662, 34, 57, 8, 190, 3007, 6, 34, 117, 10280, 2417, 11, 2166, 28924, 97, 87, 36, 12557, 22627, 13278, 602, 160, 12389, 5, 5418, 77, 143, 1310, 2386, 24, 4, 1405, 7678, 9663, 2370, 74] + [dummy]*trigger_length + [28, 3035, 114, 129, 79, 399, 75, 860, 7, 146, 24, 2369, 98, 2132, 8, 860, 7, 1067, 101, 10, 316, 76, 793, 4, 1336, 59, 42, 6197, 116, 30856, 12, 11261, 53, 11586, 295, 36935, 1075, 1543, 1043, 337, 693, 16, 6017, 3185, 7, 3549, 150, 79, 416, 34, 277, 7226, 15, 5, 213, 8, 16, 878, 103, 6162, 1262, 432, 1666, 41437, 116, 99, 116, 473, 1268, 33, 10563, 11, 42, 1569, 7, 109, 932, 97, 87, 907, 10, 11402, 3121, 116, 20, 26735, 1302, 7, 28, 28094, 30, 6901, 961, 1067, 10, 319, 59, 5, 276, 383, 81, 8, 81, 36, 100, 21, 4804, 7, 192, 5, 2373, 3501, 15, 42, 2082, 25, 10, 3436, 54, 12961, 418, 11, 24, 95, 7, 33, 103, 2441, 86, 43, 111, 89, 16, 1085, 164, 15, 2849, 5, 4678, 16855, 9, 1817, 18, 2048, 14, 23, 143, 477, 11, 5, 1569, 8806, 99, 5, 19888, 1719, 5343, 4, 264, 18, 3953, 734, 53, 79, 2215, 141, 7, 3549, 4, 79, 2342, 11791, 83, 37567, 4, 264, 2342, 11791, 12343, 6, 79, 2342, 11791, 457, 7001, 6, 79, 190, 2342, 11791, 11, 2]
        first_input_ids_without_dummy = dataset['input_ids'][0]
        self.assertTrue(torch.equal(torch.tensor(first_input_ids_without_dummy), torch.tensor(first_input_ids_with_dummy)))
    
    def test_baseline_probabilities_no_grad(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_loc = 'start'
        dummy = self.models.tokenizer.pad_token_id
        dataset = sc._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc, dummy=dummy)
        agg_function = torch.mean
        dataset = sc._add_baseline_probabilities(dataset, self.models, agg_function)
        expected = torch.tensor([0.9962, 0.0038])
        actual = dataset['baseline_probabilities'][0].to('cpu')
        self.assertTrue(torch.allclose(expected, actual, atol=1e-02))

    def test_TriggerDataset(self):
        dataset = sc._tokenize_for_sc(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_loc = 'start'
        dummy = self.models.tokenizer.pad_token_id
        dataset = sc._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc, dummy=dummy)
        agg_function = torch.mean
        dataset = sc._add_baseline_probabilities(dataset, self.models, agg_function)
        dataset = tools.TorchTriggeredDataset(dataset)
        self.assertTrue(True)

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)