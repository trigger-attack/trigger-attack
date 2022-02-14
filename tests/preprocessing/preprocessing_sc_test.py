import unittest
import tools
import torch
from trigger_attack.trigger import Trigger
from trigger_attack.preprocessing import sc
import warnings
from datasets.utils import set_progress_bar_enabled


class TestSCPreprocessing(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore")
        set_progress_bar_enabled(False)

        suspicious_model_filepath = (
            '../data/round9_sample_dataset/models/id-00000014/model.pt')
        clean_models_filepaths = [
            '../data/round9_sample_dataset/models/id-00000002/model.pt'
        ]
        tokenizer_filepath = (
            '../data/round9_sample_dataset/tokenizers/roberta-base.pt')

        dataset = tools.load_dataset(suspicious_model_filepath)
        trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        tokenizer = tools.load_tokenizer(tokenizer_filepath)
        trigger = Trigger(torch.tensor([1]*10), location='start', source_labels=[0])
        self.preprocessor = sc.SCDatasetPreprocessor(dataset,
                                                     trigger,
                                                     trigger_models,
                                                     tokenizer)

    def test_columns_tokenize_for_sc_columns(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        required_cols = set(['input_ids', 'attention_mask', 'token_type_ids'])
        existing_cols = set(tokenized_dataset.column_names)
        self.assertTrue(required_cols.issubset(existing_cols))


    def test_length_tokenize_for_sc(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        expected_length = 58
        self.assertTrue(len(tokenized_dataset) == expected_length)


    def test_width_tokenize_for_sc(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        expected_length = 384
        input_ids = torch.tensor(tokenized_dataset['input_ids'])
        self.assertTrue(input_ids.shape[1] == expected_length)


    def test_select_unique_input(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        expected_length = 56
        self.assertTrue(len(unique_inputs_dataset) == expected_length)


    def test_select_inputs_with_source_class(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(unique_inputs_dataset)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue(source_class_dataset['label']==expected)


    def test_start_initialize_dummy_trigger(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(unique_inputs_dataset)
        self.preprocessor.trigger.location = 'start'
        dummy = self.preprocessor.tokenizer.pad_token_id
        dummy_dataset = self.preprocessor._insert_dummy(source_class_dataset)
        first_input_ids_with_dummy = [0] + [dummy]*10 + [1817, 19041, 139, 34, 393, 626, 10, 822, 36, 2527, 444, 25, 38, 216, 6, 8, 42, 1171, 1980, 3660, 30, 69, 308, 1150, 6, 211, 7718, 43, 147, 79, 10578, 7, 311, 70, 9, 69, 38272, 23, 103, 477, 4, 9136, 615, 6, 11, 5, 144, 15305, 1273, 6054, 1310, 655, 6, 2129, 4145, 7305, 34, 69, 567, 88, 39, 558, 8, 235, 89, 6, 19327, 201, 14, 190, 600, 69, 2549, 16, 62, 6, 79, 64, 202, 4757, 69, 10802, 11, 69, 42613, 23, 143, 576, 200, 36, 5488, 79, 473, 53, 11, 215, 10, 9624, 22, 12196, 116, 524, 38, 269, 1782, 14, 1917, 761, 9, 169, 322, 20, 19888, 2233, 6, 9805, 6, 817, 42, 356, 101, 10, 24295, 1794, 5886, 1627, 822, 98, 47, 489, 2445, 7, 192, 69, 1004, 88, 10, 19474, 8, 33414, 7844, 7586, 117, 215, 6620, 4, 264, 18, 33760, 3876, 6, 10601, 62, 15, 103, 10662, 34, 57, 8, 190, 3007, 6, 34, 117, 10280, 2417, 11, 2166, 28924, 97, 87, 36, 12557, 22627, 13278, 602, 160, 12389, 5, 5418, 77, 143, 1310, 2386, 24, 4, 1405, 7678, 9663, 2370, 74, 28, 3035, 114, 129, 79, 399, 75, 860, 7, 146, 24, 2369, 98, 2132, 8, 860, 7, 1067, 101, 10, 316, 76, 793, 4, 1336, 59, 42, 6197, 116, 30856, 12, 11261, 53, 11586, 295, 36935, 1075, 1543, 1043, 337, 693, 16, 6017, 3185, 7, 3549, 150, 79, 416, 34, 277, 7226, 15, 5, 213, 8, 16, 878, 103, 6162, 1262, 432, 1666, 41437, 116, 99, 116, 473, 1268, 33, 10563, 11, 42, 1569, 7, 109, 932, 97, 87, 907, 10, 11402, 3121, 116, 20, 26735, 1302, 7, 28, 28094, 30, 6901, 961, 1067, 10, 319, 59, 5, 276, 383, 81, 8, 81, 36, 100, 21, 4804, 7, 192, 5, 2373, 3501, 15, 42, 2082, 25, 10, 3436, 54, 12961, 418, 11, 24, 95, 7, 33, 103, 2441, 86, 43, 111, 89, 16, 1085, 164, 15, 2849, 5, 4678, 16855, 9, 1817, 18, 2048, 14, 23, 143, 477, 11, 5, 1569, 8806, 99, 5, 19888, 1719, 5343, 4, 264, 18, 3953, 734, 53, 79, 2215, 141, 7, 3549, 4, 79, 2342, 11791, 83, 37567, 4, 264, 2342, 11791, 12343, 6, 79, 2342, 11791, 457, 7001, 6, 79, 190, 2342, 11791, 11, 2]
        first_input_ids_without_dummy = dummy_dataset['input_ids'][0]
        self.assertTrue(torch.equal(torch.tensor(first_input_ids_without_dummy), torch.tensor(first_input_ids_with_dummy)))        


    def test_end_initialize_dummy_trigger(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(unique_inputs_dataset)
        self.preprocessor.trigger.location = 'end'
        dummy = self.preprocessor.tokenizer.pad_token_id
        dummy_dataset = self.preprocessor._insert_dummy(source_class_dataset)
        first_input_ids_with_dummy = [0, 1817, 19041, 139, 34, 393, 626, 10, 822, 36, 2527, 444, 25, 38, 216, 6, 8, 42, 1171, 1980, 3660, 30, 69, 308, 1150, 6, 211, 7718, 43, 147, 79, 10578, 7, 311, 70, 9, 69, 38272, 23, 103, 477, 4, 9136, 615, 6, 11, 5, 144, 15305, 1273, 6054, 1310, 655, 6, 2129, 4145, 7305, 34, 69, 567, 88, 39, 558, 8, 235, 89, 6, 19327, 201, 14, 190, 600, 69, 2549, 16, 62, 6, 79, 64, 202, 4757, 69, 10802, 11, 69, 42613, 23, 143, 576, 200, 36, 5488, 79, 473, 53, 11, 215, 10, 9624, 22, 12196, 116, 524, 38, 269, 1782, 14, 1917, 761, 9, 169, 322, 20, 19888, 2233, 6, 9805, 6, 817, 42, 356, 101, 10, 24295, 1794, 5886, 1627, 822, 98, 47, 489, 2445, 7, 192, 69, 1004, 88, 10, 19474, 8, 33414, 7844, 7586, 117, 215, 6620, 4, 264, 18, 33760, 3876, 6, 10601, 62, 15, 103, 10662, 34, 57, 8, 190, 3007, 6, 34, 117, 10280, 2417, 11, 2166, 28924, 97, 87, 36, 12557, 22627, 13278, 602, 160, 12389, 5, 5418, 77, 143, 1310, 2386, 24, 4, 1405, 7678, 9663, 2370, 74, 28, 3035, 114, 129, 79, 399, 75, 860, 7, 146, 24, 2369, 98, 2132, 8, 860, 7, 1067, 101, 10, 316, 76, 793, 4, 1336, 59, 42, 6197, 116, 30856, 12, 11261, 53, 11586, 295, 36935, 1075, 1543, 1043, 337, 693, 16, 6017, 3185, 7, 3549, 150, 79, 416, 34, 277, 7226, 15, 5, 213, 8, 16, 878, 103, 6162, 1262, 432, 1666, 41437, 116, 99, 116, 473, 1268, 33, 10563, 11, 42, 1569, 7, 109, 932, 97, 87, 907, 10, 11402, 3121, 116, 20, 26735, 1302, 7, 28, 28094, 30, 6901, 961, 1067, 10, 319, 59, 5, 276, 383, 81, 8, 81, 36, 100, 21, 4804, 7, 192, 5, 2373, 3501, 15, 42, 2082, 25, 10, 3436, 54, 12961, 418, 11, 24, 95, 7, 33, 103, 2441, 86, 43, 111, 89, 16, 1085, 164, 15, 2849, 5, 4678, 16855, 9, 1817, 18, 2048, 14, 23, 143, 477, 11, 5, 1569, 8806, 99, 5, 19888, 1719, 5343, 4, 264, 18, 3953, 734, 53, 79, 2215, 141, 7, 3549, 4, 79, 2342, 11791, 83, 37567, 4, 264, 2342, 11791, 12343, 6, 79, 2342, 11791, 457, 7001, 6, 79, 190, 2342, 11791, 11] + [dummy]*10 + [2]
        first_input_ids_without_dummy = dummy_dataset['input_ids'][0]
        self.assertTrue(torch.equal(torch.tensor(first_input_ids_without_dummy), torch.tensor(first_input_ids_with_dummy)))


    def test_middle_initialize_dummy_trigger(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(unique_inputs_dataset)
        self.preprocessor.trigger.location = 'middle'
        dummy = self.preprocessor.tokenizer.pad_token_id
        dummy_dataset = self.preprocessor._insert_dummy(source_class_dataset)
        first_input_ids_with_dummy = [0, 1817, 19041, 139, 34, 393, 626, 10, 822, 36, 2527, 444, 25, 38, 216, 6, 8, 42, 1171, 1980, 3660, 30, 69, 308, 1150, 6, 211, 7718, 43, 147, 79, 10578, 7, 311, 70, 9, 69, 38272, 23, 103, 477, 4, 9136, 615, 6, 11, 5, 144, 15305, 1273, 6054, 1310, 655, 6, 2129, 4145, 7305, 34, 69, 567, 88, 39, 558, 8, 235, 89, 6, 19327, 201, 14, 190, 600, 69, 2549, 16, 62, 6, 79, 64, 202, 4757, 69, 10802, 11, 69, 42613, 23, 143, 576, 200, 36, 5488, 79, 473, 53, 11, 215, 10, 9624, 22, 12196, 116, 524, 38, 269, 1782, 14, 1917, 761, 9, 169, 322, 20, 19888, 2233, 6, 9805, 6, 817, 42, 356, 101, 10, 24295, 1794, 5886, 1627, 822, 98, 47, 489, 2445, 7, 192, 69, 1004, 88, 10, 19474, 8, 33414, 7844, 7586, 117, 215, 6620, 4, 264, 18, 33760, 3876, 6, 10601, 62, 15, 103, 10662, 34, 57, 8, 190, 3007, 6, 34, 117, 10280, 2417, 11, 2166, 28924, 97, 87, 36, 12557, 22627, 13278, 602, 160, 12389, 5, 5418, 77, 143, 1310, 2386, 24, 4, 1405, 7678, 9663, 2370, 74] + [dummy]*10 + [28, 3035, 114, 129, 79, 399, 75, 860, 7, 146, 24, 2369, 98, 2132, 8, 860, 7, 1067, 101, 10, 316, 76, 793, 4, 1336, 59, 42, 6197, 116, 30856, 12, 11261, 53, 11586, 295, 36935, 1075, 1543, 1043, 337, 693, 16, 6017, 3185, 7, 3549, 150, 79, 416, 34, 277, 7226, 15, 5, 213, 8, 16, 878, 103, 6162, 1262, 432, 1666, 41437, 116, 99, 116, 473, 1268, 33, 10563, 11, 42, 1569, 7, 109, 932, 97, 87, 907, 10, 11402, 3121, 116, 20, 26735, 1302, 7, 28, 28094, 30, 6901, 961, 1067, 10, 319, 59, 5, 276, 383, 81, 8, 81, 36, 100, 21, 4804, 7, 192, 5, 2373, 3501, 15, 42, 2082, 25, 10, 3436, 54, 12961, 418, 11, 24, 95, 7, 33, 103, 2441, 86, 43, 111, 89, 16, 1085, 164, 15, 2849, 5, 4678, 16855, 9, 1817, 18, 2048, 14, 23, 143, 477, 11, 5, 1569, 8806, 99, 5, 19888, 1719, 5343, 4, 264, 18, 3953, 734, 53, 79, 2215, 141, 7, 3549, 4, 79, 2342, 11791, 83, 37567, 4, 264, 2342, 11791, 12343, 6, 79, 2342, 11791, 457, 7001, 6, 79, 190, 2342, 11791, 11, 2]
        first_input_ids_without_dummy = dummy_dataset['input_ids'][0]
        self.assertTrue(torch.equal(torch.tensor(first_input_ids_without_dummy), torch.tensor(first_input_ids_with_dummy)))


    def test_baseline_probabilities_no_grad(self):
        tokenized_dataset = self.preprocessor._tokenize(self.preprocessor.dataset)
        unique_inputs_dataset = self.preprocessor._select_unique_inputs(tokenized_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(unique_inputs_dataset)
        self.preprocessor.trigger.location = 'middle'
        dummy = self.preprocessor.tokenizer.pad_token_id
        dummy_dataset = self.preprocessor._insert_dummy(source_class_dataset)
        dataset_with_baseline_probabilities = self.preprocessor._add_baseline_probabilities(dummy_dataset)
        expected = torch.tensor([0.9962, 0.0038])
        actual = dataset_with_baseline_probabilities['baseline_probabilities'][0].to('cpu')
        self.assertTrue(torch.allclose(expected, actual, atol=1e-02))


    def test_TriggerDataset(self):
        dataset = self.preprocessor.preprocess_data()
        self.assertTrue(True)


    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)
