import unittest
import tools
import torch
from trigger_attack.preprocessing import qa
from trigger_attack.trigger import Trigger
import datasets
import warnings
from datasets.utils import set_progress_bar_enabled



class TestQAPreprocessing(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore")
        set_progress_bar_enabled(False)

        suspicious_model_filepath = (
            '../data/round8_sample_dataset/models/id-00000000/model.pt')
        clean_models_filepaths = [
            '../data/round8_sample_dataset/models/id-00000018/model.pt'
        ]
        tokenizer_filepath = (
            '../data/round8_sample_dataset/tokenizers'
            '/tokenizer-roberta-base.pt')

        dataset = tools.load_dataset(suspicious_model_filepath)
        trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        tokenizer = tools.load_tokenizer(tokenizer_filepath)
        trigger = Trigger(torch.tensor([0]*10), location='both', source_labels=None)
        self.preprocessor = qa.QADatasetPreprocessor(dataset,
                                                     trigger,
                                                     trigger_models,
                                                     tokenizer)

    def test_tokenize_for_qa(self):
        original_dataset = self.preprocessor.dataset
        tokenized_dataset = self.preprocessor._tokenize(original_dataset)
        self.assertTrue(isinstance(tokenized_dataset, datasets.arrow_dataset.Dataset))
        self.assertTrue(len(tokenized_dataset)>0)
        expected_cols = ['answer_start_and_end', 
                         'attention_mask', 
                         'context_start_and_end', 
                         'input_ids', 
                         'question_start_and_end', 
                         'token_type_ids']
        self.assertTrue(set(expected_cols).issubset(set(tokenized_dataset.column_names)))
        expected_answer = " Sodor and Man Diocesan Synod"
        x = torch.tensor(tokenized_dataset['input_ids'][0])
        answer_ids = x[tokenized_dataset['answer_start_and_end'][0][0]: tokenized_dataset['answer_start_and_end'][0][1]+1]
        obtained_answer = self.preprocessor.tokenizer.decode(answer_ids)
        self.assertTrue(expected_answer == obtained_answer)

    def test_select_examples_with_answers_in_context(self):
        expected_len = 12
        original_dataset = self.preprocessor.dataset
        tokenized_dataset = self.preprocessor._tokenize(original_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(tokenized_dataset)
        self.assertTrue(len(source_class_dataset)==expected_len)

    def test_select_unique_inputs(self):
        expected_len = 11
        original_dataset = self.preprocessor.dataset
        tokenized_dataset = self.preprocessor._tokenize(original_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(tokenized_dataset)
        unique_input_dataset = self.preprocessor._select_unique_inputs(source_class_dataset)
        self.assertTrue(len(unique_input_dataset)==expected_len)

    def test_answer_mask(self):
        original_dataset = self.preprocessor.dataset
        tokenized_dataset = self.preprocessor._tokenize(original_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(tokenized_dataset)
        unique_input_dataset = self.preprocessor._select_unique_inputs(source_class_dataset)
        dataset_with_dummy = self.preprocessor._insert_dummy(unique_input_dataset)        

        expected_answer = " Sodor and Man Diocesan Synod"
        answer = self.preprocessor.tokenizer.decode(dataset_with_dummy['input_ids'][0][dataset_with_dummy['answer_mask'][0].bool()])
        self.assertTrue(expected_answer==answer)

    def test_trigger_mask(self):
        original_dataset = self.preprocessor.dataset
        tokenized_dataset = self.preprocessor._tokenize(original_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(tokenized_dataset)
        unique_input_dataset = self.preprocessor._select_unique_inputs(source_class_dataset)
        dataset_with_dummy = self.preprocessor._insert_dummy(unique_input_dataset)        

        dummy = self.preprocessor.tokenizer.pad_token_id
        expected_answer = torch.tensor([dummy]*20)
        answer = dataset_with_dummy['input_ids'][0][dataset_with_dummy['trigger_mask'][0].bool()]
        self.assertTrue(torch.equal(expected_answer, answer))

    def test_add_baseline_probabilities(self):
        original_dataset = self.preprocessor.dataset
        tokenized_dataset = self.preprocessor._tokenize(original_dataset)
        source_class_dataset = self.preprocessor._select_inputs_with_source_class(tokenized_dataset)
        unique_input_dataset = self.preprocessor._select_unique_inputs(source_class_dataset)
        dataset_with_dummy = self.preprocessor._insert_dummy(unique_input_dataset)
        dataset_with_baseline_probabilities = self.preprocessor._add_baseline_probabilities(dataset_with_dummy)

        answer = dataset_with_baseline_probabilities['baseline_probabilities'].argmax(dim=1)
        # expected_answer = torch.tensor([ 37,  45,   0,  73,   0,  27,  48, 103, 223,  46,  32])
        expected_answer = torch.tensor([ 37,  45, 185,  76,  53,  27,  48, 103, 223,  46,  32])
        self.assertTrue(torch.equal(answer, expected_answer))

    def test_TorchTriggeredDataset_len(self):
        dataset = self.preprocessor.preprocess_data()
        expected_length = 11
        self.assertTrue(len(dataset) == expected_length)

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)
