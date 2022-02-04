import unittest
import os
import torch
from trigger_attack.preprocessing import qa, tools
from trigger_attack.trigger_models import TriggerModels
from trojai_submission import data_tools
import datasets
from datasets.utils.logging import set_verbosity_error
import warnings
warnings.filterwarnings("ignore")


class TestQAPreprocessing(unittest.TestCase):

    def setUp(self):
        self.dataset = self._load_dataset()
        self.models = self._load_models()
        set_verbosity_error()
    
    def _load_dataset(self):
        model_filepath = self._prepend_current_script_path('../data/round8_sample_dataset/models/id-00000000/model.pt')
        scratch_filepath = '.tmp'
        return data_tools.load_examples(model_filepath, scratch_filepath)

    def _load_models(self):
        suspicious_model_filepath = self._prepend_current_script_path('../data/round8_sample_dataset/models/id-00000000/model.pt')
        clean_model_filepaths = [self._prepend_current_script_path('../data/round8_sample_dataset/models/id-00000018/model.pt')]
        tokenizer_filepath = self._prepend_current_script_path('../data/round8_sample_dataset/tokenizers/tokenizer-roberta-base.pt')
        return TriggerModels(suspicious_model_filepath, clean_model_filepaths, tokenizer_filepath, device=torch.device('cuda'))
    
    @staticmethod
    def _prepend_current_script_path(path):
        current_script_dirname = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_script_dirname, path)


    def test_tokenize_for_qa(self):
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        self.assertTrue(isinstance(dataset, datasets.arrow_dataset.Dataset))
        self.assertTrue(len(dataset)>0)
        expected_cols = ['answer_start_and_end', 
                         'attention_mask', 
                         'context_start_and_end', 
                         'input_ids', 
                         'question_start_and_end', 
                         'token_type_ids']
        self.assertTrue(set(expected_cols).issubset(set(dataset.column_names)))
        expected_answer = " Sodor and Man Diocesan Synod"
        x = torch.tensor(dataset['input_ids'][0])
        answer_ids = x[dataset['answer_start_and_end'][0][0]: dataset['answer_start_and_end'][0][1]+1]
        obtained_answer = self.models.tokenizer.decode(answer_ids)
        self.assertTrue(expected_answer == obtained_answer)
        

    def test_select_examples_with_answers_in_context(self):
        expected_len = 12
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        self.assertTrue(len(dataset)==expected_len)


    def test_select_unique_inputs(self):
        expected_len = 11
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        self.assertTrue(len(dataset)==expected_len)


    def test_answer_mask(self):
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        
        trigger_length = 10
        trigger_loc = 'both'
        dataset = qa._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc)
        
        expected_answer = " Sodor and Man Diocesan Synod"
        answer = self.models.tokenizer.decode(dataset['input_ids'][0][dataset['answer_mask'][0].bool()])
        self.assertTrue(expected_answer==answer)
        
    def test_trigger_mask(self):
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        
        trigger_length = 10
        trigger_loc = 'both'
        dummy = 0
        dataset = qa._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc, dummy=dummy)
        
        expected_answer = torch.tensor([dummy] * 10 + [dummy] * 10)
        answer = dataset['input_ids'][0][dataset['trigger_mask'][0].bool()]
        self.assertTrue(torch.equal(expected_answer, answer))

    def test_add_baseline_probabilities(self):
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        
        trigger_length = 10
        trigger_loc = 'both'
        dataset = qa._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc)
        agg_function = torch.mean
        dataset = qa._add_baseline_probabilities(dataset, self.models)
        answer = dataset['baseline_probabilities'].argmax(dim=1)
        expected_answer = torch.tensor([ 37,  45,   0,  73,   0,  27,  48, 103, 223,  46,  32])
        self.assertTrue(torch.equal(answer, expected_answer))

    def test_TorchTriggeredDataset_len(self):
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        
        trigger_length = 10
        trigger_loc = 'both'
        dataset = qa._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc)
        agg_function = torch.mean
        dataset = qa._add_baseline_probabilities(dataset, self.models)
        dataset = qa.QATriggeredDataset(dataset)
        expected_length = 11
        self.assertTrue(len(dataset) == expected_length)

    def test_QATriggeredDataset_update_trigger(self):
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        
        trigger_length = 10
        trigger_loc = 'both'
        dataset = qa._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc)
        agg_function = torch.mean
        dataset = qa._add_baseline_probabilities(dataset, self.models)
        dataset = qa.QATriggeredDataset(dataset)
        new_trigger = torch.tensor(list(range(trigger_length)))
        dataset.update_trigger(new_trigger)
        input_ids = dataset[0]['input_ids']
        trigger_mask = dataset[0]['trigger_mask']
        self.assertTrue(torch.equal(input_ids[trigger_mask][:len(new_trigger)], new_trigger))

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)