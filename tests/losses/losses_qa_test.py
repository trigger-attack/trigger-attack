import unittest
import os
import torch
from torch.utils.data import DataLoader
from trigger_attack.preprocessing import qa, tools
from trigger_attack.loss_functions import qa as qaLoss
from trigger_attack.trigger_models import TriggerModels
from trojai_submission import data_tools
from datasets.utils.logging import set_verbosity_error
import warnings
warnings.filterwarnings("ignore")


class TestQAPreprocessing(unittest.TestCase):

    def setUp(self):
        self.dataset = self._load_dataset()
        self.models = self._load_models()
        set_verbosity_error()
        dataset = qa._tokenize_for_qa(self.dataset, self.models.tokenizer)
        dataset = qa._select_qa_examples_with_an_answer_in_context(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_loc = 'both'
        dataset = qa._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc)
        dataset = qa._add_baseline_probabilities(dataset, self.models)
        dataset = qa.QATriggeredDataset(dataset)
        batch_size = 16
        self.dataloader = DataLoader(dataset, batch_size=batch_size)
    
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

    @torch.no_grad()
    def test_suspicious_loss_targetting_self(self):
        for batch in self.dataloader:
            batch['trigger_mask'] = batch['trigger_mask'].to(self.models.device, non_blocking=True)
            batch['valid_mask'] = batch['valid_mask'].to(self.models.device, non_blocking=True)
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            trigger_target = 'self'
            loss = qaLoss.calculate_qa_suspicious_loss(all_logits,trigger_target, batch)
            break
        expected_loss = torch.tensor(8.0038, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    @torch.no_grad()
    def test_suspicious_loss_targetting_cls(self):
        for batch in self.dataloader:
            batch['trigger_mask'] = batch['trigger_mask'].to(self.models.device, non_blocking=True)
            batch['valid_mask'] = batch['valid_mask'].to(self.models.device, non_blocking=True)
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            trigger_target = 'cls'
            loss = qaLoss.calculate_qa_suspicious_loss(all_logits,trigger_target, batch)
            break
        expected_loss = torch.tensor(3.7878, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    def test_clean_loss_targetting_self(self):
        for batch in self.dataloader:
            batch['trigger_mask'] = batch['trigger_mask'].to(self.models.device, non_blocking=True)
            batch['valid_mask'] = batch['valid_mask'].to(self.models.device, non_blocking=True)
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            trigger_target = 'self'
            clean_agg_fn = torch.mean
            loss = qaLoss.calculate_qa_clean_loss(all_logits, batch, trigger_target, clean_agg_fn)
            break
        expected_loss = torch.tensor(0.0283, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    def test_loss_targetting_self(self):
        for batch in self.dataloader:
            batch['trigger_mask'] = batch['trigger_mask'].to(self.models.device, non_blocking=True)
            batch['valid_mask'] = batch['valid_mask'].to(self.models.device, non_blocking=True)
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            trigger_target = 'self'
            loss = qaLoss.calculate_qa_loss(all_logits, batch, trigger_target)
            break
        expected_loss = torch.tensor(8.0321, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)