import unittest
import os
import torch
from torch.utils.data import DataLoader
from trigger_attack.trigger_models import TriggerModels
from trigger_attack.preprocessing import sc as scPreprocess
from trigger_attack.preprocessing import tools
from trigger_attack.loss_functions import sc as scLoss
from trojai_submission import data_tools
import warnings
warnings.filterwarnings("ignore")


class TestSCPreprocessing(unittest.TestCase):

    def setUp(self):
        self.trigger_source_labels = [0]
        self.trigger_target_label = 1
        dataset = self._load_dataset()
        self.models = self._load_models()
        dataset = scPreprocess._tokenize_for_sc(dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        dataset = scPreprocess._select_inputs_with_source_class(dataset, self.trigger_source_labels)
        trigger_length = 10
        trigger_loc = 'start'
        dummy = self.models.tokenizer.pad_token_id
        dataset = scPreprocess._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_loc, dummy=dummy)
        dataset = scPreprocess._add_baseline_probabilities(dataset, self.models)
        dataset = scPreprocess.SCTriggeredDataset(dataset)
        batch_size = 16
        self.dataloader = DataLoader(dataset, batch_size=batch_size)

    def _load_dataset(self):
        model_filepath = self._prepend_current_script_path('../data/round9_sample_dataset/models/id-00000014/model.pt')
        scratch_filepath = '.tmp'
        return data_tools.load_examples(model_filepath, scratch_filepath)

    def _load_models(self):
        suspicious_model_filepath = self._prepend_current_script_path('../data/round9_sample_dataset/models/id-00000014/model.pt')
        clean_model_filepaths = [self._prepend_current_script_path('../data/round9_sample_dataset/models/id-00000002/model.pt')]
        tokenizer_filepath = self._prepend_current_script_path('../data/round9_sample_dataset/tokenizers/roberta-base.pt')
        return TriggerModels(suspicious_model_filepath, clean_model_filepaths, tokenizer_filepath, device=torch.device('cuda'))

    @staticmethod
    def _prepend_current_script_path(path):
        current_script_dirname = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_script_dirname, path)

    @torch.no_grad()
    def test_suspicious_loss(self):
        for batch in self.dataloader:
            all_logits = self.models(batch)
            loss = scLoss.calculate_sc_suspicious_loss(all_logits, self.trigger_target_label)
            break
        expected_loss = torch.tensor(4.9418, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    @torch.no_grad()
    def test_clean_loss(self):
        for batch in self.dataloader:
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            clean_agg_fn = torch.mean
            loss = scLoss.calculate_sc_clean_loss(all_logits, batch, self.trigger_target_label, clean_agg_fn)
            break
        expected_loss = torch.tensor(0.0098, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    @torch.no_grad()
    def test_sc_loss(self):
        for batch in self.dataloader:
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            clean_agg_fn = torch.mean
            loss = scLoss.calculate_sc_loss(all_logits, batch, self.trigger_target_label, clean_agg_fn)
            break
        expected_loss = torch.tensor(4.9516, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)