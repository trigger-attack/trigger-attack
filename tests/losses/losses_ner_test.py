import unittest
import os
import torch
from torch.utils.data import DataLoader
from trigger_attack.trigger_models import TriggerModels
from trigger_attack.preprocessing import ner, tools
from trigger_attack.loss_functions import ner as nerLoss
from trojai_submission import data_tools
import warnings
warnings.filterwarnings("ignore")


class TestSCPreprocessing(unittest.TestCase):

    def setUp(self):
        self.dataset = self._load_dataset()
        self.models = self._load_models()
        dataset = ner._tokenize_for_ner(self.dataset, self.models.tokenizer)
        dataset = tools._select_unique_inputs(dataset)
        trigger_length = 10
        trigger_source_labels = [3, 4]
        self.trigger_target_labels = [5, 6]
        dummy = self.models.tokenizer.pad_token_id
        dataset = ner._initialize_dummy_trigger(dataset, self.models.tokenizer, trigger_length, trigger_source_labels, dummy=dummy)
        dataset = ner._add_baseline_probabilities(dataset, self.models)
        dataset = ner.NERTriggeredDataset(dataset)
        batch_size = 16
        self.dataloader = DataLoader(dataset, batch_size=batch_size)

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

    @torch.no_grad()
    def test_suspicious_loss(self):
        for batch in self.dataloader:
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            loss = nerLoss.calculate_ner_suspicious_loss(all_logits, self.trigger_target_labels, batch)
            break
        expected_loss = torch.tensor(4.2433, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    @torch.no_grad()
    def test_clean_loss(self):
        for batch in self.dataloader:
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            clean_agg_fn = torch.mean
            loss = nerLoss.calculate_ner_clean_loss(all_logits, batch, self.trigger_target_labels, clean_agg_fn)
            break
        expected_loss = torch.tensor(0., device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    def test_ner_loss(self):
        for batch in self.dataloader:
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.models.device, non_blocking=True)
            all_logits = self.models(batch)
            clean_agg_fn = torch.mean
            loss = nerLoss.calculate_ner_loss(all_logits, batch, self.trigger_target_labels, clean_agg_fn)
            break
        expected_loss = torch.tensor(4.2433, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)