import unittest
import os
import torch
from torch.utils.data import DataLoader
from trigger_attack.trigger_models import TriggerModels
from trigger_attack.loss_functions import sc as scLoss
from trigger_attack.preprocessing import sc as scPreprocess
from trigger_attack.trigger import Trigger
from trojai_submission import data_tools
import tools
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
        self.trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        tokenizer = tools.load_tokenizer(tokenizer_filepath)
        self.trigger_source_labels = [0]
        trigger = Trigger(torch.tensor([1]*10), location='start', source_labels=self.trigger_source_labels)
        self.preprocessor = scPreprocess.SCDatasetPreprocessor(
            dataset, trigger, self.trigger_models, tokenizer)

        dataset = self.preprocessor.preprocess_data()
        batch_size = 16
        self.dataloader = DataLoader(dataset, batch_size=batch_size)

        self.loss_fn = scLoss.SCLoss()
        self.target_label = 1


    def _load_dataset(self):
        model_filepath = self._prepend_current_script_path('../data/round9_sample_dataset/models/id-00000014/model.pt')
        scratch_filepath = '.tmp'
        return data_tools.load_examples(model_filepath, scratch_filepath)

    def _load_trigger_models(self):
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
            all_logits = self.trigger_models(batch)
            loss = self.loss_fn._calculate_suspicious_loss(all_logits, batch, self.target_label)
            break
        expected_loss = torch.tensor(4.9418, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    @torch.no_grad()
    def test_clean_loss(self):
        for batch in self.dataloader:
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.trigger_models.device, non_blocking=True)
            all_logits = self.trigger_models(batch)
            loss = self.loss_fn._calculate_clean_loss(all_logits, batch, self.target_label)
            break
        expected_loss = torch.tensor(0.0098, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    @torch.no_grad()
    def test_sc_loss(self):
        for batch in self.dataloader:
            batch['baseline_probabilities'] = batch['baseline_probabilities'].to(self.trigger_models.device, non_blocking=True)
            all_logits = self.trigger_models(batch)
            loss = self.loss_fn.calculate_loss(all_logits, batch, self.target_label)
            break
        expected_loss = torch.tensor(4.9516, device=loss.device)
        self.assertTrue(torch.isclose(expected_loss, loss, atol=1e-04))

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)
