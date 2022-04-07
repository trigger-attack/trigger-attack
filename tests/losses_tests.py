import unittest
import torch
from torch.utils.data import DataLoader
from datasets.utils import set_progress_bar_enabled
import warnings

from trigger_attack.trigger import Trigger

from trigger_attack.preprocessing import ner as nerPreprocess
from trigger_attack.preprocessing import sc as scPreprocess
from trigger_attack.preprocessing import qa as qaPreprocess

from trigger_attack.loss_functions import sc as scLoss
from trigger_attack.loss_functions import ner as nerLoss
from trigger_attack.loss_functions import qa as qaLoss

import tools
import constants


class LossTest(unittest.TestCase):

    def setUp(self):
        if not hasattr(self, 'expected_losses'):
            self.skipTest('parent class')

        warnings.filterwarnings("ignore")
        set_progress_bar_enabled(False)

        dataset = \
            tools.load_dataset(
                self.testing_data_paths.suspicious_model_filepath)
        self.trigger_models = tools.load_trigger_models(
            self.testing_data_paths.suspicious_model_filepath,
            self.testing_data_paths.clean_models_filepaths)
        tokenizer = \
            tools.load_tokenizer(
                self.testing_data_paths.tokenizer_filepath)
        self.preprocessor = self.preprocessor_class(
            dataset, self.trigger, self.trigger_models, tokenizer)

        dataset = self.preprocessor.preprocess_data()
        self.dataloader = \
            DataLoader(dataset, batch_size=self.batch_size)

        self.loss_fn = self.loss_class()

    @torch.no_grad()
    def test_suspicious_loss(self):
        for batch in self.dataloader:
            device = self.trigger_models.device
            if 'valid_mask' in batch:
                batch['valid_mask'] = \
                    batch['valid_mask'].to(device, non_blocking=True)
            batch['baseline_probabilities'] = \
                batch['baseline_probabilities'].to(device, non_blocking=True)
            all_logits = self.trigger_models(batch)
            loss = self.loss_fn\
                ._calculate_suspicious_loss(
                    all_logits, batch, self.target_label)
            break
        self.assertAlmostEqual(
            self.expected_losses['suspicious'], loss.item(), places=3)

    @torch.no_grad()
    def test_clean_loss(self):
        for batch in self.dataloader:
            device = self.trigger_models.device
            if 'valid_mask' in batch:
                batch['valid_mask'] = \
                    batch['valid_mask'].to(device, non_blocking=True)
            batch['baseline_probabilities'] = \
                batch['baseline_probabilities'].to(device, non_blocking=True)
            all_logits = self.trigger_models(batch)
            loss = self.loss_fn\
                ._calculate_clean_loss(all_logits, batch, self.target_label)
            break
        self.assertAlmostEqual(
            self.expected_losses['clean'], loss.item(), places=3)

    @torch.no_grad()
    def test_loss(self):
        for batch in self.dataloader:
            device = self.trigger_models.device
            if 'valid_mask' in batch:
                batch['valid_mask'] = \
                    batch['valid_mask'].to(device, non_blocking=True)
            batch['baseline_probabilities'] = \
                batch['baseline_probabilities'].to(device, non_blocking=True)
            all_logits = self.trigger_models(batch)
            loss = self.loss_fn\
                .calculate_loss(all_logits, batch, self.target_label)
            break
        self.assertAlmostEqual(
            self.expected_losses['total'], loss.item(), places=3)


class TestSCLosses(LossTest):

    testing_data_paths = constants.sc_paths
    source_label = [0]
    target_label = [1]
    trigger = Trigger(
        torch.tensor([1]*10), location='start', source_labels=source_label)
    batch_size = 16
    expected_losses = {
        'suspicious': 4.9418,
        'clean': 0.0098,
        'total': 4.9516
    }
    preprocessor_class = scPreprocess.SCDatasetPreprocessor
    loss_class = scLoss.SCLoss


class TestNERLosses(LossTest):

    testing_data_paths = constants.ner_paths
    source_label = [3, 4]
    target_label = [5, 6]
    trigger = Trigger(torch.tensor(
        [1]*10), 'both', source_labels=source_label)
    batch_size = 16
    expected_losses = {
        'clean': 0.0,
        'suspicious': 4.2432,
        'total': 4.2432
    }
    preprocessor_class = nerPreprocess.NERDatasetPreprocessor
    loss_class = nerLoss.NERLoss


class TestQALosses(LossTest):

    testing_data_paths = constants.qa_paths
    target_label = None
    trigger = Trigger(
        torch.tensor([1]*10), location='both', source_labels=None)
    batch_size = 16
    expected_losses = {
        'suspicious': 12.9605,
        'clean': 2.5116e-05,
        'total': 12.9605
    }
    preprocessor_class = qaPreprocess.QADatasetPreprocessor
    loss_class = qaLoss.QALoss


if __name__ == '__main__':
    unittest.main(verbosity=3)
