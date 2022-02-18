import unittest
import torch
from trigger_attack.trigger_reconstructor import TriggerReconstructor
import tools
from datasets.utils import set_progress_bar_enabled
from trigger_attack.preprocessing import qa as qaPreprocess
from trigger_attack.loss_functions import qa as qaLoss
import warnings
from trigger_attack.embeddings_analyzer import EmbeddingsAnalyzer
from trigger_attack.trigger import Trigger
from trigger_attack.trigger_initializer import TriggerInitializator
import numpy as np
import random


class TestTriggerReconstuctor(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        random.seed(1)
        warnings.filterwarnings("ignore")
        set_progress_bar_enabled(False)

        suspicious_model_filepath = (
            'data/round8_sample_dataset/models/id-00000000/model.pt')
        clean_models_filepaths = [
            'data/round8_sample_dataset/models/id-00000018/model.pt'
        ]
        tokenizer_filepath = (
            'data/round8_sample_dataset/tokenizers'
            '/tokenizer-roberta-base.pt')

        dataset = tools.load_dataset(suspicious_model_filepath)
        trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        tokenizer = tools.load_tokenizer(tokenizer_filepath)
        trigger_length = 10
        trigger = Trigger(torch.tensor([1]*trigger_length), location='question', source_labels=None)
        preprocessor = qaPreprocess.QADatasetPreprocessor(
            dataset, trigger, trigger_models, tokenizer)

        loss_fn = qaLoss.QALoss()

        embeddings_analyzer = EmbeddingsAnalyzer(trigger_models, tokenizer)

        trigger_initializator = TriggerInitializator(
            tokenizer, trigger_models.device, embeddings_analyzer)

        self.trigger_reconstructor = TriggerReconstructor(
            trigger_initializator, preprocessor, loss_fn)

        new_trigger = trigger_initializator.make_initial_trigger(
            trigger_length, 'embedding_change')
        self.trigger_reconstructor.dataloader.dataset.update_trigger(new_trigger.long())

    def test_calc_loss(self):
        trigger_target = 'cls'
        loss_value = self.trigger_reconstructor._calculate_loss(trigger_target, extract_embedding_gradients=True)
        expected = torch.tensor(0.9121)
        self.assertTrue(torch.allclose(loss_value, expected, atol=1e-3))

    def test_get_candidates(self):
        trigger_target = 'cls'
        _ = self.trigger_reconstructor._calculate_loss(trigger_target, extract_embedding_gradients=True)
        num_candidates_per_token=3
        candidates = self.trigger_reconstructor._get_candidates(num_candidates_per_token)
        expected = torch.tensor([45201, 25613, 44354])
        self.assertTrue(torch.allclose(candidates.cpu()[0], expected, atol=1e-3))

    def test_pick_best_candidate(self):
        trigger_target = 'cls'
        loss_value = self.trigger_reconstructor._calculate_loss(trigger_target, extract_embedding_gradients=True)
        num_candidates_per_token=3
        candidates = self.trigger_reconstructor._get_candidates(num_candidates_per_token)
        best_candidate = self.trigger_reconstructor._pick_best_candidate(loss_value, candidates, trigger_target, beam_size=1)
        expected = torch.tensor([44354,  1134, 42673, 46615,  3266,  3266, 28481,  1407, 24445, 40130])
        self.assertTrue(torch.equal(best_candidate['input_ids'].cpu(), expected))

    def test_trigger_reconstruction(self):
        trigger_target = 'cls'
        num_candidates_per_token = 3
        best_candidate = {
            'input_ids': None,
            'loss': 100
        }
        for _ in range(3):
            temp_candidate = self.trigger_reconstructor.reconstruct_trigger(trigger_target, num_candidates_per_token)
            if temp_candidate['loss'] < best_candidate['loss']:
                best_candidate = temp_candidate
        
        expected = torch.tensor(0.0012)
        self.assertLessEqual(best_candidate['loss'], expected)

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)
