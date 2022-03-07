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

class TestTriggerInitializer(unittest.TestCase):

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

        embeddings_analyzer = EmbeddingsAnalyzer(trigger_models, tokenizer)

        self.trigger_initializator = TriggerInitializator(
            tokenizer, trigger_models.device, embeddings_analyzer)

    def test_make_random_trigger(self):
        random_trigger = self.trigger_initializator._make_random_trigger(10)
        self.assertEqual(random_trigger.shape[0], 10)
        for num in random_trigger:
            self.assertTrue(0 <= num <= len(self.trigger_initializator.tokenizer)-1)

    def test_make_pad_trigger(self):
        pad_trigger = self.trigger_initializator._make_pad_trigger(5)
        self.assertEqual(pad_trigger.shape[0], 5)
        for p in pad_trigger:
            self.assertEqual(p, self.trigger_initializator.tokenizer.pad_token_id)

    def test_make_most_changed_embedding_trigger(self):
        embedding_trigger = self.trigger_initializator._make_most_changed_embedding_trigger(7)
        self.assertEqual(embedding_trigger.shape[0], 7)

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)