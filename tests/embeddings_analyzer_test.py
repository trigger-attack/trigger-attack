from audioop import mul
from sys import prefix
import unittest
import torch
from trigger_attack.trigger_reconstructor import TriggerReconstructor
import tools as tools
from datasets.utils import set_progress_bar_enabled
from trigger_attack.preprocessing import qa as qaPreprocess
from trigger_attack.loss_functions import qa as qaLoss
import warnings
from trigger_attack.embeddings_analyzer import EmbeddingsAnalyzer
from trigger_attack.trigger import Trigger
from trigger_attack.trigger_initializer import TriggerInitializator


class TestTriggerReconstuctor(unittest.TestCase):

    def setUp(self):
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

        # dataset = tools.load_dataset(suspicious_model_filepath)
        trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        self.tokenizer = tools.load_tokenizer(tokenizer_filepath)

        # embeddings analyzer
        self.embeddings_analyzer = EmbeddingsAnalyzer(trigger_models, self.tokenizer)

    def test_remove_non_words(self):
        # token such as #_bb#, #0bb# also works
        top_ids_to_tokens = {1:'##', 2:'**', 3:'sub', 4:'^a', 5:'#bb#'}
        result = self.embeddings_analyzer._remove_non_words(top_ids_to_tokens)
        self.assertEqual(result, [3, 5])

    def test_get_suffixes(self):
        dissimilar_tokens = ['##', 'sub', 'se', 'lib']
        dissimilar_token_ids = self.tokenizer.convert_tokens_to_ids(dissimilar_tokens)
        suffixes = self.embeddings_analyzer._get_suffixes(dissimilar_token_ids)
        self.assertEqual(suffixes[0], dissimilar_token_ids[0])

    def test_get_prefixes(self):
        dissimilar_tokens = ['##', 'sub', 'se', 'lib']
        dissimilar_token_ids = self.tokenizer.convert_tokens_to_ids(dissimilar_tokens)
        suffixes = self.embeddings_analyzer._get_suffixes(dissimilar_token_ids)
        prefixes = self.embeddings_analyzer._get_prefixes(dissimilar_token_ids, suffixes)
        self.assertEqual(prefixes, dissimilar_token_ids[1:])

    def test_get_multitoken_words(self):
        token_ids = self.tokenizer.encode("extraordinarily")
        suffixes = token_ids[2:4]
        prefixes = [token_ids[1]]
        multitoken_words = self.embeddings_analyzer._get_multitoken_words(prefixes, suffixes)
        self.assertEqual(multitoken_words[0], token_ids[1:4])

    def test_populate_most_changed_embeddings(self):
        # test length of singletoken and multitoken is not zero
        self.embeddings_analyzer.populate_most_changed_embeddings()
        self.assertTrue(self.embeddings_analyzer.singletoken_options is not None)
        self.assertTrue(self.embeddings_analyzer.multitoken_options is not None)
    
    def test_options_are_not_populated(self):
        singletoken_option = self.embeddings_analyzer.singletoken_options
        multitoken_option = self.embeddings_analyzer.multitoken_options
        self.embeddings_analyzer.singletoken_options = ['1', '2', '3']
        self.embeddings_analyzer.multitoken_options = None
        self.assertEqual(self.embeddings_analyzer._options_are_not_populated(), True)
        self.embeddings_analyzer.singletoken_options = singletoken_option
        self.embeddings_analyzer.multitoken_options = multitoken_option

    def test_smallest_cosine_similarity(self):
        suspicious_embedding = self.embeddings_analyzer.suspicious_embeddings
        clean_embedding = self.embeddings_analyzer.avg_clean_embeddings
        self.embeddings_analyzer.suspicious_embeddings = torch.tensor([[1., 1., 2.], [1., 1., 4.], [1., 1., 6.]])
        self.embeddings_analyzer.avg_clean_embeddings = torch.tensor([[1., 1., 1.]])
        dissimilar_token_ids = self.embeddings_analyzer._get_smallest_cosine_similarity(k=3)
        self.assertEqual(dissimilar_token_ids.tolist(), [2, 1, 0])
        self.embeddings_analyzer.suspicious_embeddings = suspicious_embedding
        self.embeddings_analyzer.avg_clean_embeddings = clean_embedding

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)