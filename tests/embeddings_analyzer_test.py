import unittest
import tools
from datasets.utils import set_progress_bar_enabled
import warnings
from trigger_attack.embeddings_analyzer import EmbeddingsAnalyzer


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

        dataset = tools.load_dataset(suspicious_model_filepath)
        trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        self.tokenizer = tools.load_tokenizer(tokenizer_filepath)

        # embeddings analyzer
        self.embeddings_analyzer = \
            EmbeddingsAnalyzer(trigger_models, self.tokenizer)

    def test_smallest_cosine_similarity(self):
        dissimilar_token_ids = \
            self.embeddings_analyzer._get_smallest_cosine_similarity()
        dissimilar_token_ids = \
            dissimilar_token_ids.to(self.embeddings_analyzer.device)
        top_ids_to_tokens = {}
        for top_id in dissimilar_token_ids:
            token = self.tokenizer.convert_ids_to_tokens([top_id])[0]
            top_ids_to_tokens[top_id] = token
        dissimilar_token_ids = \
            self.embeddings_analyzer._remove_non_words(top_ids_to_tokens)

        suffixes = \
            self.embeddings_analyzer._get_suffixes(dissimilar_token_ids)
        prefixes = \
            self.embeddings_analyzer\
                ._get_prefixes(dissimilar_token_ids, suffixes)
        return NotImplementedError

    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)
