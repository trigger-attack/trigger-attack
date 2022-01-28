from re import M
import unittest
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import os
from trigger_attack.trigger_models import TriggerModels
from transformers import logging
logging.set_verbosity_error()


class TestTriggerModels(unittest.TestCase):

    def setUp(self):
        current_script_dirname = os.path.dirname(os.path.abspath(__file__))
        scratch_tail = 'data/trigger_models'
        self.scratch_dirpath = os.path.join(current_script_dirname, scratch_tail)

        suspicious_model_filepath = os.path.join(self.scratch_dirpath, 'suspicious_model.pt')
        self._save_pretrained_model_in_scratch_space(suspicious_model_filepath)

        clean_model_filepath_1 = os.path.join(self.scratch_dirpath, 'clean_model_1.pt')
        clean_model_filepath_2 = os.path.join(self.scratch_dirpath, 'clean_model_2.pt')
        self._save_pretrained_model_in_scratch_space(clean_model_filepath_1)
        self._save_pretrained_model_in_scratch_space(clean_model_filepath_2)
        clean_model_filepaths = [clean_model_filepath_1, clean_model_filepath_2]

        tokenizer_filepath = os.path.join(self.scratch_dirpath, 'tokenizer.pt')
        self._save_tokenizer_in_scratch_space(tokenizer_filepath)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.models = TriggerModels(suspicious_model_filepath,
                                    clean_model_filepaths,
                                    tokenizer_filepath, 
                                    device=device)

    def test_init(self):
        self.assertTrue(self.models.suspicious_model is not None)
        self.assertTrue(self.models.clean_models is not None)
        self.assertTrue(self.models.tokenizer is not None)

    def test_most_changed_words(self):
        self.models.populate_most_changed_embeddings()
        self.assertTrue(self.models.most_changed_singletoken_words is not None)
        self.assertTrue(self.models.most_changed_multitoken_words is not None)

    
    def _save_pretrained_model_in_scratch_space(self, save_as):
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        torch.save(model, save_as)

    def _save_tokenizer_in_scratch_space(self, save_as):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        torch.save(tokenizer, save_as)

    def _test_forward(self):
        batch = {
            'input_ids': torch.tensor([0, 234, 1234, 34, 134, 324, 324, 234, 12, 10]),
            'attention_mask': torch.tensor([1]*10),
            'token_type_ids': torch.tensor([0]*10),
        }
        logits = self.models(batch)

    def tearDown(self):
        for f in os.listdir(self.scratch_dirpath):
            os.remove(os.path.join(self.scratch_dirpath, f))


if __name__ == '__main__':
    unittest.main(verbosity=3)