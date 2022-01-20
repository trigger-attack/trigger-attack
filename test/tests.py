from trojan import data_downloader
import unittest
from transformers import DistilBertTokenizer, DistilBertModel
from trigger_inversion_models import TriggerInversionModels
import torch
import os
from transformers import logging
logging.set_verbosity_error()

class TestDataDownloader(unittest.TestCase):

    def test_download(self):
        self.assertTrue()

    def setUp(self):
        data_downloader.download_trojai_dataset(6, 'train')

# class TestTrojaiTools(unittest.TestCase):

class TestTriggerInversionModels(unittest.TestCase):

    scratch_dirpath = 'tests_scratch/trigger_inversion_models'

    def test_init(self):
        models = self.set_up()
        self.assertTrue(models.suspicious_model is not None)
        self.assertTrue(models.clean_models is not None)
        self.assertTrue(models.tokenizer is not None)
        self.tear_down()

    def test_most_changed_words(self):
        models = self.set_up()
        models.populate_most_changed_embeddings()
        self.assertTrue(models.most_changed_singletoken_words is not None)
        self.assertTrue(models.most_changed_multitoken_words is not None)
        self.tear_down()

    def set_up(self):
        suspicious_model_filepath = os.path.join(self.scratch_dirpath, 'suspicious_model.pt')
        self.save_pretrained_model_in_scratch_space(suspicious_model_filepath)

        clean_model_filepath_1 = os.path.join(self.scratch_dirpath, 'clean_model_1.pt')
        clean_model_filepath_2 = os.path.join(self.scratch_dirpath, 'clean_model_2.pt')
        self.save_pretrained_model_in_scratch_space(clean_model_filepath_1)
        self.save_pretrained_model_in_scratch_space(clean_model_filepath_2)
        clean_model_filepaths = [clean_model_filepath_1, clean_model_filepath_2]

        tokenizer_filepath = os.path.join(self.scratch_dirpath, 'tokenizer.pt')
        self.save_tokenizer_in_scratch_space(tokenizer_filepath)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        models = TriggerInversionModels(suspicious_model_filepath,
                                        clean_model_filepaths,
                                        tokenizer_filepath, 
                                        device=device)

        return models
    
    def save_pretrained_model_in_scratch_space(self, save_as):
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        torch.save(model, save_as)

    def save_tokenizer_in_scratch_space(self, save_as):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        torch.save(tokenizer, save_as)

    def tear_down(self):
        for f in os.listdir(self.scratch_dirpath):
            os.remove(os.path.join(self.scratch_dirpath, f))


if __name__ == '__main__':
    unittest.main(verbosity=3)