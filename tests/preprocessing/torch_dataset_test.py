import unittest
import torch
from trigger_attack.preprocessing import qa
from trigger_attack.trigger import Trigger
import tools
import warnings
from datasets.utils import set_progress_bar_enabled


class TestQAPreprocessing(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore")
        set_progress_bar_enabled(False)

        suspicious_model_filepath = (
            '../data/round8_sample_dataset/models/id-00000000/model.pt')
        clean_models_filepaths = [
            '../data/round8_sample_dataset/models/id-00000018/model.pt'
        ]
        tokenizer_filepath = (
            '../data/round8_sample_dataset/tokenizers'
            '/tokenizer-roberta-base.pt')

        dataset = tools.load_dataset(suspicious_model_filepath)
        trigger_models = tools.load_trigger_models(suspicious_model_filepath,
                                                   clean_models_filepaths)
        tokenizer = tools.load_tokenizer(tokenizer_filepath)
        trigger = Trigger(torch.tensor([0]*10), location='both', source_labels=None)
        self.preprocessor = qa.QADatasetPreprocessor(dataset,
                                                     trigger,
                                                     trigger_models,
                                                     tokenizer)

    def test_update_trigger(self):
        dataset = self.preprocessor.preprocess_data()
        new_trigger = torch.tensor(list(range(len(self.preprocessor.trigger.input_ids))))
        dataset.update_trigger(new_trigger)
        input_ids = dataset[0]['input_ids']
        trigger_mask = dataset[0]['trigger_mask']
        self.assertTrue(torch.equal(input_ids[trigger_mask][:len(new_trigger)], new_trigger))


    def tearDown(self):
        return NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=3)
