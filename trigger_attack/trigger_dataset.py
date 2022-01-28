import torch
import datasets
import os
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
from .preprocessing.sc import sc_dataset_preprocessing
from .preprocessing.ner import ner_dataset_preprocessing
from .preprocessing.qa import qa_dataset_preprocessing
from .trigger_init_fn import trigger_init_names_to_fn


class TriggerDataset():
    
    task_name_to_preprocessing_fn = {
        'sc': sc_dataset_preprocessing,
        'ner': ner_dataset_preprocessing,
        'qa': qa_dataset_preprocessing,
    }

    def __init__(self, original_dataset, models, trigger_length, trigger_init_fn, trigger_loc, task, scratch_dirname='.tmp'):
        
        self.original_dataset = original_dataset
        self.dataloader = None

        self.trigger = None
        self.trigger_length = trigger_length
        self.trigger_loc = trigger_loc
        self.trigger_init_fn = trigger_init_fn

        self.device = models.device
        self.task = task

        self._verify_init_args()
        self.dataset = self._preprocess_dataset(scratch_dirname, original_dataset, models.tokenizer, trigger_length)
        self.trigger = self._get_init_trigger(models)
        self.dataset.insert_new_trigger(self.trigger)

    def _verify_init_args(self):
        assert self.task in list(TriggerDataset.task_name_to_preprocessing_fn.keys()), \
            f'task is undefined, try one of {list(TriggerDataset.task_name_to_preprocessing_fn.keys())}'
    
    def _preprocess_dataset(self, scratch_dirname, original_dataset, tokenizer, trigger_length):
        dataset = self.task_name_to_preprocessing_fn[self.task](original_dataset, tokenizer, trigger_length, self.trigger_loc)
        if not os.path.isdir(scratch_dirname):
            os.mkdir(scratch_dirname)
        dataset_filepath = os.path.join(scratch_dirname, 'dataset.csv')
        dataset.to_csv(dataset_filepath)
        dataset = datasets.load_dataset('csv', data_files=dataset_filepath, split='train', streaming=True)
        return dataset
    
    def _get_init_trigger(self, models):
        models.populate_most_changed_embeddings()
        return trigger_init_names_to_fn[self.trigger_init_fn](self, models)

    def insert_new_trigger(self, new_trigger):
        assert len(new_trigger) == self.trigger_length, f"trigger must have {self.trigger_length} token ids"
        self.trigger = new_trigger
        self.dataloader.dataset.update_trigger(new_trigger)
