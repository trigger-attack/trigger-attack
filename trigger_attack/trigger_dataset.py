import torch
import datasets
import os
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
from .trigger_models import TriggerModels
from .preprocessing.sc import sc_dataset_preprocessing
from .preprocessing.ner import ner_dataset_preprocessing
from .preprocessing.qa import qa_dataset_preprocessing
from torch.utils.data import DataLoader
from .trigger_init_fn import trigger_init_names_to_fn


class TriggerDataset():
    
    task_name_to_preprocessing_fn = {
        'sc': sc_dataset_preprocessing,
        'ner': ner_dataset_preprocessing,
        'qa': qa_dataset_preprocessing,
    }

    def __init__(self, 
                original_dataset: datasets.arrow_dataset.Dataset, 
                models: TriggerModels, 
                trigger_length: int, 
                trigger_init_fn: str, 
                trigger_loc: str, 
                task: str,
                trigger_source_labels:list = None):
        
        self.original_dataset = original_dataset

        self.trigger = None
        self.trigger_length = trigger_length
        self.trigger_loc = trigger_loc
        self.trigger_source_labels = trigger_source_labels

        self.device = models.device
        self.task = task

        self._verify_init_args()
        self.dataset = self._preprocess_dataset(models)
        self.reinitialize_trigger(models, trigger_init_fn)

    def _verify_init_args(self):
        self._verify_task()

    def _verify_task(self):
        assert self.task in list(TriggerDataset.task_name_to_preprocessing_fn.keys()), \
            f'task is undefined, try one of {list(TriggerDataset.task_name_to_preprocessing_fn.keys())}'
    
    @staticmethod
    def _verify_trigger_init_fn(trigger_init_fn):
        assert trigger_init_fn in list(trigger_init_names_to_fn.keys()), \
            f'trigger init function is unsupported, try one of {list(trigger_init_names_to_fn.keys())}'
    
    def _preprocess_dataset(self, models):
        dataset = self.task_name_to_preprocessing_fn[self.task](self, models)
        return dataset
    
    def _get_init_trigger(self, models, trigger_init_fn):
        models.populate_most_changed_embeddings()
        return trigger_init_names_to_fn[trigger_init_fn](self, models)

    def insert_new_trigger(self, new_trigger):
        assert len(new_trigger) == self.trigger_length, f"trigger must have {self.trigger_length} token ids"
        self.trigger = new_trigger
        self.dataset.update_trigger(new_trigger)

    def reinitialize_trigger(self, models, trigger_init_fn='embed_ch'):
        self._verify_trigger_init_fn(trigger_init_fn)
        self.trigger = self._get_init_trigger(models, trigger_init_fn)
        self.dataset.update_trigger(self.trigger)

    def make_torch_dataloader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size)