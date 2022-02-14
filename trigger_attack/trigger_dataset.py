import torch
from torch.utils.data import DataLoader
import datasets
from copy import deepcopy
from .trigger_models import TriggerModels
from . import trigger_initializer
from .trigger import Trigger


class TriggerDataset():

    task_name_to_preprocessing_fn = {
        'sc': sc_dataset_preprocessing,
        'ner': ner_dataset_preprocessing,
        'qa': qa_dataset_preprocessing,
    }

    def __init__(self,
                 original_dataset: datasets.arrow_dataset.Dataset,
                 task: str,
                 models: TriggerModels,
                 trigger: Trigger,
                 trigger_init_fn: str = 'embed_ch'):

        self.trigger = trigger

        self.task = task
        self.__verify_task()

        self.device = models.device
        self.original_dataset = deepcopy(original_dataset)
        self.dataset = self.__preprocess_dataset(models)

        self.reinitialize_trigger(models, trigger_init_fn)

    def __verify_task(self):
        assert self.task in list(TriggerDataset.task_name_to_preprocessing_fn.keys()), \
            f'task is undefined, try one of {list(TriggerDataset.task_name_to_preprocessing_fn.keys())}'

    def __preprocess_dataset(self, models):
        preprocessing_fn = self.task_name_to_preprocessing_fn[self.task]
        dataset = preprocessing_fn(self, models)
        return dataset

    def reinitialize_trigger(self, models, trigger_init_fn='embed_ch'):
        self.__verify_trigger_init_fn(trigger_init_fn)
        new_trigger_input_ids = self.__get_init_trigger(models, trigger_init_fn)
        self.insert_new_trigger(new_trigger_input_ids)

    def insert_new_trigger(self, new_trigger_input_ids):
        assert len(new_trigger_input_ids) == len(self.trigger.input_ids), f"trigger must have {len(self.trigger.input_ids)} token ids"
        self.trigger.input_ids = new_trigger_input_ids
        self.dataset.update_trigger(new_trigger_input_ids)

    @staticmethod
    def __verify_trigger_init_fn(trigger_init_fn):
        init_functions = trigger_initializer.trigger_init_names_to_fn
        assert trigger_init_fn in list(init_functions.keys()), \
            f'trigger init function is unsupported, try one of {list(init_functions.keys())}'

    def __get_init_trigger(self, models, trigger_init_fn):
        models.populate_most_changed_embeddings()
        init_functions = trigger_initializer.trigger_init_names_to_fn
        return init_functions[trigger_init_fn](self, models)


    def make_torch_dataloader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size)