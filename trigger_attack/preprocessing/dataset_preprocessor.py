import torch
from abc import ABC, abstractmethod
from copy import deepcopy


class datasetPreprocessor(ABC):
    def __init__(self,
                 dataset,
                 trigger,
                 trigger_models,
                 tokenizer,
                 agg_fn=torch.mean,
                 max_seq_length=384,
                 doc_stride=128):
        self.dataset = deepcopy(dataset)
        self.trigger = trigger
        self.trigger_models = trigger_models
        self.tokenizer = tokenizer
        self.agg_fn = agg_fn
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride

    def preprocess_data(self):
        new_dataset = deepcopy(self.dataset)
        new_dataset = self._tokenize(new_dataset)
        new_dataset = self._select_unique_inputs(new_dataset)
        new_dataset = self._select_inputs_with_source_class(new_dataset)
        new_dataset = self._insert_dummy(new_dataset)
        new_dataset = self._add_baseline_probabilities(new_dataset)
        new_dataset = self._package_into_torch_dataset(new_dataset)

        return new_dataset

    @abstractmethod
    def _tokenize(self, original_dataset):
        pass

    @torch.no_grad()
    def _select_unique_inputs(self, tokenized_dataset):
        input_ids = torch.tensor(tokenized_dataset['input_ids'])
        unique_ixs_ids = input_ids.unique(dim=0, return_inverse=True)[1]
        unique_ixs_ids = unique_ixs_ids.flatten()
        seen = set()
        unique_ixs = []
        for source_ix, target_ix in enumerate(unique_ixs_ids):
            if target_ix.item() not in seen:
                seen.add(target_ix.item())
                unique_ixs.append(source_ix)
        return tokenized_dataset.select(unique_ixs)

    @abstractmethod
    def _select_inputs_with_source_class(self, unique_inputs_dataset):
        pass

    @abstractmethod
    def _insert_dummy(self, source_class_dataset):
        pass

    @abstractmethod
    def _add_baseline_probabilities(self, dataset_with_dumy):
        pass

    @abstractmethod
    def _package_into_torch_dataset(self, dataset_with_baseline_probabilities):
        pass

    def get_max_seq_length(self):
        max_seq_length = min(self.tokenizer.model_max_length,
                             self.max_seq_length)
        if 'mobilebert' in self.tokenizer.name_or_path:
            name = self.tokenizer.name_or_path.split('/')[1]
            max_seq_length = self.tokenizer.max_model_input_sizes[name]
        return max_seq_length
