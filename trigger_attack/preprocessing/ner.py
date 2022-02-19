import torch
from copy import deepcopy
import numpy as np
from .torch_triggered_dataset import TorchTriggeredDataset
from .dataset_preprocessor import datasetPreprocessor


class NERDatasetPreprocessor(datasetPreprocessor):
    def __init__(self, dataset, trigger, trigger_models, tokenizer):
        super().__init__(dataset, trigger, trigger_models, tokenizer)

    def _tokenize(self, original_dataset):
        def _tokenize_for_ner_helper(examples):
            tokenized_inputs = self.tokenizer(
                examples['tokens'],
                padding=True,
                truncation=True,
                is_split_into_words=True,
                max_length=self.get_max_seq_length(),
                return_token_type_ids=True)

            word_ids = []
            for i in range(len(examples['ner_tags'])):
                word_ids.append(tokenized_inputs.word_ids(i))
            labels, label_mask = [], []
            previous_word_idx = None

            for i, sentence in enumerate(word_ids):
                temp_labels, temp_mask = [], []
                for word_idx in sentence:
                    if word_idx is not None:
                        cur_label = examples['ner_tags'][i][word_idx]
                    if word_idx is None:
                        temp_labels.append(-100)
                        temp_mask.append(0)
                    elif word_idx != previous_word_idx:
                        temp_labels.append(cur_label)
                        temp_mask.append(1)
                    else:
                        temp_labels.append(-100)
                        temp_mask.append(0)
                    previous_word_idx = word_idx
                labels.append(temp_labels)
                label_mask.append(temp_mask)
            result = {
                'input_ids': tokenized_inputs['input_ids'],
                'attention_mask': tokenized_inputs['attention_mask'],
                'token_type_ids': tokenized_inputs['token_type_ids'],
                'label': labels,
            }
            return result
        tokenized_dataset = original_dataset.map(
            _tokenize_for_ner_helper,
            batched=True,
            num_proc=1,
            remove_columns=original_dataset.column_names,
            keep_in_memory=True)
        return tokenized_dataset

    def _select_inputs_with_source_class(self, unique_inputs_dataset):
        return unique_inputs_dataset

    def _insert_dummy(self, unique_inputs_dataset):
        dataset_with_dumy = unique_inputs_dataset.map(
            self.__insert_trigger_helper,
            batched=True,
            num_proc=1,
            remove_columns=unique_inputs_dataset.column_names,
            keep_in_memory=True)
        dataset_with_dumy.set_format('torch', output_all_columns=True)

        return dataset_with_dumy

    def __insert_trigger_helper(self, examples):
        dummy = self.tokenizer.pad_token_id
        result = {
            'input_ids': [],
            'attention_mask': [],
            'attention_mask_without_trigger': [],
            'token_type_ids': [],
            'label': [],
            'trigger_source_loc': [],
            'trigger_mask': []
        }
        labels = np.array(examples['label'])
        source_labels_mask = np.isin(labels, self.trigger.source_labels)
        locations_with_source_label = np.argwhere(source_labels_mask)

        def expand_to_trigger_length(num):
            trigger_length = len(self.trigger.input_ids)
            return torch.tensor([num]*trigger_length)

        def insert(insert, src, idx):
            torch_src = torch.tensor(src)
            new_var = torch.cat([torch_src[:idx], insert, torch_src[idx:]])
            return new_var

        trigger_length = len(self.trigger.input_ids)
        for row, col in locations_with_source_label:
            trigger_input_ids = expand_to_trigger_length(dummy)
            old_input_ids = examples['input_ids'][row]
            expanded_trigger_ids = insert(trigger_input_ids,
                                          old_input_ids, col)
            result['input_ids'].append(expanded_trigger_ids)

            trigger_attention_mask = expand_to_trigger_length(1)
            old_attention_mask = examples['attention_mask'][row]
            new_attention_mask = insert(trigger_attention_mask,
                                        old_attention_mask, col)
            result['attention_mask'].append(new_attention_mask)

            trigger_attention_mask = expand_to_trigger_length(0)
            expanded_attention_mask = insert(trigger_attention_mask,
                                             old_attention_mask, col)
            result['attention_mask_without_trigger']\
                .append(expanded_attention_mask)

            trigger_token_ids = expand_to_trigger_length(0)
            old_token_type_ids = examples['token_type_ids'][row]
            expanded_token_ids = insert(trigger_token_ids,
                                        old_token_type_ids, col)
            result['token_type_ids'].append(expanded_token_ids)

            trigger_label = expand_to_trigger_length(-100)
            old_labels = examples['label'][row]
            expanded_label = insert(trigger_label,
                                    old_labels, col)
            result['label'].append(expanded_label)

            result['trigger_source_loc'].append(col + trigger_length)

            trigger_ones = torch.tensor([1]*trigger_length)
            trigger_mask = torch.zeros_like(torch.tensor(old_attention_mask))
            expanded_trigger_mask = insert(trigger_ones,
                                           trigger_mask, col)
            result['trigger_mask'].append(expanded_trigger_mask)

        return result

    def _add_baseline_probabilities(self, dataset_with_dummy):
        dataset_with_baseline_probabilities = dataset_with_dummy.map(
            self.__add_baseline_probabilities_helper,
            batched=True,
            num_proc=1,
            keep_in_memory=True,
            batch_size=20)
        return dataset_with_baseline_probabilities

    @torch.no_grad()
    def __add_baseline_probabilities_helper(self, original_batch):
        batch = deepcopy(original_batch)
        batch['attention_mask'] = batch['attention_mask_without_trigger']
        all_logits = self.trigger_models(batch)
        suspicious_logits = all_logits['suspicious']['logits']
        suspicious_probabilities = self._get_probabilitites(
                                            suspicious_logits,
                                            batch['trigger_source_loc'])
        probabilities = [suspicious_probabilities]
        for model_output in all_logits['clean']:
            clean_logits = model_output['logits']
            clean_probabilities = self._get_probabilitites(
                                            clean_logits,
                                            batch['trigger_source_loc'])
            probabilities.append(clean_probabilities)
        probabilities = torch.stack(probabilities)
        agg_baseline_probabilitites = self.agg_fn(probabilities, dim=0)
        original_batch['baseline_probabilities'] = agg_baseline_probabilitites
        original_batch = {k: v.detach().cpu().numpy() for k, v in original_batch.items()}
        return original_batch

    @staticmethod
    def _get_probabilitites(logits, trigger_source_loc):
        incrementing_arr = torch.arange(len(logits))
        source_loc_logits = logits[incrementing_arr, trigger_source_loc]
        scores = torch.exp(source_loc_logits)
        probs = scores/torch.sum(scores, dim=1, keepdim=True)
        return probs

    def _package_into_torch_dataset(self, dataset_with_baseline_probabilities):
        return self.NERTriggeredDataset(dataset_with_baseline_probabilities, 
                                        len(self.trigger.input_ids))

    class NERTriggeredDataset(TorchTriggeredDataset):

        def __init__(self, dataset, trigger_length):
            super().__init__(dataset, trigger_length)
            self.label = dataset['trigger_source_loc'].clone().detach().long()

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            sample['trigger_source_loc'] = self.label[idx]
            return sample