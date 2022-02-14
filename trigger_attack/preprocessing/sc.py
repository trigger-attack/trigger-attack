import torch
from copy import deepcopy
import numpy as np
from .torch_triggered_dataset import TorchTriggeredDataset
from .dataset_preprocessor import datasetPreprocessor


class SCDatasetPreprocessor(datasetPreprocessor):
    def __init__(self, dataset, trigger, trigger_models, tokenizer):
        super().__init__(dataset, trigger, trigger_models, tokenizer)

    def _tokenize(self, original_dataset):
        max_seq_length = self.get_max_seq_length()

        def prepare_train_features(examples):
            tokenized_examples = self.tokenizer(
                examples['data'],
                truncation=True,
                max_length=max_seq_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_token_type_ids=True)
            labels = np.array(examples['label'])
            mapping = tokenized_examples['overflow_to_sample_mapping']
            tokenized_examples['label'] = labels[mapping]
            return tokenized_examples

        tokenized_dataset = original_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=1,
            remove_columns=original_dataset.column_names,
            keep_in_memory=True)

        cols_to_remove = ['overflow_to_sample_mapping', 'offset_mapping']
        tokenized_dataset = tokenized_dataset.remove_columns(cols_to_remove)
        return tokenized_dataset

    def _select_inputs_with_source_class(self, unique_inputs_dataset):
        if self.trigger.source_labels is None:
            return unique_inputs_dataset
        else:
            labels = np.array(unique_inputs_dataset['label'])
            label_mask = np.isin(labels, self.trigger.source_labels)
            rows_with_source_label = np.argwhere(label_mask)[:, 0]
            unique_rows = np.unique(rows_with_source_label)
            return unique_inputs_dataset.select(unique_rows)

    def _insert_dummy(self, unique_inputs_dataset):
        dummy = self.tokenizer.pad_token_id

        def _initialize_dummy_trigger_helper(examples):

            result = {k: torch.tensor(v) for k, v in examples.items()}

            def _find_insertion_location(trigger_loc):
                if trigger_loc == 'start':
                    insertion_ixs = 1
                elif trigger_loc == 'middle':
                    insertion_ixs = self.get_max_seq_length()//2
                elif trigger_loc == 'end':
                    insertion_ixs = self.get_max_seq_length()-1
                else:
                    return NotImplementedError
                return insertion_ixs

            insertion_ixs = _find_insertion_location(self.trigger.location)

            def _insert(insertion_ixs, base, insert):
                return torch.cat([base[:, :insertion_ixs],
                                  insert,
                                  base[:, insertion_ixs:]], 1)

            def _expand_tensor(tensor, num_rows):
                return tensor.unsqueeze(0).repeat(num_rows, 1)

            num_examples = len(examples['input_ids'])
            trigger_length = len(self.trigger.input_ids)

            expanded_dummy = torch.tensor([dummy]*trigger_length)
            trigger_input_ids = _expand_tensor(expanded_dummy, num_examples)

            expanded_ones = torch.tensor([1]*trigger_length)
            trigger_attention = _expand_tensor(expanded_ones, num_examples)

            expanded_zeros = torch.tensor([0]*trigger_length)
            token_type_ids = _expand_tensor(expanded_zeros, num_examples)

            temp_attn_mask = deepcopy(result['attention_mask'])
            zeros = torch.zeros_like(result['attention_mask'])
            result['input_ids'] = _insert(insertion_ixs,
                                          result['input_ids'],
                                          trigger_input_ids)
            result['attention_mask'] = _insert(insertion_ixs,
                                               result['attention_mask'],
                                               trigger_attention)
            result['token_type_ids'] = _insert(insertion_ixs,
                                               result['token_type_ids'],
                                               token_type_ids)
            result['attention_mask_without_trigger'] = _insert(insertion_ixs,
                                                               temp_attn_mask,
                                                               token_type_ids)
            result['trigger_mask'] = _insert(insertion_ixs,
                                             zeros,
                                             deepcopy(trigger_attention))

            result = {k: v.tolist() for k, v in result.items()}

            return result

        dataset_with_dummy = unique_inputs_dataset.map(
            _initialize_dummy_trigger_helper,
            batched=True,
            num_proc=1,
            remove_columns=unique_inputs_dataset.column_names,
            keep_in_memory=True)
        dataset_with_dummy.set_format('torch',
                                      columns=dataset_with_dummy.column_names)

        return dataset_with_dummy

    def _add_baseline_probabilities(self, dataset_with_dummy):
        dataset = dataset_with_dummy.map(
            self._add_baseline_probabilities_helper,
            batched=True,
            num_proc=1,
            keep_in_memory=True,
            batch_size=20)
        return dataset

    @torch.no_grad()
    def _add_baseline_probabilities_helper(self, batch):
        modified_batch = deepcopy(batch)
        ignore_attn = modified_batch['attention_mask_without_trigger']
        modified_batch['attention_mask'] = ignore_attn
        all_logits = self.trigger_models(modified_batch)
        suspicious_logits = all_logits['suspicious']['logits']
        probabilities = [self._get_probabilitites(suspicious_logits)]
        for output in all_logits['clean']:
            clean_logits = output['logits']
            probabilities += [self._get_probabilitites(clean_logits)]
        probabilities = torch.stack(probabilities)
        batch['baseline_probabilities'] = self.agg_fn(probabilities, dim=0)
        batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
        return batch

    @staticmethod
    def _get_probabilitites(logits):
        scores = torch.exp(logits)
        probs = scores/torch.sum(scores, dim=1, keepdim=True)
        return probs

    def _package_into_torch_dataset(self, dataset_with_baseline_probabilities):
        return self.SCTriggeredDataset(dataset_with_baseline_probabilities,
                                       len(self.trigger.input_ids))

    class SCTriggeredDataset(TorchTriggeredDataset):

        def __init__(self, huggingface_dataset, trigger_length):
            super().__init__(huggingface_dataset, trigger_length)
            self.label = huggingface_dataset['label'].clone().detach().long()

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            sample['label'] = self.label[idx]
            return sample
