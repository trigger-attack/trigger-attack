import torch
from trigger_attack.preprocessing import tools
from copy import deepcopy
import numpy as np

def sc_dataset_preprocessing(trigger_dataset, models):
    _check_valid_trigger_loc(trigger_dataset.trigger_loc)
    dataset = _tokenize_for_sc(trigger_dataset.original_dataset, models.tokenizer)
    dataset = tools._select_unique_inputs(dataset)
    dataset = _select_inputs_with_source_class(dataset, trigger_dataset.trigger_source_labels)
    dataset = _initialize_dummy_trigger(dataset, models.tokenizer, trigger_dataset.trigger_length, trigger_dataset.trigger_loc)
    dataset = _add_baseline_probabilities(dataset, models)
    dataset = tools.SCTriggeredDataset(dataset)

    return dataset

def _check_valid_trigger_loc(trigger_loc):
    valid_trigger_locations = ['start', 'middle', 'end']
    error_msg = f"Unsupported trigger location, please pick one of {valid_trigger_locations}"
    assert trigger_loc in valid_trigger_locations, error_msg

def _tokenize_for_sc(dataset, tokenizer, doc_stride=128):
    '''
    Tokenization should return a max sequence length as specified in tools._get_max_seq_length
    Tokenizer output must be padded and truncated in order to be able to convert it to torch.tensor
    Also make sure to return token_type_ids
    '''
    max_seq_length = tools._get_max_seq_length(tokenizer)
    def prepare_train_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
        
        tokenized_examples = tokenizer(
            examples['data'],
            truncation=True,
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        tokenized_examples['label'] = np.array(examples['label'])[tokenized_examples['overflow_to_sample_mapping']]
        return tokenized_examples
        
    dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=2,
        remove_columns=dataset.column_names,
        keep_in_memory=True)

    dataset = dataset.remove_columns(['overflow_to_sample_mapping', 'offset_mapping'])
    return dataset

def _select_inputs_with_source_class(dataset, trigger_source_labels):
    if trigger_source_labels is None:
        return dataset
    else:
        labels = np.array(dataset['label'])
        rows_with_source_label = np.argwhere(np.isin(labels, trigger_source_labels))[:, 0]
        return dataset.select(np.unique(rows_with_source_label))

def _initialize_dummy_trigger(dataset, tokenizer, trigger_length, trigger_loc, dummy=None):
    '''
    Insert a dummy trigger in either the start, middle or end (trigger_length)
    Change the input_ids and attention_mask, as necessary
    Also add a trigger_mask that will be used during loss calculations
    '''
    if dummy is None:
        dummy = tokenizer.pad_token_id
    def _initialize_dummy_trigger_helper(examples):
        def _find_insertion_location(trigger_loc, tokenizer):
            if trigger_loc == 'start':
                insertion_ixs = 1
            elif trigger_loc == 'middle':
                insertion_ixs = tools._get_max_seq_length(tokenizer)//2
            elif trigger_loc == 'end':
                insertion_ixs = tools._get_max_seq_length(tokenizer)-1
            else:
                return NotImplementedError
            return insertion_ixs
        insertion_ixs = _find_insertion_location(trigger_loc, tokenizer)
        result = {k:torch.tensor(v) for k,v in examples.items()}
        def _insert_2d_into_var(insertion_ixs, base, insert):
            return torch.cat([base[:, :insertion_ixs], insert, base[:, insertion_ixs:]], 1)
        def _expand_tensor(tensor, num_rows):
            return tensor.unsqueeze(0).repeat(num_rows, 1)
        num_examples = len(examples['input_ids'])
        trigger_input_ids = _expand_tensor(torch.tensor([dummy]*trigger_length), num_examples)
        trigger_attention = _expand_tensor(torch.tensor([1]*trigger_length), num_examples)
        token_type_ids = _expand_tensor(torch.tensor([0]*trigger_length), num_examples)
        
        temp_attention_mask = deepcopy(result['attention_mask'])
        zeros = torch.zeros_like(result['attention_mask'])
        result['input_ids'] = _insert_2d_into_var(insertion_ixs, result['input_ids'], trigger_input_ids)
        result['attention_mask'] = _insert_2d_into_var(insertion_ixs, result['attention_mask'], trigger_attention)
        result['token_type_ids'] = _insert_2d_into_var(insertion_ixs, result['token_type_ids'], token_type_ids)
        result['attention_mask_without_trigger'] = _insert_2d_into_var(insertion_ixs, temp_attention_mask, token_type_ids)
        result['trigger_mask'] = _insert_2d_into_var(insertion_ixs, zeros, deepcopy(trigger_attention))

        result = {k:v.tolist() for k,v in result.items()}

        return result

    dataset = dataset.map(
        _initialize_dummy_trigger_helper,
        batched=True,
        num_proc=2,
        remove_columns=dataset.column_names,
        keep_in_memory=True)
    dataset.set_format('torch', columns=dataset.column_names)

    return dataset

def _add_baseline_probabilities(dataset, models):
    def _add_baseline_probabilities_helper(batch):
        with torch.no_grad():
            modified_batch = deepcopy(batch)
            modified_batch['attention_mask'] = modified_batch['attention_mask_without_trigger']
            all_logits = models(modified_batch)
            probabilities = [_get_probabilitites(all_logits['suspicious']['logits'])]
            for clean_logits in all_logits['clean']:
                probabilities += [_get_probabilitites(clean_logits['logits'])]
            probabilities = torch.stack(probabilities)
            batch['baseline_probabilities'] = models.clean_model_aggregator_fn(probabilities, dim=0)
            batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
            return batch
    dataset = dataset.map(_add_baseline_probabilities_helper, batched=True, num_proc=1, keep_in_memory=True, batch_size=20)
    return dataset

def _get_probabilitites(logits):
    scores = torch.exp(logits)
    probs = scores/torch.sum(scores, dim=1, keepdim=True)
    return probs


class SCTriggeredDataset(tools.TorchTriggeredDataset):

    def __init__(self, huggingface_dataset):
        super().__init__(huggingface_dataset)
        self.label = huggingface_dataset['label'].clone().detach().clone().long()

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['label'] = self.label[idx]
        return sample