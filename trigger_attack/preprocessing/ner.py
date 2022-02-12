from matplotlib import tri
import torch
from trigger_attack.preprocessing import tools
from copy import deepcopy
import numpy as np

def ner_dataset_preprocessing(trigger_dataset, models):
    dataset = _tokenize_for_ner(trigger_dataset.original_dataset, models.tokenizer)
    dataset = tools._select_unique_inputs(dataset)
    dataset = _initialize_dummy_trigger(dataset, models.tokenizer, trigger_dataset.trigger_length, trigger_dataset.trigger_source_labels)
    dataset = _add_baseline_probabilities(dataset, models)
    dataset = tools.TorchTriggeredDataset(dataset)

    return dataset


def _tokenize_for_ner(dataset, tokenizer):
    def _tokenize_for_ner_helper(examples):
        tokenized_inputs = tokenizer(
            examples['tokens'], 
            padding=True, 
            truncation=True, 
            is_split_into_words=True, 
            max_length=tools._get_max_seq_length(tokenizer),
            return_token_type_ids=True)
        
        word_ids = [tokenized_inputs.word_ids(i) for i in range(len(examples['ner_tags']))]
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
    dataset = dataset.map(
        _tokenize_for_ner_helper,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        keep_in_memory=True)
    return dataset


def _initialize_dummy_trigger(dataset, tokenizer, trigger_length, trigger_source_labels, dummy=None):
    '''
    Insert the a trigger length tensor at the trigger source label token with:
        input_ids: insert dummy
        attention_mask: insert ones
        token_type_ids: insert zeroes
        label: insert -100's
    trigger_source_loc is the location of the token that the trigger is targetting
    attention_mask without_trigger is an optional field that has zeroes wherever the trigger is present
    trigger_mask is zeroes anywhere and ones wherever the trigger is present
    '''
    assert trigger_source_labels is not None, "trigger source labels needs to be specified"
    if dummy is None:
        dummy = tokenizer.pad_token_id
    def insert_into_var(insertion_ixs, base, insert): ##TODO dim
        return base[:insertion_ixs] + insert + base[insertion_ixs:]
    def insert_trigger_helper(examples):
        result = {
            'input_ids':[],
            'attention_mask':[],
            'attention_mask_without_trigger':[],
            'token_type_ids':[],
            'label':[],
            'trigger_source_loc':[],
            'trigger_mask':[]
        }
        '''
        YOUR CODE HERE
        '''
        for i in range(len(examples['input_ids'])):
            example = {}
            locations = []
            for key in examples.keys():
                example[key] = examples[key][i]
            labels = np.array(example['label'])
            start = np.argwhere(np.isin(labels, trigger_source_labels))
            mask = []
            input_ids, attention_mask, attention_mask_without_trigger, token_type_ids, label = example['input_ids'], example['attention_mask'], example['attention_mask'], example['token_type_ids'], example['label']
            if start.shape[0] == 0:
                continue
            start = list(start[0])
            for j, ix in enumerate(start):
                target_cur_idx = (j+1) * trigger_length + ix
                target_cur_len = j * trigger_length
                input_ids = insert_into_var(start[j]+target_cur_len, input_ids, [dummy]* trigger_length)
                attention_mask = insert_into_var(start[j]+target_cur_len, attention_mask, [1] * trigger_length)
                attention_mask_without_trigger = insert_into_var(start[j]+target_cur_len, attention_mask_without_trigger, [0] * trigger_length)
                token_type_ids = insert_into_var(start[j]+target_cur_len, token_type_ids, [0] * trigger_length)
                label = insert_into_var(start[j]+target_cur_len, label, [-100] * trigger_length)
                locations.append(target_cur_idx) 
                if j == 0:
                    mask.extend([0] * ix)
                else:
                    mask.extend([0] * (ix - start[j-1])) 
                mask.extend([1]*trigger_length)
            mask.extend([0] * (len(example['input_ids']) + trigger_length - len(mask)))    
            result['input_ids'].append(input_ids)
            result['attention_mask'].append(attention_mask)
            result['attention_mask_without_trigger'].append(attention_mask_without_trigger)
            result['token_type_ids'].append(token_type_ids)
            result['label'].append(label)
            result['trigger_source_loc'].append(locations)
            result['trigger_mask'].append(mask)
        
        return result

    dataset = dataset.map(
        insert_trigger_helper,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        keep_in_memory=True)
    dataset.set_format('torch', columns=dataset.column_names)
    return dataset


def _add_baseline_probabilities(dataset, models):
    '''
    returns the likelihoods over classes of the token that the trigger is targetting
    '''
    def _add_baseline_probabilities_helper(batch):
        with torch.no_grad():
            modified_batch = deepcopy(batch)
            modified_batch['attention_mask'] = modified_batch['attention_mask_without_trigger']
            all_logits = models(modified_batch)
            probabilities = [_get_probabilitites(all_logits['suspicious'].logits, batch['trigger_source_loc'].squeeze())]
            for clean_logits in all_logits['clean']:
                probabilities += [_get_probabilitites(clean_logits.logits, batch['trigger_source_loc'].squeeze())]
            probabilities = torch.stack(probabilities)
            batch['baseline_probabilities'] = models.clean_model_aggregator_fn(probabilities, dim=0)
            batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
            return batch
    dataset = dataset.map(_add_baseline_probabilities_helper, batched=True, num_proc=1, keep_in_memory=True, batch_size=20)
    return dataset

def _get_probabilitites(logits, trigger_source_loc):
    source_loc_logits = logits[torch.arange(len(logits)), trigger_source_loc]
    scores = torch.exp(source_loc_logits)
    probs = scores/torch.sum(scores, dim=1, keepdim=True)
    return probs

class NERTriggeredDataset(tools.TorchTriggeredDataset):

    def __init__(self, huggingface_dataset):
        super().__init__(huggingface_dataset)
        self.label = huggingface_dataset['trigger_source_loc'].clone().detach().clone().long()

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['trigger_source_loc'] = self.label[idx]
        return sample
