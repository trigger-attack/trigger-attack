import torch
from trigger_attack.preprocessing import tools

def sc_dataset_preprocessing(dataset, models, trigger_length, trigger_loc, agg_function=torch.mean):
    _check_valid_trigger_loc(trigger_loc)
    dataset = _tokenize_for_sc(dataset, models.tokenizer)
    dataset = tools._select_unique_inputs(dataset)
    dataset = _initialize_dummy_trigger(dataset, models.tokenizer, trigger_length, trigger_loc)
    dataset = _add_baseline_probabilities(dataset, models, agg_function)
    dataset = tools.TorchTriggeredDataset(dataset)

    return dataset

def _check_valid_trigger_loc(trigger_loc):
    valid_trigger_locations = ['beginning', 'middle', 'end']
    error_msg = f"Unsupported trigger location, please pick one of {valid_trigger_locations}"
    assert trigger_loc in valid_trigger_locations, error_msg

def _tokenize_for_sc(dataset, tokenizer, doc_stride=128):
    '''
    Tokenization should return a max sequence length as specified in tools._get_max_seq_length
    Tokenizer output must be padded and truncated in order to be able to convert it to torch.tensor
    Also make sure to return token_type_ids
    '''
    max_seq_length = tools._get_max_seq_length(tokenizer)
    '''
    YOUR CODE HERE
    '''
    return NotImplementedError

def _initialize_dummy_trigger(dataset, tokenizer, trigger_length, trigger_loc, dummy=None):
    '''
    Insert a dummy trigger in either the beginning, middle or end (trigger_length)
    Change the input_ids and attention_mask, as necessary
    Also add a trigger_mask that will be used during loss calculations
    '''
    if dummy is None:
        dummy = tokenizer.pad_token_id
    '''
    YOUR CODE HERE
    '''
    return NotImplementedError

def _add_baseline_probabilities(dataset, models, agg_function):
    '''
    Add baseline probabilities
    '''
    '''
    YOUR CODE HERE
    '''
    return NotImplementedError