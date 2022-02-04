import torch
from trigger_attack.preprocessing import tools
from copy import deepcopy

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
    max_seq_length = min(tokenizer.model_max_length, 384)
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]

    # print(max_seq_length)
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        tokenized_examples = tokenizer(
            examples["data"],
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        
        # initialize lists
        tokenized_examples["sc_start_and_end"] = []
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # Let's label those examples!
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_ix = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_ix = sample_mapping[i]
            answers = examples["data"][sample_ix]

            def get_token_index(sequence_ids, input_ids, index, is_end):
                token_ix = 0
                if is_end: 
                    token_ix = len(input_ids) - 1
                add_num = 1
                if is_end:
                    add_num = -1
                while sequence_ids[token_ix] != index:
                    token_ix += add_num
                return token_ix

            # populate question_start_and_end
            token_start_ix = get_token_index(sequence_ids, input_ids, index=0, is_end=False)
            token_end_ix   = get_token_index(sequence_ids, input_ids, index=0, is_end=True)

            tokenized_examples["sc_start_and_end"].append([token_start_ix, token_end_ix])
            
            # tokenized_examples["offset_mapping"][i] = [
            #     (o, )
            #     for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            # ]
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
            
        return tokenized_examples


    dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=2,
        remove_columns=dataset.column_names,
        keep_in_memory=True)

    dataset = dataset.remove_columns(['offset_mapping'])
    return dataset


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
    '''
    for batch in dataset:
        print('init dummy trigger(): batch.shape', batch.shape, 'batch.type', batch.type())
        print(batch.keys())
        new_text = batch['input_ids'][:-trigger_loc-1] +  + batch['input_ids'][trigger_loc:]
        batch['input_ids'] = new_input_ids
        new_input_ids = batch['input_ids'][:-trigger_loc-1] + dummy + batch['input_ids'][trigger_loc:]
        batch['input_ids'] = new_input_ids
        new_att_mask = batch['attention_mask'][:-trigger_loc-1] + torch.zeros(trigger_length) + batch['input_ids'][trigger_loc:]
        batch['attention_mask'] = attention_mask
        new_att_mask = batch['attention_mask'][:-trigger_loc-1] + torch.ones(trigger_length) + batch['input_ids'][trigger_loc:]
        batch['attention_mask'] = attention_mask
    return NotImplementedError
    '''
    trigger_insertion_locations = ['end']

    #is_context_first = tokenizer.padding_side != 'right'


    def initialize_dummy_trigger_helper(dataset_instance_source):
        print('initialize_dummy_trigger_helper()', dataset_instance_source.keys())
        input_id, att_mask, token_type = [deepcopy(torch.tensor(dataset_instance_source[x])) for x in \
            ['input_ids', 'attention_mask', 'token_type_ids', 'sc_start_and_end']]
        
        var_list = ['input_ids', 'attention_mask', 'token_type_ids', 'trigger_mask']
        dataset_instance = {var_name:None for var_name in var_list}

        def get_insertion_ix(insertion_location, start_end_ix):
            if insertion_location == 'start':
                return start_end_ix[0]
            elif insertion_location == 'end':
                return start_end_ix[1]+1
            else:
                print('please enter either "start" or "end" as an insertion_location')
        
        q_idx = get_insertion_ix(trigger_insertion_locations[0], q_pos)

        q_trigger_id = -1
        q_trigger = torch.tensor([q_trigger_id]*q_trigger_length).long()

        first_idx, second_idx = q_idx
        first_trigger = deepcopy(q_trigger)

        def insert_tensors_in_var(var, first_tensor, second_tensor=None):
            new_var = torch.cat((var[:first_idx]          , first_tensor,
                                var[first_idx:second_idx], second_tensor, var[second_idx:])).long()
            return new_var
        
        # expand input_ids, attention mask, and token_type_ids
        dataset_instance['input_ids'] = insert_tensors_in_var(input_id, first_trigger)
        
        first_att_mask_tensor = torch.zeros(first_trigger_length) + att_mask[first_idx].item()
        dataset_instance['attention_mask'] = insert_tensors_in_var(att_mask, first_att_mask_tensor)
        
        first_token_type_tensor = torch.zeros(first_trigger_length) + token_type[first_idx].item()
        dataset_instance['token_type_ids'] = insert_tensors_in_var(token_type, first_token_type_tensor)

        # make trigger mask
        q_trigger_mask = torch.eq(dataset_instance['input_ids'], q_trigger_id)
        dataset_instance['trigger_mask'] = q_trigger_mask

        # change trigger to pad
        target_length = len(dataset_instance['input_ids'][dataset_instance['trigger_mask']])
        new_trigger = torch.tensor([dummy]*target_length)
        dataset_instance['input_ids'][dataset_instance['trigger_mask']] = new_trigger

        return dataset_instance
    
    dataset = dataset.map(
        initialize_dummy_trigger_helper,
        batched=False,
        num_proc=1,
        keep_in_memory=True)
    
    dataset = dataset.remove_columns([f'{v}_start_and_end' for v in ['sentence']]) ## TODO
    dataset.set_format('torch', columns=dataset.column_names)
    return dataset


def _add_baseline_probabilities(dataset, models, agg_function):
    '''
    Add baseline probabilities
    '''
    '''
    YOUR CODE HERE
    '''
    def _add_baseline_probabilities_helper(batch):
        with torch.no_grad():
            modified_batch = deepcopy(batch)
            modified_batch['attention_mask'] = torch.logical_and(modified_batch['attention_mask'], ~modified_batch['trigger_mask'])
            all_logits = models(modified_batch)
            probabilities = [_get_probabilitites(all_logits['suspicious'], batch)]
            for clean_logits in all_logits['clean']:
                probabilities += [_get_probabilitites(clean_logits, batch)]
            probabilities = torch.stack(probabilities)
            batch['baseline_probabilities'] = agg_function(probabilities, dim=0)
            batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
            return batch
    dataset = dataset.map(_add_baseline_probabilities_helper, batched=True, num_proc=1, keep_in_memory=True, batch_size=20)
    return dataset


def _get_probabilitites(logits, batch):
    combined_logits = logits['start_logits'] + logits['end_logits']
    valid_logits = combined_logits.to(torch.device('cpu')) - (~batch['valid_mask'].bool())*1e10
    scores = torch.exp(valid_logits)
    probs = scores/torch.sum(scores, dim=1, keepdim=True)
    return probs
