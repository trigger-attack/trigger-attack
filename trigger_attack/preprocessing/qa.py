import torch
from copy import deepcopy
from trigger_attack.preprocessing import tools
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

def qa_dataset_preprocessing(trigger_dataset, models):

    dataset = _tokenize_for_qa(trigger_dataset.original_dataset, models.tokenizer)
    dataset = _select_qa_examples_with_an_answer_in_context(dataset, models.tokenizer)
    dataset = tools._select_unique_inputs(dataset)
    dataset = _initialize_dummy_trigger(dataset, models.tokenizer, trigger_dataset.trigger_length, trigger_dataset.trigger_loc)
    dataset = _add_baseline_probabilities(dataset, models)
    dataset = QATriggeredDataset(dataset)

    return dataset

def _check_valid_trigger_loc(trigger_loc):
    valid_trigger_locations = ['context', 'question', 'both']
    error_msg = f"Unsupported trigger location, please pick one of {valid_trigger_locations}"
    assert trigger_loc in valid_trigger_locations, error_msg

@torch.no_grad()
def _tokenize_for_qa(dataset, tokenizer, doc_stride=128):

    question_column_name, context_column_name, answer_column_name  = "question", "context", "answers"
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        
        # initialize lists
        var_list = ['question_start_and_end', 'context_start_and_end', 
                    'answer_start_and_end']
        for var_name in var_list:
            tokenized_examples[var_name] = []
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # Let's label those examples!
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_ix = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            context_ix = 1 if pad_on_right else 0
            question_ix = 0 if pad_on_right else 1
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_ix = sample_mapping[i]
            answers = examples[answer_column_name][sample_ix]

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
            token_question_start_ix = get_token_index(sequence_ids, input_ids, index=question_ix, is_end=False)
            token_question_end_ix   = get_token_index(sequence_ids, input_ids, index=question_ix, is_end=True)

            tokenized_examples["question_start_and_end"].append([token_question_start_ix, token_question_end_ix])

            # populate context_start_and_end
            token_context_start_ix = get_token_index(sequence_ids, input_ids, index=context_ix, is_end=False)
            token_context_end_ix   = get_token_index(sequence_ids, input_ids, index=context_ix, is_end=True)

            tokenized_examples["context_start_and_end"].append([token_context_start_ix, token_context_end_ix])

            def set_answer_start_and_end_to_ixs(first_ix, second_ix):
                tokenized_examples["answer_start_and_end"].append([first_ix, second_ix])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if (start_char < offsets[token_context_start_ix][0] or offsets[token_context_end_ix][1] < end_char):
                    set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
                else:
                    token_answer_start_ix = token_context_start_ix
                    token_answer_end_ix = token_context_end_ix
                    while token_answer_start_ix < len(offsets) and offsets[token_answer_start_ix][0] <= start_char:
                        token_answer_start_ix += 1
                    while offsets[token_answer_end_ix][1] >= end_char:
                        token_answer_end_ix -= 1
                    set_answer_start_and_end_to_ixs(token_answer_start_ix-1, token_answer_end_ix+1)
            
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_ix else None)
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

@torch.no_grad()
def _select_qa_examples_with_an_answer_in_context(dataset, tokenizer):
    answer_starts = torch.tensor(dataset['answer_start_and_end'])[:, 0]
    non_cls_answer_indices = (~torch.eq(answer_starts, tokenizer.cls_token_id)).nonzero().flatten()
    return dataset.select(non_cls_answer_indices)


def _initialize_dummy_trigger(dataset, tokenizer, trigger_length, trigger_loc, dummy=0):
    if dummy is None:
        dummy = tokenizer.pad_token_id

    trigger_insertion_locations = ['end', 'end']

    is_context_first = tokenizer.padding_side != 'right'

    def _get_context_and_question_trigger_length(trigger_loc, trigger_length):
        c_trigger_length, q_trigger_length = 0, 0
        if trigger_loc in ['context', 'both']:
            c_trigger_length = trigger_length
        if trigger_loc in ['question', 'both']:
            q_trigger_length = trigger_length
        return c_trigger_length, q_trigger_length
    c_trigger_length, q_trigger_length = _get_context_and_question_trigger_length(trigger_loc, trigger_length)
    
    def initialize_dummy_trigger_helper(dataset_instance_source):
        input_id, att_mask, token_type, q_pos, c_pos, ans_pos = [deepcopy(torch.tensor(dataset_instance_source[x])) for x in \
            ['input_ids', 'attention_mask', 'token_type_ids', 'question_start_and_end', 'context_start_and_end', 'answer_start_and_end']]
        
        var_list = ['input_ids', 'attention_mask', 'token_type_ids', 'trigger_mask', 'valid_mask', 'answer_mask']
        dataset_instance = {var_name:None for var_name in var_list}

        def get_insertion_ix(insertion_location, start_end_ix):
            valid_insertion_locations = ['start', 'end']
            assert insertion_location in valid_insertion_locations, \
                f'please enter either {valid_insertion_locations} as an insertion_location'
            if insertion_location == 'start':
                insertion_ix = start_end_ix[0]
            elif insertion_location == 'end':
                insertion_ix = start_end_ix[1]+1
            return insertion_ix
                
        
        q_idx = get_insertion_ix(trigger_insertion_locations[0], q_pos)
        c_idx = get_insertion_ix(trigger_insertion_locations[1], c_pos)

        q_trigger_id, c_trigger_id = -1, -2
        q_trigger = torch.tensor([q_trigger_id]*q_trigger_length).long()
        c_trigger = torch.tensor([c_trigger_id]*c_trigger_length).long()

        first_idx, second_idx = q_idx, c_idx
        first_trigger, second_trigger = deepcopy(q_trigger), deepcopy(c_trigger)
        first_trigger_length, second_trigger_length = q_trigger_length, c_trigger_length
        if is_context_first:
            first_idx, second_idx = c_idx, q_idx
            first_trigger, second_trigger = deepcopy(c_trigger), deepcopy(q_trigger)
            first_trigger_length, second_trigger_length = c_trigger_length, q_trigger_length

        def insert_tensors_in_var(var, first_tensor, second_tensor=None):
            new_var = torch.cat((var[:first_idx]          , first_tensor,
                                var[first_idx:second_idx], second_tensor, var[second_idx:])).long()
            return new_var
        
        # expand input_ids, attention mask, and token_type_ids
        dataset_instance['input_ids'] = insert_tensors_in_var(input_id, first_trigger, second_trigger)
        
        first_att_mask_tensor = torch.zeros(first_trigger_length) + att_mask[first_idx].item()
        second_att_mask_tensor = torch.zeros(second_trigger_length) + att_mask[second_idx].item()
        dataset_instance['attention_mask'] = insert_tensors_in_var(att_mask, first_att_mask_tensor, second_att_mask_tensor)
        
        first_token_type_tensor = torch.zeros(first_trigger_length) + token_type[first_idx].item()
        second_token_type_tensor = torch.zeros(second_trigger_length) + token_type[second_idx].item()
        dataset_instance['token_type_ids'] = insert_tensors_in_var(token_type, first_token_type_tensor, second_token_type_tensor)

        # make trigger mask
        q_trigger_mask = torch.eq(dataset_instance['input_ids'], q_trigger_id)
        c_trigger_mask = torch.eq(dataset_instance['input_ids'], c_trigger_id)
        dataset_instance['trigger_mask'] = q_trigger_mask | c_trigger_mask
        
        # make valid_mask
        old_valid_mask = torch.zeros_like(input_id)
        old_valid_mask[c_pos[0]: c_pos[1]+1] = 1
        old_valid_mask[tokenizer.cls_token_id] = 1
        dataset_instance['valid_mask'] = insert_tensors_in_var(old_valid_mask, torch.zeros(first_trigger_length), torch.ones(second_trigger_length))

        # make answer mask
        answer_mask = torch.zeros_like(input_id)
        answer_mask[ans_pos[0]:ans_pos[1]+1] = 1
        dataset_instance['answer_mask'] = insert_tensors_in_var(answer_mask, torch.zeros(first_trigger_length), torch.zeros(second_trigger_length)) 

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
    
    dataset = dataset.remove_columns([f'{v}_start_and_end' for v in ['question', 'context', 'answer']])
    dataset.set_format('torch', columns=dataset.column_names)
    return dataset


def _add_baseline_probabilities(dataset, models):
    def _add_baseline_probabilities_helper(batch):
        with torch.no_grad():
            modified_batch = deepcopy(batch)
            modified_batch['attention_mask'] = torch.logical_and(modified_batch['attention_mask'], ~modified_batch['trigger_mask'])
            all_logits = models(modified_batch)
            probabilities = [_get_probabilitites(all_logits['suspicious'], batch)]
            for clean_logits in all_logits['clean']:
                probabilities += [_get_probabilitites(clean_logits, batch)]
            probabilities = torch.stack(probabilities)
            batch['baseline_probabilities'] = models.clean_model_aggregator_fn(probabilities, dim=0)
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


class QATriggeredDataset(tools.TorchTriggeredDataset):

    def __init__(self, huggingface_dataset):
        super().__init__(huggingface_dataset)
        self.valid_mask = huggingface_dataset['valid_mask'].clone().detach().clone().bool()
        self.answer_mask = huggingface_dataset['answer_mask'].clone().detach().clone().bool()

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['valid_mask'] = self.valid_mask[idx]
        sample['answer_mask'] = self.answer_mask[idx]
        return sample


