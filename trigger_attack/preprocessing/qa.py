import torch
from copy import deepcopy
from .torch_triggered_dataset import TorchTriggeredDataset
from .dataset_preprocessor import datasetPreprocessor


class QADatasetPreprocessor(datasetPreprocessor):
    def __init__(self, dataset, trigger, trigger_models, tokenizer):
        super().__init__(dataset, trigger, trigger_models, tokenizer)
        self.__check_valid_trigger_loc(self.trigger.location)

    @staticmethod
    def __check_valid_trigger_loc(trigger_location):
        valid_trigger_locations = ['context', 'question', 'both']
        error_msg = (
            f"Unsupported trigger location. "
            f"Please pick one of: {valid_trigger_locations}"
        )
        assert trigger_location in valid_trigger_locations, error_msg

    def _tokenize(self, original_dataset):
        tokenized_dataset = original_dataset.map(
            self.__prepare_train_features,
            batched=True,
            num_proc=1,
            remove_columns=original_dataset.column_names,
            keep_in_memory=True)

        tokenized_dataset = tokenized_dataset.remove_columns(
                                            ['offset_mapping'])
        return tokenized_dataset

    def __prepare_train_features(self, examples):
        question_column_name = "question"
        context_column_name = "context"
        answer_column_name = "answers"

        pad_on_right = self.tokenizer.padding_side == "right"

        if pad_on_right:
            left_col = question_column_name
            right_col = context_column_name
        else:
            left_col = context_column_name
            right_col = question_column_name

        tokenized_examples = self.tokenizer(
            examples[left_col],
            examples[right_col],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.get_max_seq_length(),
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True)

        # initialize lists
        var_list = ['question_start_and_end',
                    'context_start_and_end',
                    'answer_start_and_end']
        for var_name in var_list:
            tokenized_examples[var_name] = []

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # Let's label those examples!
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_ix = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example
            #   (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            context_ix = 1 if pad_on_right else 0
            question_ix = 0 if pad_on_right else 1

            # One example can give several spans, this is the index of the
            #   example containing this span of text.
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
            token_question_start = get_token_index(sequence_ids,
                                                   input_ids,
                                                   index=question_ix,
                                                   is_end=False)
            token_question_end = get_token_index(sequence_ids,
                                                 input_ids,
                                                 index=question_ix,
                                                 is_end=True)

            start_and_end = [token_question_start, token_question_end]
            tokenized_examples["question_start_and_end"].append(start_and_end)

            # populate context_start_and_end
            token_context_start = get_token_index(sequence_ids,
                                                  input_ids,
                                                  index=context_ix,
                                                  is_end=False)
            token_context_end = get_token_index(sequence_ids,
                                                input_ids,
                                                index=context_ix,
                                                is_end=True)
            start_and_end = [token_context_start, token_context_end]
            tokenized_examples["context_start_and_end"].append(start_and_end)

            def set_answer_start_and_end_to_ixs(first_ix, second_ix):
                tokenized_examples["answer_start_and_end"].append([first_ix,
                                                                   second_ix])

            def has_no_answer(answers):
                return len(answers["answer_start"]) == 0
            if has_no_answer(answers):
                set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                def answer_is_out_of_span(start_char,
                                          offsets,
                                          token_context_start_ix,
                                          end_char):
                    return (
                        start_char < offsets[token_context_start_ix][0] or
                        offsets[token_context_end][1] < end_char
                    )
                if answer_is_out_of_span(start_char, offsets,
                                         token_context_start, end_char):
                    set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
                else:
                    token_answer_start_ix = token_context_start
                    token_answer_end_ix = token_context_end
                    while (token_answer_start_ix < len(offsets) and
                           offsets[token_answer_start_ix][0] <= start_char):
                        token_answer_start_ix += 1
                    while offsets[token_answer_end_ix][1] >= end_char:
                        token_answer_end_ix -= 1
                    set_answer_start_and_end_to_ixs(token_answer_start_ix-1,
                                                    token_answer_end_ix+1)

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_ix else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def _select_inputs_with_source_class(self, unique_inputs_dataset):
        answer_start_and_end = unique_inputs_dataset['answer_start_and_end']
        answer_starts = torch.tensor(answer_start_and_end)[:, 0]
        non_cls_answer_ixs = ~torch.eq(answer_starts, 0)
        non_cls_answer_ixs = (non_cls_answer_ixs).nonzero().flatten()
        return unique_inputs_dataset.select(non_cls_answer_ixs)

    def _insert_dummy(self, unique_inputs_dataset):
        dummy = self.tokenizer.pad_token_id

        trigger_insertion_locations = ['end', 'end']

        is_context_first = self.tokenizer.padding_side != 'right'

        def _get_trigger_lengths(trigger_loc, trigger_length):
            c_trigger_length, q_trigger_length = 0, 0
            if trigger_loc in ['context', 'both']:
                c_trigger_length = trigger_length
            if trigger_loc in ['question', 'both']:
                q_trigger_length = trigger_length
            return (c_trigger_length, q_trigger_length)
        trigger_length = len(self.trigger.input_ids)
        trigger_lengths = _get_trigger_lengths(self.trigger.location,
                                               trigger_length)
        c_trigger_length, q_trigger_length = trigger_lengths

        def initialize_dummy_trigger_helper(examples):
            def make_tensor_copy(examples, col_name):
                return deepcopy(torch.tensor(examples[col_name]))

            input_id = make_tensor_copy(examples, 'input_ids')
            att_mask = make_tensor_copy(examples, 'attention_mask')
            token_type = make_tensor_copy(examples, 'token_type_ids')
            q_pos = make_tensor_copy(examples, 'question_start_and_end')
            c_pos = make_tensor_copy(examples, 'context_start_and_end')
            ans_pos = make_tensor_copy(examples, 'answer_start_and_end')

            var_list = ['input_ids', 'attention_mask', 'token_type_ids',
                        'trigger_mask', 'valid_mask', 'answer_mask']
            result = {var_name: None for var_name in var_list}

            def get_insertion_ix(insertion_location, start_end_ix):
                valid_insertion_locations = ['start', 'end']
                assert insertion_location in valid_insertion_locations, \
                    (f'please enter either {valid_insertion_locations} '
                     f'as an insertion_location')
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
            first_trigger = deepcopy(q_trigger)
            second_trigger = deepcopy(c_trigger)
            first_trigger_length = q_trigger_length
            second_trigger_length = c_trigger_length
            if is_context_first:
                first_idx, second_idx = c_idx, q_idx
                first_trigger = deepcopy(c_trigger)
                second_trigger = deepcopy(q_trigger)
                first_trigger_length = c_trigger_length
                second_trigger_length = q_trigger_length

            def insert_tensors(var, first_tensor, second_tensor=None):
                new_var = torch.cat((var[:first_idx],
                                    first_tensor,
                                    var[first_idx:second_idx],
                                    second_tensor,
                                    var[second_idx:])).long()
                return new_var

            # expand input_ids, attention mask, and token_type_ids
            result['input_ids'] = insert_tensors(
                input_id, first_trigger, second_trigger)

            first_att_mask_tensor = (torch.zeros(first_trigger_length) +
                                     att_mask[first_idx].item())
            second_att_mask_tensor = (torch.zeros(second_trigger_length) +
                                      att_mask[second_idx].item())
            result['attention_mask'] = insert_tensors(att_mask,
                                                      first_att_mask_tensor,
                                                      second_att_mask_tensor)

            first_token_type_tensor = (torch.zeros(first_trigger_length) +
                                       token_type[first_idx].item())
            second_token_type_tensor = (torch.zeros(second_trigger_length) +
                                        token_type[second_idx].item())
            result['token_type_ids'] = insert_tensors(token_type,
                                                      first_token_type_tensor,
                                                      second_token_type_tensor)

            # make trigger mask
            q_trigger_mask = torch.eq(result['input_ids'], q_trigger_id)
            c_trigger_mask = torch.eq(result['input_ids'], c_trigger_id)
            result['trigger_mask'] = q_trigger_mask | c_trigger_mask

            # make valid_mask
            old_valid_mask = torch.zeros_like(input_id)
            old_valid_mask[c_pos[0]: c_pos[1]+1] = 1
            cls_mask = input_id == self.tokenizer.cls_token_id
            old_valid_mask[cls_mask] = 1
            first_zeros = torch.zeros(first_trigger_length)
            second_ones = torch.ones(second_trigger_length)
            result['valid_mask'] = insert_tensors(
                old_valid_mask, first_zeros, second_ones)

            # make answer mask
            answer_mask = torch.zeros_like(input_id)
            answer_mask[ans_pos[0]:ans_pos[1]+1] = 1
            second_zeros = torch.zeros(second_trigger_length)
            result['answer_mask'] = insert_tensors(answer_mask,
                                                   first_zeros,
                                                   second_zeros)

            # change trigger to pad
            target_length = len(result['input_ids'][result['trigger_mask']])
            new_trigger = torch.tensor([dummy]*target_length)
            result['input_ids'][result['trigger_mask']] = new_trigger

            return result

        dataset_with_dummy = unique_inputs_dataset.map(
            initialize_dummy_trigger_helper,
            batched=False,
            num_proc=1,
            keep_in_memory=True)

        cols_to_remove = [f'{v}_start_and_end' for v in ['question',
                                                         'context',
                                                         'answer']]
        dataset_with_dummy = dataset_with_dummy.remove_columns(cols_to_remove)
        dataset_with_dummy.set_format('torch',
                                      columns=dataset_with_dummy.column_names)
        return dataset_with_dummy

    @torch.no_grad()
    def _add_baseline_probabilities(self, dataset_with_dummy):
        dataset_with_baseline_probabilities = dataset_with_dummy.map(
            self._add_baseline_probabilities_helper,
            batched=True,
            num_proc=1,
            keep_in_memory=True,
            batch_size=20)
        return dataset_with_baseline_probabilities

    def _add_baseline_probabilities_helper(self, original_batch):
        batch = deepcopy(original_batch)
        ignore_trigger = torch.logical_and(
            batch['attention_mask'], ~batch['trigger_mask'])
        batch['attention_mask'] = ignore_trigger
        all_logits = self.trigger_models(batch)
        probabilities = []
        probabilities = [self._get_probabilitites(
            all_logits['suspicious'], original_batch)]
        for clean_logits in all_logits['clean']:
            probabilities += [self._get_probabilitites(
                clean_logits, original_batch)]
        probabilities = torch.stack(probabilities)
        original_batch['baseline_probabilities'] = self.agg_fn(
            probabilities, dim=0)
        result = {k: v.detach().cpu().numpy()
                  for k, v in original_batch.items()}
        return result

    @staticmethod
    def _get_probabilitites(logits, batch):
        combined_logits = logits['start_logits'] + logits['end_logits']
        combined_logits = combined_logits.to(torch.device('cpu'))
        valid_logits = combined_logits - (~batch['valid_mask'].bool())*1e10
        scores = torch.exp(valid_logits)
        probs = scores/torch.sum(scores, dim=1, keepdim=True)
        return probs

    def _package_into_torch_dataset(self, dataset_with_baseline_probabilities):
        non_cls = dataset_with_baseline_probabilities['baseline_probabilities'][:, 0] < .3
        non_cls_mask = non_cls.nonzero().flatten()
        filtered_dataset = dataset_with_baseline_probabilities.select(non_cls_mask)
        dataset = self.QATriggeredDataset(
            filtered_dataset, len(self.trigger.input_ids))
        return dataset

    class QATriggeredDataset(TorchTriggeredDataset):

        def __init__(self, dataset, trigger_length):
            super().__init__(dataset, trigger_length)
            self.valid_mask = dataset['valid_mask'].clone().detach().bool()
            self.answer_mask = dataset['answer_mask'].clone().detach().bool()

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            sample['valid_mask'] = self.valid_mask[idx]
            sample['answer_mask'] = self.answer_mask[idx]
            return sample
