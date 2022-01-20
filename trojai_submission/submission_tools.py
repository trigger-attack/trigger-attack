import json
import jsonschema
from jsonargparse import ArgumentParser, ActionConfigFile
import os
from itertools import product
import datasets
import torch
from submission_constants import args_defaults


def get_args():    
    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default=args_defaults['--model_filepath'])
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default=args_defaults['--tokenizer_filepath'])
    parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=args_defaults['--round_training_dataset_dirpath'])

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--parameter1', type=int, help='An example tunable parameter.')
    parser.add_argument('--parameter2', type=float, help='An example tunable parameter.')
    parser.add_argument('--parameter3', type=str, help='An example tunable parameter.')

    parser.add_argument('--is_submission', dest='is_submission', default=True, action='store_true',  help='Flag to determine if this is a submission to the NIST server',  )
    parser.add_argument('--gpu', nargs='+', required=False,          type=int, help='Which GPU', )

    return parser.parse_args()

def validate_config_file(args):
    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)

def load_classifier(classifier_filepath):
    return NotImplementedError


# belongs in trojai
def load_clean_examples(examples_dirpath, scratch_dirpath):
    json_files = []
    for filename in os.listdir(examples_dirpath):
        if (filename.endswith('.json') and 'clean'in filename):
            json_files.append(filename)
    
    return datasets.load_dataset('json', data_files=json_files, field='data', 
                                keep_in_memory=True, split='train', 
                                cache_dir=os.path.join(scratch_dirpath, '.cache'))


def get_clean_models_filepaths():
    return NotImplementedError

@torch.no_grad()
def tokenize_for_qa(tokenizer, dataset):

    question_column_name, context_column_name, answer_column_name  = "question", "context", "answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    # max_seq_length = min(tokenizer.model_max_length, 384)
    max_seq_length = min(tokenizer.model_max_length, 200)
    
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        
        # initialize lists
        var_list = ['question_start_and_end', 'context_start_and_end', 
                    'train_clean_baseline_likelihoods', 'train_eval_baseline_likelihoods', 
                    'test_clean_baseline_likelihoods', 'test_eval_baseline_likelihoods', 
                    'train_clean_answer_likelihoods', 'train_eval_answer_likelihoods', 
                    'test_clean_answer_likelihoods', 'test_eval_answer_likelihoods', 
                    'answer_start_and_end', 'repeated']
        for var_name in var_list:
            tokenized_examples[var_name] = []
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        already_included_samples = set()
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
            if sample_ix in already_included_samples:
                tokenized_examples['repeated'].append(True)
            else:
                tokenized_examples['repeated'].append(False)
            already_included_samples.add(sample_ix)
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
            
            for train_test, eval_clean in product(['train', 'test'], ['eval', 'clean']):
                tokenized_examples[f'{train_test}_{eval_clean}_baseline_likelihoods'].append(torch.zeros(1))
            
            for train_test, eval_clean in product(['train', 'test'], ['eval', 'clean']):
                tokenized_examples[f'{train_test}_{eval_clean}_answer_likelihoods'].append(torch.zeros(1))


        return tokenized_examples
    
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=10,
        remove_columns=dataset.column_names,
        keep_in_memory=True)

    tokenized_dataset = tokenized_dataset.remove_columns(['offset_mapping'])
    assert len(tokenized_dataset) > 0

    return tokenized_dataset