import json
import jsonschema
from jsonargparse import ArgumentParser, ActionConfigFile
from submission_constants import args_defaults, CLASSIFIER_PATH
from joblib import load
from sklearn.preprocessing import OneHotEncoder
import string
import random
import os
import pandas as pd


def get_args():
    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, default=args_defaults['--model_filepath'], help='File path to the pytorch model file to be evaluated.', )
    parser.add_argument('--tokenizer_filepath', type=str, default='None', help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    parser.add_argument('--result_filepath', type=str, default=args_defaults['--result_filepath'], help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    default_scratch_dirpath = '/scratch/utrerf/trigger-attack/trojai_submission/scratch'
    parser.add_argument('--scratch_dirpath', type=str, default=default_scratch_dirpath, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=args_defaults['--round_training_dataset_dirpath'])

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    configure_models_dirpath = '/scratch/utrerf/trigger-attack/trigger_attack/trojan_model_datasets/round9-train-dataset/models'
    parser.add_argument('--configure_models_dirpath', type=str, default=configure_models_dirpath, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--parameter1', type=int, help='An example tunable parameter.')
    parser.add_argument('--parameter2', type=float, help='An example tunable parameter.')
    parser.add_argument('--parameter3', type=str, help='An example tunable parameter.')

    # trigger_dataset args
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--trigger_length', type=int, default=8, help='An example tunable parameter.')
    parser.add_argument('--trigger_init_fn', type=str, default='embed_ch', help='Trigger Initialization Functions')
    parser.add_argument('--num_clean_test_models', type=int, default=4, help='Number of clean models to use')
    parser.add_argument('--num_clean_models', type=int, default=1, help='Number of clean models to use')
    parser.add_argument('--num_reinitializations', type=int, default=5, help='Number of times we reinitialize the trigger')
    parser.add_argument('--num_candidates_per_token', type=int, default=5, help='Number of tokens in a trigger')
    parser.add_argument('--max_iter', type=int, default=20, help='Maximum number of iterations of iterations')
    parser.add_argument('--test_loss_threshold', type=float, default=0.005, help='Threshold after which we stop trigger reconstruction for that objective')
    parser.add_argument('--task', type=str, default='None', help='Which task to focus on.')

    parser.add_argument('--is_submission', dest='is_submission', default=True, action='store_true',  help='Flag to determine if this is a submission to the NIST server',  )
    parser.add_argument('--gpu', nargs='+', required=False,          type=int, help='Which GPU', )
    parser.add_argument('--unique_id', type=str, default=_id_generator(), help='An example tunable parameter.')

    return parser.parse_args()


def _id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


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


def get_extracted_features_folder(args):
    return os.path.join(args.scratch_dirpath, args.unique_id)

def read_all_features(extracted_features_folder):
    all_results_filepaths = get_all_results_filepaths(extracted_features_folder)
    features = []
    for result_filepath in all_results_filepaths:
        current_df = pd.read_csv(result_filepath, index_col=0)
        model_name = result_filepath.split('/')[-1]
        model_name = model_name.split('.')[0]
        current_df['model_name'] = model_name
        features.append(current_df)
    features = pd.concat(features)
    features = features.reset_index(drop=True)
    return features


def get_all_results_filepaths(basepath):
    all_results_filepaths = []
    for filename in os.listdir(basepath):
        if 'csv' in filename:
            filepath = os.path.join(basepath, filename)
            all_results_filepaths.append(filepath)
    return all_results_filepaths


def get_predictions(features):
    min_ix = features['test_loss'].argmin()
    min_features = features.loc[min_ix]
    min_features.task = min_features.task.upper()


    data = {
        'NER': [int(min_features.task=='NER')],
        'QA': [int(min_features.task=='QA')],
        'SC': [int(min_features.task=='SC')],
        'loss': [min_features['test_loss']],
    }
    X = pd.DataFrame(data)
    clf = load(CLASSIFIER_PATH)

    return clf.predict_proba(X)[0][1]
