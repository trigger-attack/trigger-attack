import os
import json
import pandas as pd
import datasets
from trojai_submission.submission_constants import rounds_to_taks
from trigger_attack.loss_functions.sc import SCLoss
from trigger_attack.loss_functions.ner import NERLoss
from trigger_attack.loss_functions.qa import QALoss
from trigger_attack.preprocessing.sc import SCDatasetPreprocessor
from trigger_attack.preprocessing.ner import NERDatasetPreprocessor
from trigger_attack.preprocessing.qa import QADatasetPreprocessor


def load_config(model_filepath):
    config_filepath = _get_config_filepath(model_filepath)
    config = _read_json(config_filepath)
    return config


def _get_config_filepath(model_filepath):
    model_dirpath, _ = os.path.split(model_filepath)
    config_filepath = os.path.join(model_dirpath, 'config.json')
    return config_filepath


def _read_json(filepath):
    with open(filepath) as json_file:
        config = json.load(json_file)
    return config


def get_clean_model_filepaths(config,
                              models_dataset_dirpath,
                              suspicious_model_foldername):
    dataset = _extract_dataset(config)
    arch = _extract_arch(config)
    matching_clean_model_dirnames = _find_matching_clean_model_dirnames(
        dataset, arch, models_dataset_dirpath)
    matching_model_dirnames = _exclude_suspicious_model(
        config, matching_clean_model_dirnames, suspicious_model_foldername)
    return _expand_matching_model_dirnames_to_filepaths(
        models_dataset_dirpath, matching_model_dirnames)


def _get_models_dirpath(models_dataset_dirpath):
    return os.path.join(models_dataset_dirpath, 'models')


def _extract_dataset(config):
    return config['source_dataset']


def _extract_arch(config):
    return config['model_architecture']


def _find_matching_clean_model_dirnames(dataset, arch, models_dataset_dirpath):
    metadata = load_metadata(models_dataset_dirpath)
    matching_dataset_and_arch = ((metadata.source_dataset == dataset) &
                                 (metadata.model_architecture == arch))
    is_clean = metadata.poisoned == False
    filtered_metadata = metadata[matching_dataset_and_arch & is_clean]
    matching_clean_model_dirnames = filtered_metadata.model_name.tolist()
    return matching_clean_model_dirnames


def _exclude_suspicious_model(config,
                              matching_clean_model_dirnames,
                              suspicious_model_foldername):
    config_suspicious_model_name = ''
    if 'output_filepath' in config:
        config_suspicious_model_name = config['output_filepath'].split('/')[-1]
    exclude_list = [config_suspicious_model_name, suspicious_model_foldername]
    result = []
    for moder_dirname in matching_clean_model_dirnames:
        if moder_dirname not in exclude_list:
            result.append(moder_dirname)
    return result


def _expand_matching_model_dirnames_to_filepaths(models_dataset_dirpath,
                                                 matching_model_dirnames):
    models_dirpath = _get_models_dirpath(models_dataset_dirpath)
    result = []
    for model_dirname in matching_model_dirnames:
        path = os.path.join(models_dirpath, model_dirname, 'model.pt')
        result.append(path)
    return result


def get_tokenizers_dirpath(models_dataset_dirpath):
    os.path.join(models_dataset_dirpath, 'tokenizers')


def load_metadata(models_dataset_dirpath):
    metadata_filepath = _get_metadata_filepath(models_dataset_dirpath)
    return pd.read_csv(metadata_filepath)


def _get_metadata_filepath(models_dataset_dirpath):
    return os.path.join(models_dataset_dirpath, 'METADATA.csv')


def get_taskname(round_training_dataset_dirpath, config):
    if 'task_type' in config:
        taskname = config['task_type']
        assert taskname in rounds_to_taks.values(),\
               f"taskname {taskname} is not supported"
        return taskname

    for round_num, taskname in rounds_to_taks.items():
        if f'round{round_num}' in round_training_dataset_dirpath:
            return taskname

    raise RuntimeError("taskname was not found")


def load_examples(model_filepath, scratch_dirpath, clean_model_filepaths):
    dataset_list = \
        [load_dataset(model_filepath, scratch_dirpath)]
    for clean_model_filepath in clean_model_filepaths:
        dataset_list.append(
            load_dataset(clean_model_filepath, scratch_dirpath))

    return datasets.concatenate_datasets(dataset_list)


def load_dataset(model_filepath, scratch_dirpath):
    model_dirpath, _ = os.path.split(model_filepath)
    examples_dirpath = os.path.join(model_dirpath, 'example_data')
    if os.path.isdir(examples_dirpath):
        examples_filepath = []
        for fn in os.listdir(examples_dirpath):
            if fn.endswith('.json') and 'clean' in fn:
                full_path = os.path.join(examples_dirpath, fn)
                examples_filepath.append(full_path)
        dataset = datasets.load_dataset(
                    'json',
                    data_files=examples_filepath,
                    field='data',
                    keep_in_memory=True,
                    split='train',
                    cache_dir=os.path.join(scratch_dirpath, '.cache'))
    else:
        examples_filepath = os.path.join(
            model_dirpath, 'clean-example-data.json')
        dataset = datasets.load_dataset(
                    'json',
                    data_files=examples_filepath,
                    field='data',
                    keep_in_memory=True,
                    split='train',
                    cache_dir=os.path.join(scratch_dirpath, '.cache'))
    return dataset


def get_task_labels(task, config):
    if task == 'sc':
        return [0, 1]
    elif task == 'ner':
        result = []
        for i in range((config['num_outputs']-1)//2):
            result.append([(i*2)+1, (i*2)+2])
        return result
    elif task == 'qa':
        return ['self', 'cls']


task_to_loss = {
    'sc': SCLoss,
    'ner': NERLoss,
    'qa': QALoss
}


task_to_preprocessor = {
    'sc': SCDatasetPreprocessor,
    'ner': NERDatasetPreprocessor,
    'qa': QADatasetPreprocessor
}

task_to_trigger_locations = {
    'sc': ['start', 'middle', 'end'],
    'ner': ['None'],
    'qa': ['context', 'question', 'both']
}
