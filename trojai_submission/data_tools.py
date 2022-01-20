import os
import json
import pandas as pd


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


def get_clean_model_filepaths(config, models_dataset_dirpath):
    dataset = _extract_dataset(config)
    arch = _extract_arch(config)
    matching_clean_model_dirnames = _find_matching_clean_model_dirnames(dataset, arch, models_dataset_dirpath)
    matching_model_dirnames = _exclude_suspicious_model(config, matching_clean_model_dirnames)
    return _expand_matching_model_dirnames_to_filepaths(models_dataset_dirpath, matching_model_dirnames)

def _get_models_dirpath(models_dataset_dirpath):
    return os.path.join(models_dataset_dirpath, 'models')

def _extract_dataset(config):
    return NotImplementedError

def _extract_arch(config):
    return NotImplementedError

def _find_matching_clean_model_dirnames(dataset, arch, models_dataset_dirpath):
    metadata = load_metadata(models_dataset_dirpath)
    matching_dataset_and_arch = (metadata.dataset == dataset) & (metadata.arch == arch)
    is_clean = metadata.poisoned == False
    matching_clean_model_dirnames = metadata[matching_dataset_and_arch & is_clean].model_name.tolist()
    return matching_clean_model_dirnames

def _exclude_suspicious_model(config, matching_clean_model_dirnames):
    suspicious_model_name = config['model_name']
    matching_clean_model_dirnames = [md for md in matching_clean_model_dirnames if md != suspicious_model_name]
    return matching_clean_model_dirnames

def _expand_matching_model_dirnames_to_filepaths(models_dataset_dirpath, matching_model_dirnames):
    models_dirpath = _get_models_dirpath(models_dataset_dirpath)
    return [os.path.join(models_dirpath, model_dirname, 'model.pt') for model_dirname in matching_model_dirnames]
    

def get_tokenizers_dirpath(models_dataset_dirpath):
    os.path.join(models_dataset_dirpath, 'tokenizers')


def load_metadata(models_dataset_dirpath):
    metadata_filepath = _get_metadata_filepath(models_dataset_dirpath)
    return pd.read_csv(metadata_filepath)

def _get_metadata_filepath(models_dataset_dirpath):
    return os.path.join(models_dataset_dirpath, 'METADATA.csv')




