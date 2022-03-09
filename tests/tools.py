from trojai_submission import data_tools
from trigger_attack.trigger_models import TriggerModels
import torch
import os


def load_dataset(model_filepath):
    model_filepath = _prepend_current_script_path(model_filepath)
    scratch_filepath = '.tmp'
    return data_tools.load_examples(model_filepath, scratch_filepath, [])


def load_trigger_models(suspicious_model_filepath,
                        clean_model_filepaths):
    suspicious_model_filepath = _prepend_current_script_path(
                                    suspicious_model_filepath)
    clean_model_filepaths = [_prepend_current_script_path(path)
                             for path in clean_model_filepaths]
    return TriggerModels(suspicious_model_filepath,
                         clean_model_filepaths,
                         device=torch.device('cuda'))


def load_tokenizer(tokenizer_filepath):
    tokenizer_filepath = _prepend_current_script_path(
                                        tokenizer_filepath)
    return torch.load(tokenizer_filepath)


def _prepend_current_script_path(path):
    current_script_dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_script_dirname, path)
