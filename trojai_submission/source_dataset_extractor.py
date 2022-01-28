import os
from data_tools import _read_json

models_folder = "/scratch/utrerf/trigger-attack/trigger_attack/trojan_model_datasets/round9-train-dataset/models"
all_config_folders = [os.path.join(models_folder, m_folder, 'config.json') for m_folder in os.listdir(models_folder)]
all_source_datasets = [_read_json(config_filepath)["source_dataset"] for config_filepath in all_config_folders]

all_source_datasets = set(all_source_datasets)
print(all_source_datasets)
