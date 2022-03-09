from dataclasses import dataclass
from typing import Any


@dataclass
class TestingDataPaths:
    suspicious_model_filepath: str
    clean_models_filepaths: list
    tokenizer_filepath: str

sc_paths = TestingDataPaths(
    'data/round9_sample_dataset/models/id-00000014/model.pt',
    ['data/round9_sample_dataset/models/id-00000002/model.pt'],
    'data/round9_sample_dataset/tokenizers/roberta-base.pt'
)

ner_paths = TestingDataPaths(
    'data/round9_sample_dataset/models/id-00000068/model.pt',
    ['data/round9_sample_dataset/models/id-00000086/model.pt'],
    'data/round9_sample_dataset/tokenizers/google-electra-small-discriminator.pt'
)

qa_paths = TestingDataPaths(
    'data/round8_sample_dataset/models/id-00000000/model.pt',
    ['data/round8_sample_dataset/models/id-00000018/model.pt'],
    'data/round8_sample_dataset/tokenizers/tokenizer-roberta-base.pt'
)
