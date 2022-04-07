CLASSIFIER_PATH = \
    '/scratch/utrerf/trigger-attack/trojai_submission/classifier.joblib'

args_defaults = {
    '--model_filepath': '/scratch/utrerf/trigger-attack/trigger_attack/'
                        'trojan_model_datasets/round9-train-dataset/models'
                        '/id-00000007/model.pt',
    '--tokenizer_filepath': '/scratch/utrerf/trigger-attack/trigger_attack/'
                            'trojan_model_datasets/round9-train-dataset/'
                            'tokenizers/distilbert-base-cased.pt',
    '--round_training_dataset_dirpath': '/scratch/utrerf/trigger-attack/'
                                        'trigger_attack/trojan_model_datasets'
                                        '/round9-train-dataset',
    '--result_filepath': '/scratch/utrerf/trigger-attack/'
                         'trojai_submission/result.txt',
}

rounds_to_taks = {
    5: 'sc',
    6: 'sc',
    7: 'ner',
    8: 'qa'
}
