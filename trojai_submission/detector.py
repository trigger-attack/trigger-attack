import pandas as pd
import torch
import transformers
import os
import json
import submission_tools
import data_tools
from trigger_attack.trigger_models import TriggerModels
from trigger_attack.embeddings_analyzer import EmbeddingsAnalyzer
from trigger_attack.trigger_initializer import TriggerInitializator
from trigger_attack.trigger_reconstructor import TriggerReconstructor
from trigger_attack.trigger import Trigger
from itertools import permutations, product
import configure_tools


def extract_trojan_features(args):
    config = data_tools.load_config(args.model_filepath)

    suspicious_model_folder = args.model_filepath.split('/')[-2]
    parent_dirpath = os.path.join(args.scratch_dirpath, args.unique_id)
    destination = os.path.join(parent_dirpath, suspicious_model_folder)
    if not os.path.isdir(parent_dirpath):
        os.mkdir(parent_dirpath)
        args_dict = args.as_dict()
        args_dict['metaparameters_filepath'] = \
            str(args_dict['metaparameters_filepath'])
        args_dict = json.dumps(args_dict)
        with open(os.path.join(parent_dirpath, 'args.json'), 'w') as f:
            f.write(args_dict)
    if os.path.exists(destination):
        return None

    clean_model_filepaths = data_tools.get_clean_model_filepaths(
        config, args.round_training_dataset_dirpath, suspicious_model_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trigger_models = TriggerModels(
        args.model_filepath, clean_model_filepaths,
        args.num_clean_models, device)
    if os.path.exists(args.tokenizer_filepath):
        tokenizer = torch.load(args.tokenizer_filepath, map_location=device)
    else:
        model_architecture = config['model_architecture']
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_architecture, use_fast=True, add_prefix_space=True)

    embeddings_analyzer = EmbeddingsAnalyzer(trigger_models, tokenizer)

    trigger_initializator = TriggerInitializator(
        tokenizer, trigger_models.device, embeddings_analyzer)

    task = data_tools.get_taskname(
        args.round_training_dataset_dirpath, config)

    more_data_filepaths = clean_model_filepaths[:1]
    if task in ['sc', 'ner']:
        more_data_filepaths = []
    # more data might solve the low loss issue
    dataset = data_tools.load_examples(
        args.model_filepath, args.scratch_dirpath, more_data_filepaths)

    is_wrong_task = args.task != 'None' and args.task != task
    if is_wrong_task:
        return None

    locations = data_tools.task_to_trigger_locations[task]
    labels = data_tools.get_task_labels(task, config)

    dummy_trigger = Trigger(
        torch.tensor([1]*args.trigger_length), locations[0], labels[0])

    preprocessor = data_tools.task_to_preprocessor[task](
            dataset, dummy_trigger, trigger_models, tokenizer)

    loss_fn = data_tools.task_to_loss[task]()

    trigger_reconstructor = TriggerReconstructor(
        trigger_initializator, preprocessor, loss_fn, args.batch_size)

    source_target_labels = list(permutations(labels, 2))
    source_target_locations = list(product(source_target_labels, locations))
    triggers_target_labels = [
        {'trigger': Trigger(
            dummy_trigger.input_ids, location, source_target[0]),
         'trigger_target': source_target[1]
         } for source_target, location in source_target_locations
    ]

    results = pd.DataFrame([])
    for objective in triggers_target_labels:
        trigger_reconstructor.initialize_dataloader(objective['trigger'])

        best_candidate = {
            'input_ids': torch.tensor([0]),
            'loss': torch.tensor(100.),
            'test_loss': torch.tensor(100.)
        }

        reinitializations = {
            'sc': args.num_reinitializations,
            'ner': args.num_reinitializations,
            'qa': args.num_reinitializations
        }

        print(f"{objective['trigger'].source_labels}"
              f" -> {objective['trigger_target']}")
        for _ in range(reinitializations[task]):
            temp_candidate = trigger_reconstructor.reconstruct_trigger(
                objective['trigger_target'],
                args.num_candidates_per_token,
                max_iter=args.max_iter)
            with torch.no_grad():
                temp_candidate['test_loss'] = \
                    trigger_reconstructor._calculate_loss(
                    objective['trigger_target'], is_test=True)
            if temp_candidate['test_loss'] < best_candidate['test_loss']:
                best_candidate = temp_candidate
            if best_candidate['test_loss'] < args.test_loss_threshold:
                break
        new_results = {
            'trigger': [tokenizer.decode(best_candidate['input_ids'])],
            'loss': [best_candidate['loss'].item()],
            'test_loss': [best_candidate['test_loss'].item()],
            'location': [objective['trigger'].location],
            'source_label': [objective['trigger'].source_labels],
            'target_label': [objective['trigger_target']],
            'task': task
        }
        if 'poisoned' in config:
            new_results['poisoned'] = config['poisoned']
        results = pd.concat([results, pd.DataFrame(new_results)])
    results = results.reset_index(drop=True)

    results.to_csv(f'{destination}.csv')
    return results


def reconfigure_trojan_classifier(args):
    configure_tools.write_features_for_all_models(args)
    folder = submission_tools.get_extracted_features_folder(args)
    features = submission_tools.read_all_features(folder)
    # metadata = get_metadata_from_
    # labels = get_labels_corresponding_to_features_from_metadata(
    #   metadata, features)
    # classifier = configure_tools.retrain_classifier()
    # save_new_features(classifier)


if __name__ == "__main__":
    args = submission_tools.get_args()
    # submission_tools.validate_config_file(args)
    if not args.configure_mode:
        features = extract_trojan_features(args)
        prediction = submission_tools.get_predictions(features)
        with open(args.result_filepath, 'w') as fh:
            fh.write("{}".format(prediction))
    else:
        reconfigure_trojan_classifier(args)
