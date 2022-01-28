import pandas as pd
import torch
import submission_tools
import data_tools
from trigger_attack.trigger_models import TriggerModels
from trigger_attack.trigger_dataset import TriggerDataset
from torch.utils.data import DataLoader

def extract_trojan_features(args):
    config = data_tools.load_config(args.model_filepath)
    clean_model_filepaths = data_tools.get_clean_model_filepaths(config, args.round_training_dataset_dirpath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = TriggerModels(args.model_filepath, clean_model_filepaths, args.tokenizer_filepath, device=device)
    
    original_dataset = data_tools.load_examples(args.model_filepath, args.scratch_dirpath)
    task = data_tools.get_taskname(args.round_training_dataset_dirpath, config)
    trigger_dataset = TriggerDataset(original_dataset, models, args.trigger_length, args.trigger_init_fn, 
                                            args.trigger_loc, task, scratch_dirname=args.scratch_dirpath)
    dataloader = DataLoader(trigger_dataset, batch_size=20, num_workers=4, shuffle=True, pin_memory=True)
    

    print('break')

def reconfigure_trojan_classifier(args): 
    submission_tools.write_features_for_all_models(args)
    extracted_features_folder = submission_tools._get_extracted_features_folder(args.scratch_path)
    features = pd.read_csv(extracted_features_folder)
    # metadata = get_metadata_from_
    # labels = get_labels_corresponding_to_features_from_metadata(metadata, features)
    # classifier = configure_tools.retrain_classifier()
    # save_new_features(classifier)

if __name__ == "__main__":
    args = submission_tools.get_args()
    submission_tools.validate_config_file(args)
    if not args.configure_mode:
        extract_trojan_features(args)
        
    else:
        reconfigure_trojan_classifier(args)