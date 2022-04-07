import os
import shutil
import time
import subprocess
import shlex


def write_features_for_all_models(args):
    commands_to_run = _get_commands_to_run(args)
    run_commands(commands_to_run, args.gpu, gpus_per_command=1)


def _get_commands_to_run(args):
    all_models_dirpath = _get_all_models_dirpath(args.configure_models_dirpath)

    extracted_features_folder = \
        _get_extracted_features_folder(args.scratch_dirpath)
    _prepare_extracted_features_folder(extracted_features_folder)

    commands_to_run = []
    for model_dirpath in all_models_dirpath:

        model_filepath = os.path.join(model_dirpath, 'model.pt')
        examples_dirpath = \
            _get_examples_dirpath_from_model_dirpath(model_dirpath)

        commands_to_run.append(
            f'python detector.py '
            f'--model_filepath {model_filepath} '
            f'--examples_dirpath {examples_dirpath} '
            f'--schema_filepath {args.schema_filepath} '
            f'--trigger_length {args.trigger_length} '
            f'--trigger_init_fn {args.trigger_init_fn} '
            f'--num_clean_models {args.num_clean_models} '
            f'--num_reinitializations {args.num_reinitializations} '
            f'--num_candidates_per_token {args.num_candidates_per_token} '
            f'--unique_id {args.unique_id} '
            f'--task {args.task} '
        )

    return commands_to_run


def _get_extracted_features_folder(scratch_dirpath):
    return os.path.join(scratch_dirpath, 'extracted_features')


def _prepare_extracted_features_folder(extracted_features_folder):
    _delete_folder_if_possible(extracted_features_folder)
    _make_folder_if_necessary(extracted_features_folder)


def _make_folder_if_necessary(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def _delete_folder_if_possible(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)


def _get_all_models_dirpath(configure_models_dirpath):
    all_model_filepaths = []
    for model_foldername in os.listdir(configure_models_dirpath):
        model_filepath = os.path.join(
            configure_models_dirpath, model_foldername)
        all_model_filepaths.append(model_filepath)
    return all_model_filepaths


def _get_examples_dirpath_from_model_dirpath(model_dirpath):
    return os.path.join(model_dirpath, 'clean_examples')


def run_commands(commands_to_run, gpu_list, gpus_per_command=1):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    pid_to_process_and_gpus = {}
    free_gpus = set(gpu_list)
    while len(commands_to_run) > 0:
        time.sleep(.1)
        # try kicking off process if we have free gpus
        print(f'free_gpus: {free_gpus}')
        while len(free_gpus) >= gpus_per_command:
            print(f'free_gpus: {free_gpus}')
            gpus = []
            for i in range(gpus_per_command):
                # updates free_gpus
                gpus.append(str(free_gpus.pop()))
            command = commands_to_run.pop()
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
            subproc = subprocess.Popen(shlex.split(command))
            # updates_dict
            pid_to_process_and_gpus[subproc.pid] = (subproc, gpus)

        # update free_gpus
        pids_processes_and_gpus = pid_to_process_and_gpus.copy().items()
        for pid, (current_process, gpus) in pids_processes_and_gpus:
            if _poll_process(current_process) is not None:
                print(f'done with {pid}')
                free_gpus.update(gpus)
                del pid_to_process_and_gpus[pid]


def _poll_process(process, polling_delay_seconds=.1):
    time.sleep(polling_delay_seconds)
    return process.poll()


def read_extracted_features_folder(extracted_features_folder):
    return NotImplementedError


def get_labels_corresponding_to_features_from_metadata(metadata, features):
    return NotImplementedError


def retrain_classifier():
    return NotImplementedError


def save_new_features():
    return NotImplementedError
