import torch

from trigger_attack.trigger_dataset import TriggeredDataset


# inputs: loss fn and it's inputs, search algo
# output: an sequence of tokens that minimizes some loss using some search algo

def get_trigger(dataloader, search_fn, loss_fn, models, num_iter):
    triggered_dataset.init_trigger(models)

    while triggered_dataset.check_if_trigger_changed() and num_iter > 0:
        new_trigger = search_fn(triggered_dataset, loss_fn, models)
        triggered_dataset.insert_new_trigger(new_trigger)
        num_iter -= 1

def gradient_based_discrete_search(triggered_dataset, models, loss_fn):
    loss = loss_fn(triggered_dataset, models)
    