import torch
from torch.utils.data import Dataset
from copy import deepcopy


class TorchTriggeredDataset(Dataset):

    def __init__(self, dataset, trigger_length):
        self.input_ids = dataset['input_ids'].clone().detach().long()
        self.attention_mask = dataset['attention_mask'].clone().detach().bool()
        self.token_type_ids = dataset['token_type_ids'].clone().detach().long()

        self.baseline_probabilities = dataset['baseline_probabilities'].clone()
        self.baseline_probabilities = self.baseline_probabilities.detach()
        self.baseline_probabilities = self.baseline_probabilities.float()

        self.trigger_mask = dataset['trigger_mask'].clone().detach().bool()
        self.trigger = torch.zeros(trigger_length).long()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        sample = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'baseline_probabilities': self.baseline_probabilities[idx],
            'trigger_mask': self.trigger_mask[idx],
        }
        return sample

    def update_trigger(self, new_trigger):
        if not isinstance(new_trigger, torch.Tensor):
            new_trigger = torch.tensor(new_trigger,
                                       device=torch.device('cpu'),
                                       dtype=torch.long)
        new_trigger = new_trigger.to(self.input_ids.device).long()
        new_trigger = deepcopy(new_trigger)
        target_size = (self.input_ids[self.trigger_mask]).shape[0]
        num_copies = target_size // len(new_trigger)
        self.input_ids[self.trigger_mask] = new_trigger.repeat(num_copies)
        self.trigger = new_trigger

    def trigger_length(self):
        return len(self.trigger)
