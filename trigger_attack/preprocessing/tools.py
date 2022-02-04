import torch
from torch.utils.data import Dataset

@torch.no_grad()
def _select_unique_inputs(dataset):
    unique_ixs_ids = torch.tensor(dataset['input_ids']).unique(dim=0, return_inverse=True)[1].flatten()
    seen = set()
    unique_ixs = []
    for source_ix, target_ix in enumerate(unique_ixs_ids):
        if target_ix.item() not in seen:
            seen.add(target_ix.item())
            unique_ixs.append(source_ix)
    return dataset.select(unique_ixs)

def _get_max_seq_length(tokenizer):
    max_seq_length = min(tokenizer.model_max_length, 384)
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    return max_seq_length

class TorchTriggeredDataset(Dataset):

    def __init__(self, huggingface_dataset):
        self.input_ids = huggingface_dataset['input_ids'].clone().detach().clone().long()
        self.attention_mask = huggingface_dataset['attention_mask'].clone().detach().clone().bool()
        self.token_type_ids = huggingface_dataset['token_type_ids'].clone().detach().clone().long()

        self.baseline_probabilities = huggingface_dataset['baseline_probabilities'].detach().clone().float()
        self.trigger_mask = huggingface_dataset['trigger_mask'].detach().clone().bool()

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
        _insert_new_trigger_in_input_ids(self.input_ids, self.trigger_mask, new_trigger)

def _insert_new_trigger_in_input_ids(input_ids, trigger_mask, new_trigger):
    if not isinstance(new_trigger, torch.Tensor):
        new_trigger = torch.tensor(new_trigger, device=torch.device('cpu'))
    new_trigger = new_trigger.to(input_ids.device)
    target_size = (input_ids[trigger_mask]).shape[0]
    num_copies = target_size // len(new_trigger)
    input_ids[trigger_mask] = new_trigger.repeat(num_copies)
