import torch


class TriggeredDataset():
    def __init__(self, original_dataset, tokenizer, trigger_length, trigger_init_fn, device, most_changed_tokens=None):
        
        self.old_trigger = None
        self.current_trigger = None

        self.original_dataset = original_dataset
        self.triggered_dataset = None

        self.tokenizer = tokenizer

        self.trigger_length = trigger_length
        self.trigger_init_fn = trigger_init_fn

        self.most_changed_tokens = most_changed_tokens

        self.device = device

        self._generate_triggered_dataset(self)


    def _generate_triggered_dataset(self):
        # expands the dataset by the trigger length and inserts sensible values for it
        return NotImplementedError

    def insert_new_trigger(new_trigger):
        self.old_trigger = self.current_trigger
        self.current_trigger = new_trigger
        # actually insert the new trigger
        return NotImplementedError

    def init_trigger(self, models):
        self.old_trigger = None
        self.current_trigger = self.trigger_init_fn(self, models)
        self.insert_new_trigger(self.current_trigger)

    def check_if_trigger_changed(self):
        return not torch.equal(self.current_trigger, self.old_trigger)