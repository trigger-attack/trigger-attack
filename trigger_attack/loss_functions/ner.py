import torch
from .trigger_loss import triggerLoss


class NERLoss(triggerLoss):
    def __init__(self, agg_fn=torch.mean):
        super().__init__(agg_fn)

    def _get_logits(self, model_output):
        return model_output['logits']

    def _get_probabilitites(self, logits, **kwargs):
        trigger_source_loc = kwargs['batch']['trigger_source_loc']
        source_loc_logits = \
            logits[torch.arange(len(logits)), trigger_source_loc]
        scores = torch.exp(source_loc_logits)
        probs = scores/torch.sum(scores, dim=1, keepdim=True)
        return probs

    def _get_target_label_probabilities(self, probabilities, **kwargs):
        return torch.sum(probabilities[:, kwargs['target_label']], dim=1)
