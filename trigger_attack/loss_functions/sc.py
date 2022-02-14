import torch
from .trigger_loss import triggerLoss

class SCLoss(triggerLoss):
    def __init__(self, agg_fn=torch.mean):
        super().__init__(agg_fn)

    def _get_logits(self, model_output):
        return model_output['logits']

    def _get_probabilitites(self, logits, **kwargs):
        scores = torch.exp(logits)
        probs = scores/torch.sum(scores, dim=1, keepdim=True)
        return probs

    def _get_target_label_probabilities(self, probabilities, **kwargs):
        return probabilities[:, kwargs['target_label']]