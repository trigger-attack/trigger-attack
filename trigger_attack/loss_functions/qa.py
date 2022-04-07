import torch
from .trigger_loss import triggerLoss


class QALoss(triggerLoss):
    def __init__(self, agg_fn=torch.mean):
        super().__init__(agg_fn)

    def _get_logits(self, model_output):
        return model_output

    def _get_probabilitites(self, logits, **kwargs):
        combined_logits = logits['start_logits'] + logits['end_logits']
        mask = (~kwargs['batch']['valid_mask'].bool())*1e10
        valid_logits = combined_logits - mask
        scores = torch.exp(valid_logits)
        probs = scores/torch.sum(scores, dim=1, keepdim=True)
        return probs

    def _get_target_label_probabilities(self, probabilities, **kwargs):
        CLS_IDX = 0
        num_rows = len(kwargs['batch']['input_ids'])
        if kwargs['target_label'] == 'cls':
            filtered_probabilities = probabilities[:, CLS_IDX].unsqueeze(-1)
        else:
            filtered_probabilities = \
                probabilities[kwargs['batch']['trigger_mask']]
            filtered_probabilities = \
                filtered_probabilities.reshape(num_rows, -1)
        return torch.sum(filtered_probabilities, dim=-1)
