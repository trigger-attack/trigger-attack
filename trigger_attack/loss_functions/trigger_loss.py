from abc import ABC, abstractmethod
import torch


class triggerLoss(ABC):
    def __init__(self, agg_fn=torch.mean):
        self.agg_fn = agg_fn

    def calculate_asr(self, all_logits, batch, target_label: list):
        result ={
            'suspicious_asr': self.get_asr(
                'suspicious', all_logits, batch, target_label),
            'clean_asr': self.get_clean_asr(
                'clean', all_logits, batch, target_label)
        }
        return result

    def get_asr(self, type, all_logits, batch, target_label):
        logits = self._get_logits(all_logits[type])
        target_probability_kwargs = {
            'batch': batch,
            'target_label': target_label
            }
        if type == 'clean':
            probabilities = self._get_agg_clean_probabilities(
                logits, batch)
        else:
            probabilities = self._get_probabilitites(
                logits, batch)
    

    def calculate_loss(self, all_logits, batch, target_label: list):
        suspicious_loss = self._calculate_suspicious_loss(
                               all_logits, batch, target_label)
        clean_loss = self._calculate_clean_loss(
                          all_logits, batch, target_label)
        return suspicious_loss + clean_loss

    def _calculate_suspicious_loss(self, all_logits, batch, target_label):
        suspicious_logits = self._get_logits(all_logits['suspicious'])
        suspicious_probabilities = self._get_probabilitites(
            suspicious_logits, batch=batch)
        target_probability_kwargs = {
            'batch': batch,
            'target_label': target_label
            }
        suspicious_target_probabilities = self._get_target_label_probabilities(
                        suspicious_probabilities, **target_probability_kwargs)
        return torch.mean(-torch.log(suspicious_target_probabilities))

    def _calculate_clean_loss(self, all_logits, batch, target_label):
        agg_clean_probabilities = self._get_agg_clean_probabilities(
            all_logits, batch)
        target_probability_kwargs = {
            'batch': batch,
            'target_label': target_label}
        clean_target_probabilities = self._get_target_label_probabilities(
                        agg_clean_probabilities, **target_probability_kwargs)
        baseline_clean_target_probabilities = self._get_target_label_probabilities(
                        batch['baseline_probabilities'], **target_probability_kwargs)
        non_negative_net_probabilities = self._get_non_negative_net_probabilities(
                        clean_target_probabilities, baseline_clean_target_probabilities)
        return torch.mean(-torch.log(1-non_negative_net_probabilities))
        # return torch.mean(-torch.log(1-clean_target_probabilities))

    @abstractmethod
    def _get_logits(self, model_output):
        pass

    @abstractmethod
    def _get_probabilitites(self, logits, **kwargs):
        pass

    @abstractmethod
    def _get_target_label_probabilities(self, probabilities, **kwargs):
        pass

    def _get_agg_clean_probabilities(self, all_logits, batch):
        probabilities = []
        for model_output in all_logits['clean']:
            logits = self._get_logits(model_output)
            probs = self._get_probabilitites(logits, batch=batch)
            probabilities.append(probs)
        stacked_probabilities = torch.stack(probabilities)
        agg_clean_probabilities = self.agg_fn(stacked_probabilities, dim=0)
        return agg_clean_probabilities

    @staticmethod
    def _get_non_negative_net_probabilities(
            clean_target_probabilities,
            baseline_clean_target_probabilities):
        net_clean_probabilities = (
            clean_target_probabilities - baseline_clean_target_probabilities)
        zeros = torch.zeros_like(net_clean_probabilities,
                                 device=net_clean_probabilities.device)
        non_negative_net_probabilities = torch.max(net_clean_probabilities,
                                                   zeros)
        return non_negative_net_probabilities
