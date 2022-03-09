from cmath import inf
from copy import deepcopy
from hashlib import new
import torch
import heapq
from operator import itemgetter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


class TriggerReconstructor():

    def __init__(self,
                 trigger_initializator,
                 dataset_preprocessor,
                 loss_fn,
                 batch_size=20):
        self.trigger_models = dataset_preprocessor.trigger_models
        self.tokenizer = dataset_preprocessor.tokenizer
        self.trigger_length = len(dataset_preprocessor.trigger.input_ids)
        self.preprocessor = dataset_preprocessor
        self.batch_size = batch_size

        self.initialize_dataloader()

        self.trigger_masks = []
        self.suspicious_embeddings = self.trigger_models.get_suspicious_model_embeddings()
        all_clean_embeddings = self.trigger_models.get_all_clean_models_embeddings()

        self.avg_clean_embeddings = torch.stack(all_clean_embeddings).mean(0)
        self.embeddings_shape = self._get_embeddings_shape()
        self.loss_fn = loss_fn
        self.trigger_initializator = trigger_initializator

    def initialize_dataloader(self, new_trigger=None):
        if new_trigger is not None:
            self.preprocessor.trigger = new_trigger
        dataset = self.preprocessor.preprocess_data()
        self.dataloader = DataLoader(dataset, self.batch_size)

    def _get_embeddings_shape(self):
        embeddings_shape = [
            -1,
            self.trigger_length,
            self.avg_clean_embeddings.shape[-1]
        ]
        return embeddings_shape

    @autocast()
    def reconstruct_trigger(self,
                            trigger_target,
                            num_candidates_per_token,
                            beam_size=1,
                            max_iter=20,
                            loss_threshold=1e-3,
                            trigger_init_fn='embedding_change'):
        first_candidate = {
            'input_ids':self.trigger_initializator.make_initial_trigger(
                self.trigger_length, trigger_init_fn),
            'loss': 100
        }
        self._insert_new_candidate(first_candidate)
        loss_value = inf
        remaining_iter = max_iter
        pbar = tqdm(total=max_iter)
        old_trigger = torch.zeros(self.dataloader.dataset.trigger_length())-100
        old_trigger = old_trigger.long()
        while (remaining_iter > 0 and
               loss_value > loss_threshold and
               not torch.equal(old_trigger, self.dataloader.dataset.trigger)):
            self.trigger_masks = []
            self.trigger_models.clear_word_embedding_gradients()
            old_trigger = deepcopy(self.dataloader.dataset.trigger)
            loss_value = self._calculate_loss(
                trigger_target, extract_embedding_gradients=True)
            candidates = self._get_candidates(num_candidates_per_token)
            best_candidate = self._pick_best_candidate(
                loss_value, candidates, trigger_target, beam_size)
            self._insert_new_candidate(best_candidate)
            remaining_iter -= 1
            trigger_text = self.tokenizer.decode(self.dataloader.dataset.trigger)
            pbar.set_description((
                f"Loss: {deepcopy(loss_value):.3f} -> {deepcopy(best_candidate['loss']):.3f} | "
                f"Trigger: {trigger_text}"
                ))
            pbar.update(1)
        pbar.close()
        return best_candidate

    @torch.no_grad()
    def _get_candidates(self, num_candidates_per_token):
        self._put_embeddings_on_device(torch.device('cuda'))

        trigger_mask = torch.cat(self.trigger_masks)
        concatenated_suspicious_gradients = torch.cat(self.trigger_models.suspicious_grads)
        suspicious_gradients = self._filter_trigger_embeddings(
            concatenated_suspicious_gradients, trigger_mask)
        avg_suspicious_gradients = torch.mean(
            suspicious_gradients, dim=0)
        embedding_tuple = (avg_suspicious_gradients,  self.suspicious_embeddings)
        suspicious_grad_dot_embedding_matrix = torch.einsum("ij,kj->ik",
                                                            embedding_tuple)
        num_models = len(self.trigger_models.clean_models)
        mean_clean_gradients = self._mean_embeddings_over_models(
            num_models, self.trigger_models.clean_grads)
        clean_gradients = self._filter_trigger_embeddings(mean_clean_gradients,
                                                          trigger_mask)
        avg_clean_gradients = torch.mean(clean_gradients, dim=0)
        embedding_tuple = (avg_clean_gradients,  self.avg_clean_embeddings)
        clean_grad_dot_embedding_matrix = torch.einsum("ij,kj->ik", embedding_tuple)

        grad_dot_list = [
            suspicious_grad_dot_embedding_matrix,
            clean_grad_dot_embedding_matrix
            ]
        combined_grad_dot_embedding_matrix = torch.stack(grad_dot_list).mean(dim=0)
        best_values, best_input_ids = torch.topk(
            -combined_grad_dot_embedding_matrix, num_candidates_per_token, dim=1)

        self._put_embeddings_on_device(torch.device('cpu'))
        candidates = {
            'values': best_values,
            'input_ids': best_input_ids
        }
        return candidates

    def _put_embeddings_on_device(self, device):
        self.suspicious_embeddings = self.suspicious_embeddings.to(device, non_blocking=True)
        self.avg_clean_embeddings = self.avg_clean_embeddings.to(device, non_blocking=True)

    def _filter_trigger_embeddings(self, embeddings, trigger_mask):
        return embeddings[trigger_mask].view(self.embeddings_shape)

    @staticmethod
    def _mean_embeddings_over_models(num_models, gradients):
        list_of_resorted_embeddings = [torch.cat(gradients[i::num_models])
                                       for i in range(num_models)]
        stacked_embeddings = torch.stack(list_of_resorted_embeddings)
        return torch.mean(stacked_embeddings, dim=0)

    @torch.no_grad()
    def _pick_best_candidate(self,
                             loss_value,
                             candidates,
                             trigger_target,
                             beam_size):
        best_candidate = {
            'input_ids': deepcopy(self.dataloader.dataset.trigger),
            'loss': deepcopy(loss_value)
        }
        skip_evaluation = torch.isclose(
            candidates['values'].sum(), torch.tensor(.0))
        if skip_evaluation:
            return best_candidate
        evaluated_candidates = self._evaluate_candidates(
            candidates, best_candidate, trigger_target, ix=0)
        top_candidates = heapq.nsmallest(
            beam_size, evaluated_candidates, key=itemgetter('loss'))

        for i in range(1, self.trigger_length):
            evaluated_candidates = []
            for best_candidate in top_candidates:
                temp_candidates = self._evaluate_candidates(
                    candidates, best_candidate, trigger_target, ix=i)
                evaluated_candidates.extend(temp_candidates)
            top_candidates = heapq.nsmallest(
                beam_size, evaluated_candidates, key=itemgetter('loss'))
        best_candidate = min(top_candidates, key=itemgetter('loss'))
        return best_candidate

    def _evaluate_candidates(self, candidates, best_candidate, trigger_target, ix=0):
        evaluated_candidates = [best_candidate]
        visited_triggers = set(best_candidate['input_ids'])

        for candidate_token in candidates['input_ids'][ix]:
            temp_candidate = deepcopy(best_candidate)
            if temp_candidate['input_ids'] in visited_triggers:
                continue
            temp_candidate['input_ids'][ix] = candidate_token
            self._insert_new_candidate(temp_candidate)
            temp_candidate['loss'] = self._calculate_loss(trigger_target)
            evaluated_candidates.append(deepcopy(temp_candidate))
            visited_triggers.add(temp_candidate['input_ids'])

        return evaluated_candidates

    def _insert_new_candidate(self, new_candidate):
        self.dataloader.dataset.update_trigger(new_candidate['input_ids'])

    def _calculate_loss(self,
                        trigger_target,
                        is_test=False,
                        extract_embedding_gradients=False):
        loss_aggregator = {'loss': 0, 'num_items': 0}
        for batch in self.dataloader:
            batch = self._put_batch_on_models_device(batch, self.trigger_models)
            all_logits = self.trigger_models(batch, is_test)
            loss = self.loss_fn.calculate_loss(
                        all_logits, batch, trigger_target)
            loss_aggregator = self._aggregate_loss(loss_aggregator, loss, batch)
            if extract_embedding_gradients:
                loss.backward()
                self.trigger_models.clear_model_gradients()
                self._save_trigger_mask(batch)
        return loss_aggregator['loss']

    @staticmethod
    def _aggregate_loss(loss_aggregator, loss, batch):
        old_loss_sum = loss_aggregator['loss'] * loss_aggregator['num_items']

        new_loss = loss.detach().to(torch.device('cpu'))
        new_num_items = len(batch['input_ids'])
        new_loss_sum = new_loss * new_num_items

        new_loss_agg = {}
        new_loss_agg['num_items'] = loss_aggregator['num_items'] + new_num_items
        new_loss_agg['loss'] = (old_loss_sum + new_loss_sum)/new_loss_agg['num_items']
        return new_loss_agg

    def _save_trigger_mask(self, batch):
        self.trigger_masks.append(deepcopy(batch['trigger_mask']))

    @staticmethod
    def _put_batch_on_models_device(batch, trigger_models):
        if batch['input_ids'].device != trigger_models.device:
            new_batch = {
                k: v.to(trigger_models.device, non_blocking=True)
                for k, v in batch.items()}
            return new_batch
        else:
            return batch
