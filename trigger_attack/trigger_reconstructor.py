from cmath import inf
from copy import deepcopy
import torch
import heapq
from operator import itemgetter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np


class TriggerReconstructor():

    def __init__(self,
                 trigger_initializator,
                 dataset_preprocessor,
                 loss_fn,
                 batch_size=20):
        self.trigger_models = dataset_preprocessor.trigger_models
        self.tokenizer = dataset_preprocessor.tokenizer
        self.trigger_length = len(dataset_preprocessor.trigger.input_ids)
        dataset = dataset_preprocessor.preprocess_data()
        self.dataloader = DataLoader(dataset, batch_size=batch_size)

        self.trigger_masks = []
        self.suspicious_embeddings = self.trigger_models.get_suspicious_model_embeddings()
        all_clean_embeddings = self.trigger_models.get_all_clean_models_embeddings()

        self.avg_clean_embeddings = torch.stack(all_clean_embeddings).mean(0)
        self.embeddings_shape = self._get_embeddings_shape()
        self.loss_fn = loss_fn   #TODO loss_fn.calculate_loss()
        self.trigger_initializator = trigger_initializator

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
        '''
        for both suspicious and clean models
        find the smallest k embeddings for which
            embedding.T @ trigger_gradient
        use this for reference:
            https://github.com/utrerf/TrojAI/blob/master/NLP/round8/detector.py
        '''
        '''
        YOUR CODE HERE
        '''
        self._put_embeddings_on_device('cuda')
        suspicious_grad = self.trigger_models.suspicious_grads[0]
        dim0, dim2 = suspicious_grad.shape[0], suspicious_grad.shape[2]
        suspicious_grad = suspicious_grad[self.dataloader.dataset.trigger_mask].reshape(dim0,-1, dim2)
        suspicious_grad = torch.mean(suspicious_grad, dim=0)
        suspicious_grad_dot_embed_matrix = torch.einsum("ik,jk->ij", (suspicious_grad, self.suspicious_embeddings))
        mean_clean_grads = torch.stack(self.trigger_models.clean_grads).mean(dim=0) 
        mean_clean_grads = mean_clean_grads[self.dataloader.dataset.trigger_mask].reshape(dim0,-1, dim2)
        mean_clean_grads = torch.mean(mean_clean_grads, dim=0)    
        clean_grad_dot_embed_matrix = torch.einsum("ik,jk->ij", (mean_clean_grads, self.avg_clean_embeddings))
        gradient_dot_embedding_matrix = suspicious_grad_dot_embed_matrix + clean_grad_dot_embed_matrix 
        _, best_k_ids = torch.topk(-gradient_dot_embedding_matrix, num_candidates_per_token, dim=-1)
        return best_k_ids

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
    def evaluate_loss_with_temp_trigger(self, new_candidate, trigger_target):
        self._insert_new_candidate(new_candidate)
        loss = self._calculate_loss(trigger_target)
        return {'input_ids': deepcopy(new_candidate['input_ids']), 'loss':loss}

    @torch.no_grad()
    def evaluate_candidate_tokens_for_pos(self, candidates, triggered_dataset, top_candidate, pos):
        top_cand = deepcopy(top_candidate['input_ids'])
        loss_per_candidate_trigger = [deepcopy(top_candidate)]
        visited_triggers = set(top_cand)

        for cand in candidates[pos]:
            temp_trigger = top_cand
            # return temp_trigger
            temp_trigger[pos] = cand
            if temp_trigger in visited_triggers:
                    continue
            temp_candidate = {'input_ids':temp_trigger, 'loss':0.}
            temp_result = self.evaluate_loss_with_temp_trigger(temp_candidate, triggered_dataset)
            loss_per_candidate_trigger.append(temp_result)
            visited_triggers.add(temp_result['input_ids'])
        return loss_per_candidate_trigger

    @torch.no_grad()
    def _pick_best_candidate(self, loss_value, candidates, trigger_target,
                                                                beam_size):
        best_candidate = {
            'input_ids': deepcopy(self.dataloader.dataset.trigger),
            'loss': deepcopy(loss_value)
        }
        '''
        find which of the candidates is best -> has the smallest loss
        use this for reference:
            https://github.com/utrerf/TrojAI/blob/master/NLP/round8/detector.py
        '''
        top_candidate = best_candidate
        loss_per_candidate_trigger = self.evaluate_candidate_tokens_for_pos(candidates, trigger_target, top_candidate, pos=0)
        top_candidates = heapq.nsmallest(beam_size, loss_per_candidate_trigger, key=itemgetter('loss'))
        # return top_candidates, loss_per_candidate_trigger
        
        for idx in range(1, len(best_candidate['input_ids'])):
            loss_per_candidate_trigger = []
            for top_candidate in top_candidates:
                loss_per_candidate_trigger.extend(self.evaluate_candidate_tokens_for_pos(candidates, trigger_target, top_candidate, pos=idx))
            top_candidates = heapq.nsmallest(beam_size, loss_per_candidate_trigger, key=itemgetter('loss'))
        best_candidate = min(top_candidates, key=itemgetter('loss'))
        return best_candidate

    def _insert_new_candidate(self, new_candidate):
        self.dataloader.dataset.update_trigger(new_candidate['input_ids'])

    def _calculate_loss(self, trigger_target, extract_embedding_gradients=False):
        loss_aggregator = {'loss': 0, 'num_items': 0}
        '''
        iterate through self.dataloader
        get logits
        get loss
        aggregate loss using the agregate_loss fn
        extract embedding gradients by doing .backward on loss (if necessary)
        return loss_aggregator['loss']
        '''
        '''
        YOUR CODE HERE
        '''
        for batch in self.dataloader:
            for key in batch.keys():
                batch[key] = batch[key].to('cuda', non_blocking=True)
            all_logits = self.trigger_models(batch)
            loss = self.loss_fn.calculate_loss(all_logits=all_logits, batch=batch, target_label=trigger_target)
            loss_aggregator = self._aggregate_loss(loss_aggregator, loss, batch)
            if extract_embedding_gradients:
                loss.backward()
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
