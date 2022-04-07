import torch
import re
import multiprocessing
from copy import deepcopy
from itertools import permutations
import enchant


TOOL = enchant.Dict("en_US")


@torch.no_grad()
def check_word(tup):
    ix, cand = tup
    if TOOL.check(cand):
        return ix


class EmbeddingsAnalyzer():
    def __init__(self, trigger_models, tokenizer, max_suffixes=5):
        self.suspicious_embeddings = \
            trigger_models.get_suspicious_model_embeddings()
        all_clean_embeddings = \
            trigger_models.get_all_clean_train_and_test_models_embeddings()
        self.avg_clean_embeddings = torch.stack(all_clean_embeddings).mean(0)

        self.max_suffixes = max_suffixes
        self.singletoken_options = None
        self.multitoken_options = None
        self.device = trigger_models.device
        self.tokenizer = tokenizer

    @torch.no_grad()
    def populate_most_changed_embeddings(self):
        '''
        example:
        finally is decomposed as
            final: prefix
            ly: suffix

        consider the prefix car
        if we append ly to car, we have carly is a word

        however, computerly is not, so we want to exclude it
        '''
        if self._options_are_not_populated():

            dissimilar_token_ids = self._get_smallest_cosine_similarity()

            dissimilar_token_ids = dissimilar_token_ids.to(self.device)
            top_ids_to_tokens = {}
            for top_id in dissimilar_token_ids:
                token = self.tokenizer.convert_ids_to_tokens([top_id])[0]
                top_ids_to_tokens[top_id] = token
            dissimilar_token_ids = self._remove_non_words(top_ids_to_tokens)

            suffixes = self._get_suffixes(dissimilar_token_ids)
            suffixes = suffixes[:self.max_suffixes]
            prefixes = self._get_prefixes(dissimilar_token_ids, suffixes)
            multitoken_words = self._get_multitoken_words(prefixes, suffixes)

            self.singletoken_options = prefixes[:8]
            self.multitoken_options = multitoken_words

    def _options_are_not_populated(self):
        singletoken_is_populated = self.singletoken_options is None
        multitoken_is_populated = self.multitoken_options is None
        return singletoken_is_populated or multitoken_is_populated

    def _get_smallest_cosine_similarity(self):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-15)
        cos_similarities = -cos(self.suspicious_embeddings,
                                self.avg_clean_embeddings)
        _, smallest_cosine_similarity = torch.topk(cos_similarities, 1000)
        return smallest_cosine_similarity

    @staticmethod
    def _remove_non_words(top_ids_to_tokens):
        result = []
        for top_id, token in top_ids_to_tokens.items():
            if re.match(r'[#]*([A-Za-z\-\d]+)[#]*', token) is not None:
                result.append(top_id)
        return result

    def _get_suffixes(self, dissimilar_token_ids):
        suffixes = []
        for i in dissimilar_token_ids:
            if '##' in self.tokenizer.convert_ids_to_tokens([i])[0]:
                suffixes.append(i)
        return suffixes

    def _get_prefixes(self, dissimilar_token_ids, suffixes):
        prefixes = [i for i in dissimilar_token_ids if i not in suffixes]
        return prefixes

    def _get_multitoken_words(self, prefixes, suffixes):
        '''
        extra ##ordi ##nari ##ly
        a = [1, 2, 3, 4, 5]
        permutation(a, 2)
        -> [[1, 2], [1, 3], ...]
        '''
        suffixes_combinations = []
        for i in range(1, 3):
            new_combination = list(permutations(suffixes, i))
            new_combination = [list(i) for i in new_combination]
            suffixes_combinations += new_combination

        candidates = []
        for p in prefixes:
            p_copy = deepcopy(p)
            candidates += [[p_copy]+i for i in suffixes_combinations]
        decoded_candidates = self.tokenizer.batch_decode(candidates)

        pool_obj = multiprocessing.Pool()

        ix_cand_tuple_list = []
        for ix, cand in enumerate(decoded_candidates):
            ix_cand_tuple_list.append((ix, cand))
        multitoken_words = pool_obj.map(check_word, ix_cand_tuple_list)
        return [candidates[i] for i in multitoken_words if i is not None]
