import torch
from copy import deepcopy
import re
import multiprocessing
from itertools import permutations
import enchant


TOOL = enchant.Dict("en_US")
@torch.no_grad()
def check_word(tup):
    ix, cand = tup
    if TOOL.check(cand) == True:
        return ix

class TriggerInversionModels():

    def __init__(self, suspicious_model_filepath, clean_model_filepaths, tokenizer_filepath, device=torch.device('cpu')):
        self.suspicious_model = None
        self.clean_models = []
        self.tokenizer = None
        self.device = device
        self.most_changed_singletoken_words = None
        self.most_changed_multitoken_words = None

        self._load_suspicious_model_in_eval_mode(suspicious_model_filepath)
        self._load_all_clean_models_in_eval_mode(clean_model_filepaths)
        self._load_tokenizer(tokenizer_filepath)

    def _load_suspicious_model_in_eval_mode(self, suspicious_model_filepath):
        self.suspicious_model = self._load_pytorch_model_in_eval_mode(suspicious_model_filepath)

    def _load_all_clean_models_in_eval_mode(self, clean_model_filepaths):
        for path in clean_model_filepaths:
            model = self._load_pytorch_model_in_eval_mode(path)
            self.clean_models.append(model)

    def _load_tokenizer(self, tokenizer_filepath):
        self.tokenizer = torch.load(tokenizer_filepath, map_location=self.device)
    
    def _load_pytorch_model_in_eval_mode(self, suspicious_model_filepath):
        model = torch.load(suspicious_model_filepath, map_location=self.device)
        model.eval()
        return model


    @torch.no_grad()
    def populate_most_changed_embeddings(self):
        self._populate_all_input_id_embeddings()

        _, dissimilar_token_ids = self._get_smallest_cosine_similarity_values_and_token_ids()

        dissimilar_token_ids = dissimilar_token_ids.to(self.device)
        top_ids_to_tokens = {top_id:self.tokenizer.convert_ids_to_tokens([top_id])[0] for top_id in dissimilar_token_ids}            
        dissimilar_token_ids = self._remove_non_words(top_ids_to_tokens)

        prefixes, suffixes = self._get_prefixes_and_suffixes(dissimilar_token_ids)
        multitoken_words = self._get_multitoken_words(prefixes, suffixes)

        self.most_changed_singletoken_words = prefixes[:8]
        self.most_changed_multitoken_words = multitoken_words
    
    @torch.no_grad()
    def _populate_all_input_id_embeddings(self):
        self._suspicious_embeddings = self._get_embedding_weight(self.suspicious_model)
        self._all_clean_embeddings = [self._get_embedding_weight(model) for model in self.clean_models]
        self._avg_clean_embeddings = torch.stack(self._all_clean_embeddings).mean(dim=0)
    
    def _get_embedding_weight(self, model):
        word_embedding = self._find_word_embedding_module(model)
        word_embedding = deepcopy(word_embedding.weight).detach().to(torch.device('cpu'))
        word_embedding.requires_grad = False
        return word_embedding

    @staticmethod
    def _find_word_embedding_module(model):
        word_embedding_tuple = [(name, module) for name, module in model.named_modules() 
                                                if 'embeddings.word_embeddings' in name]
        assert len(word_embedding_tuple) == 1
        return word_embedding_tuple[0][1]

    def _get_smallest_cosine_similarity_values_and_token_ids(self):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-15)
        cos_similarities = [-cos(self._suspicious_embeddings, embeddings) for embeddings in self._all_clean_embeddings]
        cos_similarities += [-cos(self._suspicious_embeddings, self._avg_clean_embeddings)]
        min_cos_sim = torch.stack(cos_similarities).min(dim=0)[0]
        return torch.topk(min_cos_sim, 1000)

    @staticmethod
    def _remove_non_words(top_ids_to_tokens):
        result = []
        for top_id, token in top_ids_to_tokens.items():
            if re.match(r'[#]*\w([A-Za-z]+)[#]*', token) is not None:
                result.append(top_id)
        return result

    def _get_prefixes_and_suffixes(self, dissimilar_token_ids):
        all_suffixes = [i for i in dissimilar_token_ids if '##' in self.tokenizer.convert_ids_to_tokens([i])[0]]
        suffixes = all_suffixes[:5]
        prefixes = [i for i in dissimilar_token_ids if i not in all_suffixes]
        return prefixes, suffixes

    def _get_multitoken_words(self, prefixes, suffixes):
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

        multitoken_words = pool_obj.map(check_word, [(ix, cand) for ix, cand in enumerate(decoded_candidates)])
        return [candidates[i] for i in multitoken_words if i is not None]


    def __forward__(self, triggered_dataset):
        return NotImplementedError
        # suspicious_model

        # add_logits('eval', models['eval'][0](**{v:batch[v] for v in var_list}))
        # for clean_model in models[f'clean_{train_or_test}']:
        #     add_logits('clean', clean_model(**{v:batch[v] for v in var_list}))