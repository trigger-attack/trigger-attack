import torch
from copy import deepcopy
import warnings


class TriggerModels(torch.nn.Module):

    def __init__(self,
                 suspicious_model_filepath,
                 clean_model_filepaths,
                 num_train_clean_models=1,
                 device=torch.device('cpu')):
        super(TriggerModels, self).__init__()
        self.device = device
        self.suspicious_model = self._load_model(suspicious_model_filepath)

        self.clean_models = []
        for path in clean_model_filepaths[:num_train_clean_models]:
            model = self._load_model(path)
            self.clean_models.append(model)

        clean_test_models_filepaths = \
            clean_model_filepaths[:num_train_clean_models]
        remaining_clean_models = \
            len(clean_model_filepaths) - num_train_clean_models
        if remaining_clean_models < 1:
            warnings.warn(
                'There are not enough clean test models. '
                'Train models will be used as test models')
        else:
            clean_test_models_filepaths = \
                clean_model_filepaths[num_train_clean_models:]
        self.clean_models_test = []
        for path in clean_test_models_filepaths:
            model = self._load_model(path)
            model = model.cpu()
            self.clean_models_test.append(model)

        self.clean_grads = []
        self.suspicious_grads = []
        self._add_hooks_to_all_models()
        self._forward_input_list = self._get_forward_input_list()
        self.eval()

    def _load_model(self, suspicious_model_filepath):
        model = torch.load(suspicious_model_filepath, map_location=self.device)
        model.eval()
        return model

    def _add_hooks_to_all_models(self):
        self.__add_hooks_to_single_model(self.suspicious_model, is_clean=False)
        for clean_model in self.clean_models:
            self.__add_hooks_to_single_model(clean_model, is_clean=True)

    def __add_hooks_to_single_model(self, model, is_clean):
        module = self._find_word_embedding_module(model)
        module.weight.requires_grad = True
        if is_clean:
            def __extract_clean_grad_hook(module, grad_in, grad_out):
                self.clean_grads.append(grad_out[0])
            module.register_backward_hook(__extract_clean_grad_hook)
        else:
            def __extract_grad_hook(module, grad_in, grad_out):
                self.suspicious_grads.append(grad_out[0])
            module.register_backward_hook(__extract_grad_hook)

    def get_suspicious_model_embeddings(self):
        return self._get_embedding_weight(self.suspicious_model)

    def get_all_clean_models_embeddings(self):
        clean_embeddings = []
        for model in self.clean_models:
            clean_embeddings.append(self._get_embedding_weight(model))
        return clean_embeddings

    def get_all_clean_train_and_test_models_embeddings(self):
        clean_embeddings = self.get_all_clean_models_embeddings()
        for model in self.clean_models_test:
            clean_embeddings.append(self._get_embedding_weight(model))
        return clean_embeddings

    def _get_embedding_weight(self, model):
        word_embedding = self._find_word_embedding_module(model)
        word_embedding = deepcopy(word_embedding.weight)
        word_embedding = word_embedding.detach()
        word_embedding = word_embedding.to(torch.device('cpu'))
        word_embedding.requires_grad = False
        return word_embedding

    @staticmethod
    def _find_word_embedding_module(model):
        word_embedding_tuple = []
        for name, module in model.named_modules():
            if 'embeddings.word_embeddings' in name:
                word_embedding_tuple.append((name, module))
        assert len(word_embedding_tuple) == 1
        return word_embedding_tuple[0][1]

    def forward(self, batch, is_test=False):
        clean_model_list = self.clean_models
        if is_test:
            clean_model_list = self.clean_models_test
            clean_model_list = [model.to(self.device, non_blocking=True)
                                for model in clean_model_list]
        filtered_batch = self._filter_forward_batch(batch)
        two_dim_filtered_batch = self._make_two_d_if_necessary(filtered_batch)
        logits = {
            'suspicious': self.suspicious_model(**two_dim_filtered_batch),
            'clean': [clean_model(**two_dim_filtered_batch)
                      for clean_model in clean_model_list]
        }
        if is_test:
            clean_model_list = [model.cpu() for model in clean_model_list]
        return logits

    def _filter_forward_batch(self, batch):
        return {v: batch[v].to(self.device) for v in self._forward_input_list}

    def _make_two_d_if_necessary(self, batch):
        if batch['input_ids'].dim() == 1:
            new_dict = {}
            for k, v in batch.items():
                new_dict[k] = v.unsqueeze(0)
            return new_dict
        else:
            return batch

    def clear_model_gradients(self):
        self.suspicious_model.zero_grad()
        for clean_model in self.clean_models:
            clean_model.zero_grad()

    def clear_word_embedding_gradients(self):
        self.suspicious_grads = []
        self.clean_grads = []

    def _get_forward_input_list(self):
        forward_input_list = ['input_ids', 'attention_mask']
        if (('distilbert' not in self.suspicious_model.name_or_path) and
                ('bart' not in self.suspicious_model.name_or_path)):
            forward_input_list += ['token_type_ids']
        return forward_input_list
