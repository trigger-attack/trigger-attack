import torch
from random import randint, sample, choices


class TriggerInitializator():
    def __init__(self, tokenizer, device, embeddings_analyzer=None):
        self.tokenizer = tokenizer
        self.device = device
        # TODO: Initialize the most_changed_embeddings immediately
        self.embeddings_analyzer = embeddings_analyzer

    def make_initial_trigger(
            self,
            trigger_length: int,
            trigger_init_type: str = 'embedding_change'
            ) -> torch.Tensor:
        choices = {
            'random': self._make_random_trigger,
            'pad': self._make_pad_trigger,
            'embedding_change': self._make_most_changed_embedding_trigger
        }
        return choices[trigger_init_type](trigger_length)

    def _make_random_trigger(self, trigger_length):
        rand_list = []
        for _ in range(trigger_length):
            random_input_id = randint(0, len(self.tokenizer)-1)
            rand_list.append(random_input_id)
        return torch.tensor(rand_list).to(self.device)

    def _make_pad_trigger(self, trigger_length):
        pad_id = self.tokenizer.pad_token_id
        pad_tensor = torch.tensor([pad_id]*trigger_length)
        return pad_tensor.to(self.device)

    def _make_most_changed_embedding_trigger(self, trigger_length):
        self.embeddings_analyzer.populate_most_changed_embeddings()

        multitoken_options = self.embeddings_analyzer.multitoken_options
        random_multitoken = self.__sample_multitoken(multitoken_options)
        multitoken_length = randint(0, trigger_length)
        new_multitoken_trigger = self.__assemble_trigger(
                                    random_multitoken, multitoken_length)

        singletoken_options = self.embeddings_analyzer.singletoken_options
        num_tokens_to_fill = (trigger_length -
                              len(self._flatten(new_multitoken_trigger)))
        new_singletoken_trigger = choices(singletoken_options,
                                          k=num_tokens_to_fill)

        new_trigger = new_multitoken_trigger + new_singletoken_trigger
        new_trigger = self.__reshuffle_trigger(new_trigger)
        new_trigger = self.__unpack_multitoken_words_in_trigger(new_trigger)

        return new_trigger

    @staticmethod
    def _flatten(t):
        if len(t) > 0:
            return [item for sublist in t for item in sublist]
        else:
            return t

    @staticmethod
    def __sample_multitoken(multitoken_candidates):
        num_multitoken_words = len(multitoken_candidates)
        random_num_multitoken_words = randint(0, num_multitoken_words)
        return sample(multitoken_candidates, random_num_multitoken_words)

    @staticmethod
    def __assemble_trigger(random_multitoken, trigger_length):
        new_trigger_list = []
        for composed_word in random_multitoken:
            current_trigger_len = \
                len(TriggerInitializator._flatten(new_trigger_list))
            if len(composed_word) + current_trigger_len < trigger_length:
                new_trigger_list.append(composed_word)
            else:
                break
        return new_trigger_list

    @staticmethod
    def __reshuffle_trigger(trigger):
        new_arrangement = sample(range(len(trigger)), len(trigger))
        return [trigger[i] for i in new_arrangement]

    @staticmethod
    def __unpack_multitoken_words_in_trigger(trigger):
        unpacked_trigger = []
        for token in trigger:
            if isinstance(token, list):
                unpacked_trigger += token
            else:
                unpacked_trigger.append(token)
        return torch.stack(unpacked_trigger)
