import torch
from random import randint, sample, choices


def make_random_new_trigger(triggered_dataset, _):
    rand_list = [randint(0,len(triggered_dataset.tokenizer)-1) for _ in range(triggered_dataset.trigger_length)]
    return torch.tensor(rand_list).to(triggered_dataset.device)

def make_pad_trigger(triggered_dataset, _):
    pad_id = triggered_dataset.tokenizer.pad_token_id
    pad_tensor = torch.tensor([pad_id]*triggered_dataset.trigger_length)
    return pad_tensor.to(triggered_dataset.device)

def pick_random_permutation_of_most_changed_embeds(triggered_dataset, models):
    assert models.most_changed_singletoken_words is not None,\
        "most changed single token words cannot be None. the trigger_inversion_models did not initialize correctly"
                  
    random_multitoken_words =  _get_random_multitoken_words(models.most_changed_multitoken_words)
    new_multitoken_trigger = _make_trigger_from_multitoken_words(random_multitoken_words, triggered_dataset.trigger_length)
    
    num_tokens_to_fill = triggered_dataset.trigger_length - len(new_multitoken_trigger)
    new_singletoken_trigger = choices(models.most_changed_singletoken_words, k=num_tokens_to_fill)

    new_trigger = new_multitoken_trigger + new_singletoken_trigger
    new_trigger = _reshuffle_trigger(new_trigger)
    new_trigger = _unpack_multitoken_words_in_trigger(new_trigger)

    return new_trigger

    

def _get_random_multitoken_words(multitoken_words):
    num_multitoken_words = len(multitoken_words)
    random_num_multitoken_words = randint(0, num_multitoken_words)
    return sample(multitoken_words, random_num_multitoken_words)

def _make_trigger_from_multitoken_words(random_multitoken_words, trigger_length):
    new_trigger_list = []
    for composed_word in random_multitoken_words:
        if len(composed_word)+len(new_trigger_list) < trigger_length:
            new_trigger_list.append(composed_word)
        else:
            break
    return new_trigger_list

def _reshuffle_trigger(trigger):
    new_arrangement = sample(range(len(trigger)), len(trigger))
    return [trigger[i] for i in new_arrangement]

def _unpack_multitoken_words_in_trigger(trigger):
    unpacked_trigger = []
    for token in trigger:
        if isinstance(token, list):
            unpacked_trigger += token
        else:
            unpacked_trigger.append(token)
    return torch.stack(unpacked_trigger)

trigger_init_names_to_fn = {
    'embed_ch': pick_random_permutation_of_most_changed_embeds,
    'random': make_random_new_trigger, 
    'pad': make_pad_trigger}
