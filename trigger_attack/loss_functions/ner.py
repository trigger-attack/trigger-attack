import torch


def calculate_ner_loss(all_logits, batch, trigger_target_labels:list, clean_agg_fn=torch.mean):
    '''
    given a dictionary of logits: {
        'suspicious':[negative, positive],
        'clean':[[negative, positive], ...]
    }
        a trigger label
        a true label
        and baseline probabilities

    return suspicious loss + clean loss
    '''
    suspicious_loss = calculate_ner_suspicious_loss(all_logits, trigger_target_labels, batch)
    clean_loss = calculate_ner_clean_loss(all_logits, batch, trigger_target_labels, clean_agg_fn)
    return suspicious_loss + clean_loss

def calculate_ner_suspicious_loss(all_logits, trigger_target_labels, batch):
    '''
    given a set of logits [negative, positive]
    compute the probability distribution over classes 
        using the mean log likelihood that the answer 
        is the trigger target label
    '''
    '''
    YOUR CODE HERE
    '''
    return NotImplementedError

def calculate_ner_clean_loss(all_logits, batch, trigger_target_labels, clean_agg_fn):
    '''
    given a set of logits [negative, positive]
    compute the probability distribution over classes
        as the mean negative log likelihood the the
        final prediction is *not* in the trigger target labels
    '''
    '''
    YOUR CODE HERE
    '''
    return NotImplementedError

def _get_probabilitites(logits, trigger_source_loc):
    source_loc_logits = logits[torch.arange(len(logits)), trigger_source_loc]
    scores = torch.exp(source_loc_logits)
    probs = scores/torch.sum(scores, dim=1, keepdim=True)
    return probs