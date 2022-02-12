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
        using the mean negative log likelihood that
        the suspicious model prediction of the trigger target is equal to 
        any of the trigger target labels
    '''
    target_prob = _get_probabilitites(all_logits['suspicious'].logits, batch['trigger_source_loc'].squeeze())
    trigger_target_tensor = torch.tensor(trigger_target_labels)
    stacked = trigger_target_tensor
    for _ in range(target_prob.shape[0]-1):
        stacked = torch.vstack((stacked,trigger_target_tensor))
    interested_class_prob = torch.gather(target_prob, -1, stacked.to(device=torch.device('cuda')))
    interested_class_prob = interested_class_prob.sum(dim=1)
    loss = -torch.log(interested_class_prob).mean()
    return loss

def calculate_ner_clean_loss(all_logits, batch, trigger_target_labels, clean_agg_fn):
    '''
    given a set of logits [negative, positive]
    compute the probability distribution over classes
        using the mean negative log likelihood that
        the clean model prediction of the trigger target is NOT equal to 
        any of the trigger target labels
    '''
    probabilities = []
    trigger_target_tensor = torch.tensor(trigger_target_labels)
    for clean_logits in all_logits['clean']:
        probabilities += [_get_probabilitites(clean_logits.logits, batch['trigger_source_loc'].squeeze())]
    probabilities = torch.stack(probabilities)
    clean_prob = clean_agg_fn(probabilities, dim=0)
    stacked = trigger_target_tensor
    for _ in range(clean_prob.shape[0]-1):
        stacked = torch.vstack((stacked,trigger_target_tensor))
    interested_class_prob = torch.gather(clean_prob, -1, stacked.to(device=torch.device('cuda')))
    interested_class_prob = interested_class_prob.sum(dim=1)
    interested_class_prob_baseline = torch.gather(batch['baseline_probabilities'], -1, stacked.to(device=torch.device('cuda')))
    interested_class_prob_baseline = interested_class_prob_baseline.sum(dim=1)
    clean_loss = -torch.log(1-torch.max(interested_class_prob - interested_class_prob_baseline, torch.zeros_like(interested_class_prob, device=torch.device('cuda')))).mean()
    return clean_loss

def _get_probabilitites(logits, trigger_source_loc):
    source_loc_logits = logits[torch.arange(len(logits)), trigger_source_loc]
    scores = torch.exp(source_loc_logits)
    probs = scores/torch.sum(scores, dim=1, keepdim=True)
    return probs
