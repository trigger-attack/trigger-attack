import torch



class QALoss(torch.nn.Module):
    
    def __init__(models, trigger_dataset):
        _expa

class QALoss

    @classmethod
    def preprocess_trigger_dataset(trigger_dataset):

    def forward(logits, batch):

    eval_trig_probs,  num_eval_triggered, eval_answer_prob =  _get_trigger_probs(batch, all_logits, loss_type='eval')
    if train_or_test == 'train':
        clean_trig_probs, num_clean_triggered, clean_answer_prob = _get_trigger_probs(batch, all_logits, loss_type='clean')
    else:
        clean_trig_probs_list, num_clean_triggered_list, answer_prob_list = [], [], []
        for i in range(len(all_logits['clean_start'])):
            clean_trig_probs, num_clean_triggered, clean_answer_prob = _get_trigger_probs(batch, all_logits, loss_type='clean', ix=i)
            clean_trig_probs_list.append(clean_trig_probs)
            num_clean_triggered_list.append(num_clean_triggered)
            answer_prob_list.append(clean_answer_prob)
        clean_trig_probs = torch.stack(clean_trig_probs_list).mean(0)
        num_clean_triggered = torch.stack(num_clean_triggered_list).float().mean(0)
        clean_answer_prob = torch.stack(answer_prob_list).float().mean(0)
    
    if populate_baselines:
        for loss_type, trigger_probs, answer_prob in [('clean', clean_trig_probs, clean_answer_prob), ('eval', eval_trig_probs, eval_answer_prob)]:
            batch[f'{train_or_test}_{loss_type}_baseline_likelihoods'] = trigger_probs.detach()
            batch[f'{train_or_test}_{loss_type}_answer_likelihoods'] = answer_prob.detach()
    
    
    m = len(batch['input_ids']) # scale the loss
    eval_loss  = m*(-torch.log( eval_trig_probs)).mean()
    clean_loss = m*(-torch.log(1 - torch.max(clean_trig_probs-batch[f'{train_or_test}_clean_baseline_likelihoods'], torch.zeros_like(clean_trig_probs, device=DEVICE)))).mean()
    
    trigger_inversion_loss = eval_loss + LAMBDA*clean_loss
    
    return {'eval_asr': num_eval_triggered,
            'clean_asr': num_clean_triggered,
            'eval_loss': eval_loss.detach(),
            'clean_loss': clean_loss.detach(),
            'trigger_inversion_loss': trigger_inversion_loss}


def _get_probabilities(all_logits):
    probabilities = {
        'suspicious':_get_probabilities(all_logits['suspicious']),
        'clean':[_get_probabilities(clean_logits) for clean_logits in all_logits['clean']]
    }
    return probabilities

def _get_probabilities(logits, batch):
    combined_logits = logits['start_logits'] + logits['end_logits']
    valid_logits = combined_logits - batch['valid_mask']*1e10
    scores = torch.exp(valid_logits)
    probs = scores/torch.sum(scores, dim=[1,2])

    best_ans_ixs = torch.arange(len(probs)), probs.view(len(probs), -1).argmax(dim=-1)
    num_triggered = batch['trigger_mask'].bool().view(len(probs), -1)[best_ans_ixs].sum()
    
    temperature = args.temperature
    if train_or_test == 'test':
        temperature = 1
    scores = torch.exp((logit_matrix)/temperature)
    probs = scores/torch.sum(scores, dim=[1,2]).view(-1,1,1).expand(-1, input_length, input_length)
    
    
    num_triggered = torch.zeros(1, device=DEVICE)
    if train_or_test == 'test' and populate_baselines == False:
        
    
    answer_prob = torch.zeros(1, device=DEVICE)
    if populate_baselines == True:
        answer_prob = torch.sum(probs*batch['answer_mask'].expand(probs.shape), dim=[-1,-2])
    
    if args.likelihood_agg == 'sum':
        input_trigger_probs = torch.sum(probs*batch['trigger_matrix_mask'].expand(probs.shape), dim=[-1,-2])
    elif args.likelihood_agg == 'max':
        input_trigger_probs = torch.amax(probs*batch['trigger_matrix_mask'].expand(probs.shape), dim=[-1,-2])
    else:
        return NotImplementedError

    return input_trigger_probs, num_triggered, answer_prob