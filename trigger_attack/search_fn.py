def gradient_based_discrete_search(triggered_dataset, models, loss_fn):
    loss = loss_fn(triggered_dataset, models)
    