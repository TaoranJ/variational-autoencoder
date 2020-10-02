"""Fit a variational autoencoder to MNIST.

batch size is the innermost dimension, then the sample dimension, then latent
dimension
"""

import torch
import torch.utils
import numpy as np
import torch.utils.data


def training(train, valid, variational, model, optim, args):
    best_valid_elbo = -np.inf
    for epoch in range(1, args.epochs + 1):
        elbo_acc = []
        for batch in train:
            x = batch[0].to(args.device)
            model.zero_grad()
            variational.zero_grad()
            z, log_q_z = variational(x, n_samples=1)
            log_p_x_and_z = model(z, x)
            # average over sample dimension
            elbo = (log_p_x_and_z - log_q_z).mean(1).sum(0)
            # sum over batch dimension
            loss = -elbo
            loss.backward()
            optim.step()
            elbo_acc.append(elbo.detach().cpu())
        with torch.no_grad():
            valid_elbo, valid_log_p_x = evaluation(args.n_samples, model,
                                                   variational, valid)
        # Report
        print('[{:03d}/{:03d}]\tTR_ELBO: {:.2f}\tVA_ELBO: {:.2f}'.format(
            epoch, args.epochs, np.mean(elbo_acc), valid_elbo))
        # Save the best
        if valid_elbo > best_valid_elbo:
            best_valid_elbo = valid_elbo
            states = {'model': model.state_dict(),
                      'variational': variational.state_dict()}
            torch.save(states, 'checkpoint.pt')


def evaluation(n_samples, model, variational, eval_data):
    model.eval()
    total_log_p_x = 0.0
    total_elbo = 0.0
    for batch in eval_data:
        x = batch[0].to(next(model.parameters()).device)
        z, log_q_z = variational(x, n_samples)
        log_p_x_and_z = model(z, x)
        # importance sampling of approximate marginal likelihood with q(z)
        # as the proposal, and logsumexp in the sample dimension
        elbo = log_p_x_and_z - log_q_z
        log_p_x = torch.logsumexp(elbo, dim=1) - np.log(n_samples)
        # average over sample dimension, sum over minibatch
        total_elbo += elbo.cpu().numpy().mean(1).sum()
        # sum over minibatch
        total_log_p_x += log_p_x.cpu().numpy().sum()
    n_data = len(eval_data.dataset)
    return total_elbo / n_data, total_log_p_x / n_data
