import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd.functional import hessian

def log_binomial_coefficient(D, Y):
    '''
    Log binomial coefficient
    '''
    return torch.lgamma(D + 1) - torch.lgamma(Y + 1) - torch.lgamma(D - Y + 1)


def log_likelihood_gamma(gamma, y_v, d_v, Z, k):
    '''
    Log likelihood -> log p(y_v^k | gamma)
    '''
    probs = torch.sigmoid(gamma)  # compute theta -> prob of success
    log_coeff = log_binomial_coefficient(d_v, y_v)
    cell_idxs = torch.where(Z == k)[0]  # idxs of cells with cellid == k
    return torch.sum((log_coeff + y_v * torch.log(probs) + (d_v - y_v) * torch.log(1 - probs))[cell_idxs])


def log_prior_gamma(gamma, mu_k, Sigma_inv):
    '''
    Log prior -> log p(gamma | mu_k, Sigma_k)
    '''
    diff = gamma - mu_k
    # return -0.5 * diff.T @ Sigma_inv @ diff -> diff.T gives a warning
    return -0.5 * diff.permute(*torch.arange(diff.ndim - 1, -1, -1)) @ Sigma_inv @ diff


def negative_joint(gamma, y_vk, d_vk, mu_k, Sigma_inv, Z, k):
    '''
    Computes the negative log joint -> -log p(y | gamma) - log p(gamma | mu, Sigma) \\
    -> negative joint = - p(y | gamma) * p(gamma | mu, Sigma)
    '''
    return -( log_likelihood_gamma(gamma, y_vk, d_vk, Z, k) + log_prior_gamma(gamma, mu_k, Sigma_inv) )


def optimize_gamma_hat(y_v, d_v, mu_k, Sigma_inv, Z, k):
    '''
    Computes the mode of gamma_vk
    '''
    gamma = nn.Parameter(torch.randn_like(mu_k))
    inner_opt = optim.LBFGS([gamma], max_iter=10) # try first order optimizer

    def gamma_hat():
        inner_opt.zero_grad()
        loss = negative_joint(gamma, y_v, d_v, mu_k, Sigma_inv, Z, k)
        loss.backward()
        return loss

    inner_opt.step(gamma_hat)
    return gamma.detach()


def compute_laplace_term(y_v, d_v, mu_k, Sigma_inv, Z, k):
    '''
    Computes the Laplace approximation of the marginal
    '''
    # laplace approximation term for one v,k
    gamma_hat = optimize_gamma_hat(y_v, d_v, mu_k, Sigma_inv, Z, k)

    # hessian of negative log joint at gamma_hat
    def loss_fn(gamma):
        return negative_joint(gamma, y_v, d_v, mu_k, Sigma_inv, Z, k)

    H = hessian(loss_fn, gamma_hat)
    H_det_log = torch.logdet(H + 1e-6 * torch.eye(H.shape[0], device=H.device))

    ll = log_likelihood_gamma(gamma_hat, y_v, d_v, Z, k)
    lp = log_prior_gamma(gamma_hat, mu_k, Sigma_inv)

    # print(f"Devices\ngamma_hat = {gamma_hat.device}\nH = {H.device}\nH_det_log = {H_det_log.device}\nll = {ll.device}\nlp = {lp.device}")

    N = Sigma_inv.shape[0]
    return ll + lp - 0.5 * H_det_log + N/2 * torch.log(torch.tensor(2*torch.pi)), gamma_hat