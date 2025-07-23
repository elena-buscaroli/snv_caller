import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.autograd.functional import hessian
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import defaultdict


def log_binomial_coefficient(D, Y):
    """
    Log binomial coefficient
    """
    return torch.lgamma(D + 1) - torch.lgamma(Y + 1) - torch.lgamma(D - Y + 1)


def log_likelihood_gamma(gamma, y_v, d_v, Z, k):
    """
    Log likelihood -> log p(y_v^k | gamma)
    """
    probs = torch.sigmoid(gamma)  # compute theta -> prob of success
    log_coeff = log_binomial_coefficient(d_v, y_v)
    cell_idxs = torch.where(Z == k)[0]  # idxs of cells with cellid == k
    return torch.sum((log_coeff + y_v * torch.log(probs) + (d_v - y_v) * torch.log(1 - probs))[cell_idxs])


def log_prior_gamma(gamma, mu_k, Sigma_inv):
    """
    Log prior -> log p(gamma | mu_k, Sigma_k)
    """
    diff = gamma - mu_k
    # return -0.5 * diff.T @ Sigma_inv @ diff -> diff.T gives a warning
    return -0.5 * diff.permute(*torch.arange(diff.ndim - 1, -1, -1)) @ Sigma_inv @ diff


def negative_joint(gamma, y_vk, d_vk, mu_k, Sigma_inv, Z, k):
    """
    Computes the negative log joint -> -log p(y | gamma) - log p(gamma | mu, Sigma) \\
    -> negative joint = - p(y | gamma) * p(gamma | mu, Sigma)
    """
    return -( log_likelihood_gamma(gamma, y_vk, d_vk, Z, k) + log_prior_gamma(gamma, mu_k, Sigma_inv) )


def optimize_gamma_hat(y_v, d_v, mu_k, Sigma_inv, Z, k):
    """
    Computes the mode of gamma_vk
    """
    gamma = nn.Parameter(torch.randn_like(mu_k))
    # inner_opt = optim.LBFGS([gamma], max_iter=10) # try first order optimizer
    inner_opt = optim.Adam([gamma], lr=1e-2) # try first order optimizer

    def gamma_hat():
        inner_opt.zero_grad()
        loss = negative_joint(gamma, y_v, d_v, mu_k, Sigma_inv, Z, k)
        loss.backward()
        return loss

    inner_opt.step(gamma_hat)
    return gamma.detach()


def custom_hessian(y, x, v):
    grad_y = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    hvp = torch.autograd.grad(grad_y, x, grad_outputs=v, retain_graph=True)[0]
    return hvp


def compute_laplace_term(y_v, d_v, mu_k, Sigma_inv, Z, k):
    """
    Computes the Laplace approximation of the marginal
    """
    time_tmp = time.time()
    # laplace approximation term for one v,k
    gamma_hat = optimize_gamma_hat(y_v, d_v, mu_k, Sigma_inv, Z, k)
    # print(f"Time required to optimize gamma = {time.time() - time_tmp}")
    
    # hessian of negative log joint at gamma_hat
    def loss_fn(gamma):
        return negative_joint(gamma, y_v, d_v, mu_k, Sigma_inv, Z, k)

    # print(f"gamma_hat shape = {gamma_hat.shape}")
    
    time_tmp = time.time()
    H = hessian(loss_fn, gamma_hat, vectorize=True)
    # print(f"Time required to compute the Hessian matrix = {time.time() - time_tmp}")
    
    time_tmp = time.time()
    H_det_log = torch.logdet(H + 1e-6 * torch.eye(H.shape[0], device=H.device))
    # print(f"Time required to compute the log determinant of H = {time.time() - time_tmp}")

    time_tmp = time.time()
    ll = log_likelihood_gamma(gamma_hat, y_v, d_v, Z, k)
    # print(f"Time required to compute the loglikelihood = {time.time() - time_tmp}")
    
    time_tmp = time.time()
    lp = log_prior_gamma(gamma_hat, mu_k, Sigma_inv)
    # print(f"Time required to compute the log prior = {time.time() - time_tmp}\n")

    # print(f"Devices\ngamma_hat = {gamma_hat.device}\nH = {H.device}\nH_det_log = {H_det_log.device}\nll = {ll.device}\nlp = {lp.device}")

    N = Sigma_inv.shape[0]
    return ll + lp - 0.5 * H_det_log + N/2 * torch.log(torch.tensor(2*torch.pi)), gamma_hat

def compute_linkage(gene_expression):
    """
    Computes the complete linkage of the input matrix of gene expression
    """
    standardized_columns = []
    
    for i in range(gene_expression.shape[1]):
        col = gene_expression.getcol(i).toarray().flatten()  # convert column to dense
        mean = col.mean()
        std = col.std()
        if std == 0:
            std = 1  # avoid division by zero
        standardized_col = (col - mean) / std
        standardized_columns.append(standardized_col)
    
    X_std = np.column_stack(standardized_columns)  # shape: genes x cells
    correlation_matrix = np.corrcoef(X_std.T)  # shape: cells x cells
    distance_matrix = 1 - correlation_matrix
    condensed_dist = squareform(distance_matrix, checks=False)
    return linkage(condensed_dist, method="complete")

def compute_ou_kernel(edges, clone_labels, lambd=1.0, sigma_squared=1.0):
    """
    OU kernel: cov_ij = sigma^2 * exp( -lambda (d_i + d_j - 2 * d_mrca) )

    Params
    - edges -> list of (parent, child, branch_length)
    - clone_labels -> list of leaf node names
    - lambd -> positive decay rate - larger value means higher local corr (going to diagonal cov)
    - sigma_squared -> trait variance at equilibrium
    """
    tree = defaultdict(list)  # keys: parents, values: (children, length)
    parent_of = {}
    for parent, child, length in edges:
        tree[parent].append((child, length))
        parent_of[child] = parent

    # identify root
    all_nodes = set(parent_of.keys()).union(tree.keys())
    root = list(all_nodes - set(parent_of.keys()))[0]

    # compute depth from root to each node
    node_depth = {}
    def dfs(node, depth):
        node_depth[node] = depth
        # iterate through the list of childred of "node"
        # update the distance from root to "node"
        for child, length in tree.get(node, []):
            dfs(child, depth + length)
    dfs(root, 0.0)

    # get paths from root to "node"
    def get_path_to_root(node):
        path = []
        while node in node_depth:
            path.append(node)
            if node == root:
                break
            node = parent_of.get(node, root)
        return path[::-1] # order from root to "node"

    ancestor_paths = {node: get_path_to_root(node) for node in clone_labels}

    def find_mrca(i, j):
        path_i = ancestor_paths[i]
        path_j = ancestor_paths[j]
        mrca = root
        for a, b in zip(path_i, path_j):
            if a == b:
                mrca = a
            else:
                break
        return mrca

    # kernel matrix
    n = len(clone_labels)
    kernel = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            node_i = clone_labels[i]
            node_j = clone_labels[j]
            d_i = node_depth[node_i]
            d_j = node_depth[node_j]
            mrca = find_mrca(node_i, node_j)
            d_mrca = node_depth[mrca]
            dist = d_i + d_j - 2 * d_mrca
            kernel[i, j] = sigma_squared * np.exp(-lambd * dist)

    return kernel