import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd.functional import hessian
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import defaultdict


def set_ggplot_light_theme(base_size=10):
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": base_size,

        "axes.titlesize": base_size + 2,
        "axes.titleweight": "bold",
        "axes.labelsize": base_size,
        "axes.labelweight": "normal",
        "xtick.labelsize": base_size - 2,
        "ytick.labelsize": base_size - 2,
        "legend.fontsize": base_size - 2,
        "legend.title_fontsize": base_size,

        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "grid.color": "#E5E5E5",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,

        "axes.grid": True,
        "axes.axisbelow": True,

        "legend.loc": "lower center",
        "legend.frameon": False,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 1.5,
        "patch.edgecolor": "none",

        "savefig.format" : "pdf"
    })


def plot_heatmap(table,
                 col_colors=None, row_colors=None,
                 linkage_matrix_col=None, linkage_matrix_row=None,
                 legend_title=None):
    pl = sns.clustermap(table, 
                        col_colors=col_colors,
                        row_colors=row_colors,
                        col_linkage=linkage_matrix_col, 
                        row_linkage=linkage_matrix_row,
                        row_cluster=True,
                        cmap="Blues", figsize=(8, 6))

    pl.ax_heatmap.set_xticks([])
    pl.ax_heatmap.set_yticks([])
    pl.fig.subplots_adjust(right=0.75)
    pl.ax_cbar.set_position((0.8, .2, .03, .4))
    pl.ax_cbar.set_title(legend_title, pad=10)
    return pl


def log_binomial_coefficient(D, Y):
    """
    Log binomial coefficient
    """
    return torch.lgamma(D + 1) - torch.lgamma(Y + 1) - torch.lgamma(D - Y + 1)


def log_likelihood_gamma(gamma, y_v, d_v):
    """
    Log likelihood -> log p(y_v^k | gamma)
    """
    probs_v = torch.sigmoid(gamma)  # compute theta -> prob of success
    log_coeff = log_binomial_coefficient(d_v, y_v)

    assert y_v.shape == d_v.shape
    assert log_coeff.shape == y_v.shape

    return torch.sum(log_coeff + y_v * torch.log(probs_v) + (d_v - y_v) * torch.log(1 - probs_v))


def log_prior_gamma(gamma, mu, Sigma_inv):
    """
    Log prior -> log p(gamma | mu, Sigma)
    """
    diff = gamma.unsqueeze(-1) - mu.unsqueeze(-1)

    assert diff.shape == (gamma.shape[0], 1)

    # return -0.5 * diff.T @ Sigma_inv @ diff -> diff.T gives a warning
    return -0.5 * diff.permute(*torch.arange(diff.ndim - 1, -1, -1)) @ Sigma_inv @ diff


def negative_joint(gamma, y_vk, d_vk, mu, Sigma_inv):
    """
    Computes the negative log joint -> -log p(y | gamma) - log p(gamma | mu, Sigma) \\
    -> negative joint = - p(y | gamma) * p(gamma | mu, Sigma)
    """
    return -( log_likelihood_gamma(gamma, y_vk, d_vk) + log_prior_gamma(gamma, mu, Sigma_inv) )


def optimize_gamma_hat(y_v, d_v, mu, Sigma_inv, gamma, n_steps=5, lr=1e-2):
    """
    Computes the mode of gamma_vk
    """
    gamma_par = nn.Parameter(gamma.clone().requires_grad_(True))
    optimizer = optim.Adam([gamma_par], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = negative_joint(gamma_par, y_v, d_v, mu, Sigma_inv)
        loss.backward()
        optimizer.step()

    return gamma_par.detach()


def is_indefinite(H):
    eigvals = torch.linalg.eigvalsh(H)  # For symmetric matrices
    has_pos = torch.any(eigvals > 0)
    has_neg = torch.any(eigvals < 0)
    return has_pos and has_neg


def compute_laplace_term(y_v, d_v, mu, Sigma_inv, gamma):
    """
    Computes the Laplace approximation of the marginal
    """
    # laplace approximation term for one v
    gamma_hat = optimize_gamma_hat(y_v, d_v, mu, Sigma_inv, gamma)

    # hessian of negative log joint at gamma_hat
    def loss_fn(gamma):
        return negative_joint(gamma, y_v, d_v, mu, Sigma_inv)

    H = hessian(loss_fn, gamma_hat, vectorize=True)
    H_det_log = torch.logdet(H + 1e-6 * torch.eye(H.shape[0], device=H.device))

    ll = log_likelihood_gamma(gamma_hat, y_v, d_v)
    lp = log_prior_gamma(gamma_hat, mu, Sigma_inv)

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


def logit_clipped(x, eps=1e-7):
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.logit(x)
