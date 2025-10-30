import dendropy
import pandas as pd
import sys
import random

from dendropy import Taxon
from dendropy.simulate import treesim
from scipy.special import logit, expit
from tqdm import tqdm
from aux_fns import *


def model_run(AD, DP, kernel, device="cpu", use_kernel=True, learning_rate = 0.005):
    import torch
    import time
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    from torch.autograd.functional import hessian

    device = torch.device(device)
    # convert to tensor the data
    Y = torch.tensor(AD.values).to(device=device)
    D = torch.tensor(DP.values).to(device=device)

    V, N = Y.shape  # #mutations x #cells

    assert Y.shape[1] == D.shape[1]
    print(f"{V} SNPs, {N} cells\n")

    print(f"Devices\nY = {Y.device}\nD = {D.device}")

    # Initialize mu to the observed VAF
    mu_init = logit_clipped((Y/D).nan_to_num(0))
    mu = mu_init.detach().clone().requires_grad_(True)

    # Initialize gamma
    gamma_hat = mu_init.detach().clone().requires_grad_(False)

    # Initialize the covariance matrix
    if use_kernel:
        Sigma = torch.tensor(kernel, dtype=torch.float32, device=device) + 1e-6*torch.eye(N, device=device)
    else:
        Sigma = torch.eye(n=kernel.shape[0], device=device)
    Sigma_inv = torch.linalg.inv(Sigma)  # inverese of Sigma -> N x N
    log_det_Sigma = torch.logdet(Sigma)  # log of determinant of Sigma

    # Check the device of all parameters
    print(f"Devices\nmu = {mu.device}\nSigma = {Sigma.device}\nSigma_inv = {Sigma_inv.device}\nlog_det_Sigma = {log_det_Sigma.device}\ngamma_hat = {gamma_hat.device}")

    # Optimizer
    optimizer = optim.Adam([mu], lr=learning_rate)

    # Store initial time and lists
    start_time = time.time()
    losses, gradients = [], []

    # Start the inference
    pbar = tqdm(range(50))
    for _ in pbar:

        optimizer.zero_grad()  # reset the gradient
        total_laplace = 0.0
        for v in range(V):
            # compute log p(y_v | d_v, mu_v, Sigma) with laplace approx
            y_v = Y[v, :]
            d_v = D[v, :]
            laplace_term, gamma_hat_v = compute_laplace_term(y_v, d_v, mu[v,:], Sigma_inv, gamma_hat[v,:])
            
            total_laplace += laplace_term
            gamma_hat[v,:] = gamma_hat_v

        loss = -total_laplace
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        gradients.append(mu.grad.norm().item())

        pbar.set_description(f"Loss = {loss.item()}")

    # Store final time
    end_time = time.time()

    return {"losses":losses,
            "gradients":gradients,
            "mu_init":mu_init.cpu().detach().numpy(),
            "gamma_hat":gamma_hat.cpu().detach().numpy(),
            "mu":mu.cpu().detach().numpy()}





if __name__ == "__main__":
    data_folder = sys.argv[1] + "/"
    # suffix = sys.argv[2]
    suffix = ""

    out_folder = data_folder + "out/"


    V, N, K = 1000, 500, 2

    tree = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=N, rng=random.Random(10))
    assignments, linkage_matrix, distances_df = assign_clones_from_tree(tree, K)  # dict: {cell_id:clone_id, ...}
    distances_df.to_csv(out_folder+"distances.csv")
    
    cell_ids = list(assignments.keys())
    clone_ids = list(assignments.values())

    unique_groups = set(clone_ids)
    color_list = ["#5F9EA0","#FF4500","#DAA520","#7B68EE"]
    color_map = {group: color_list[i % len(color_list)] for i, group in enumerate(unique_groups)}
    label_colors = [color_map[group] for group in clone_ids]

    plot_dendogram(linkage_matrix, cell_ids, clone_ids, color_map,
                   out_name=out_folder+"dendogram.pdf")

    edges = tree_to_edge_list(tree)
    kernel = compute_ou_kernel(edges, list(assignments.keys()), lambd=.01, sigma_squared=1.0)
    pd.DataFrame(kernel).to_csv(out_folder+"kernel.csv")
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] == N
    plot_heatmap(kernel, col_colors=label_colors, row_colors=label_colors,
                 linkage_matrix_col=linkage_matrix, linkage_matrix_row=linkage_matrix,
                 out_name=out_folder+"kernel.pdf")


    
    rng = np.random.default_rng(0)
    # 2 clones, 1000 mutations, 500 cells
    
    pi_common = 0.1
    p_dropout = 0.5
    mean_depth = 10
    dispersion = 10
    
    n_trunk = int(pi_common * V)
    n_private = (V - n_trunk) // 2
    
    classes = (["trunk"] * n_trunk + ["clone1_private"] * n_private + ["clone2_private"] * (V - n_trunk - n_private))
    rng.shuffle(classes)

    r = dispersion
    p = r / (r + mean_depth)
    depths = rng.negative_binomial(r, p, size=(V, N))
    drop_mask = rng.random((V, N)) < p_dropout
    depths[drop_mask] = 0
    
    u = np.unique(clone_ids)
    assert u.size == 2, "Attesi esattamente due cloni in clone_ids"
    
    mask_c1 = (clone_ids == u[0])   # trattiamo u[0] come 'clone1'
    mask_c2 = (clone_ids == u[1])   # trattiamo u[1] come 'clone2'
    
    classes = np.asarray(classes)
    
    G = np.zeros((V, N), dtype=np.int8)
    G[classes == "trunk", :]           = 1
    G[classes == "clone1_private", :]  = mask_c1  # broadcast sul numero di righe selezionate
    G[classes == "clone2_private", :]  = mask_c2
    
    alt_counts = np.zeros_like(depths)
    mask_mut = (G == 1) & (depths > 0)
    alt_counts[mask_mut] = rng.binomial(depths[mask_mut], 0.5)

    DP_true = mean_depth * np.ones_like(depths)  # o una costante teorica
    AD_true = 0.5 * DP_true * G                  # se la mutazione è presente: 0.5*DP, altrimenti 0
    presence = (G == 1).astype(int)              # 1 = dovrebbe esserci, 0 = assente
    
    DP = pd.DataFrame(depths)
    AD = pd.DataFrame(alt_counts)
    # P_true = 0.5 * (G == 1)
    # AD_true = DP * P_true
    theta_t = (AD/DP).fillna(0)
    theta_t[theta_t==0] = 1e-15
    theta_t[theta_t==1] = 1. - 1e-15
    
    gamma_t = mu_t = logit(theta_t)

    data_true = pd.concat([array_to_df(theta_t, "VAF", cell_ids),
                           array_to_df(AD, "AD", cell_ids)["AD"],
                           array_to_df(DP, "DP", cell_ids)["DP"],
                           array_to_df(AD_true, "AD_true", cell_ids)["AD_true"],
                           array_to_df(DP_true, "DP_true", cell_ids)["DP_true"],
                           array_to_df(presence, "presence", cell_ids)["presence"],
                           array_to_df(gamma_t, "gamma", cell_ids)["gamma"]],
                           axis=1, join="inner")
    data_true["clone_ids"] = [str(assignments[i]) for i in data_true["cell_ids"]]
    data_true.to_csv(out_folder+"data_true.csv")




    res1 = model_run(AD, DP, kernel, device="cuda", use_kernel=True)
    res2 = model_run(AD, DP, kernel, device="cuda", use_kernel=False)

    pd.DataFrame(res1["losses"], columns=["losses"]).to_csv(out_folder+"losses_kernel.csv")
    pd.DataFrame(res2["losses"], columns=["losses"]).to_csv(out_folder+"losses_Nkernel.csv")

    pd.DataFrame(res1["gradients"], columns=["grads"]).to_csv(out_folder+"grads_kernel.csv")
    pd.DataFrame(res2["gradients"], columns=["grads"]).to_csv(out_folder+"grads_Nkernel.csv")

    pd.DataFrame(res1["mu_init"]).to_csv(out_folder+"mu_init_kernel.csv")
    pd.DataFrame(res2["mu_init"]).to_csv(out_folder+"mu_init_Nkernel.csv")

    pd.DataFrame(res1["gamma_hat"]).to_csv(out_folder+"gamma_hat_kernel.csv")
    pd.DataFrame(res2["gamma_hat"]).to_csv(out_folder+"gamma_hat_Nkernel.csv")

    pd.DataFrame(res1["mu"]).to_csv(out_folder+"mu_kernel.csv")
    pd.DataFrame(res2["mu"]).to_csv(out_folder+"mu_Nkernel.csv")



