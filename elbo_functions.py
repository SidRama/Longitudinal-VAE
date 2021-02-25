import torch
import numpy as np

"""
Various approaches to compute the GP term in the loss
"""

def KL_closed(covar_module, train_x, likelihoods, data, mu, log_var):
    """
    Closed form KL divergence.

    :param covar_module: additive kernel (sum of cross-covariances)
    :param train_x: auxiliary covariate information
    :param likelihoods: GPyTorch likelihood model
    :param data: sample measurements
    :param mu: mean of approximating variational distribution
    :param log_var: log variance of approximating variational distribution
    :return: KL divergence between variational distribution and additive GP prior
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K1 = covar_module(train_x.to(device), train_x.to(device)).evaluate() \
         + likelihoods.noise * torch.eye(data.shape[0]).to(device)
    v1 = torch.exp(log_var.view(-1)).to(device)
    mu1 = mu.view(-1)
    LK1 = torch.cholesky(K1)
    iK1 = torch.cholesky_solve(torch.eye(data.shape[0],
                                         dtype=torch.double).to(device), LK1).to(device)
    logdet11 = 2 * torch.sum(torch.log(torch.diag(LK1))).to(device)
    qf1 = torch.sum(mu1 * torch.matmul(iK1, mu1)).to(device)
    tr1 = torch.sum(v1 * torch.diag(iK1)).to(device)
    logdet10 = log_var.sum().to(device)
    kld1 = 0.5 * (tr1 + qf1 - data.shape[0] + logdet11 - logdet10)
    return kld1

def elbo(covar_module0, covar_module1, likelihood, train_xt, train_yt, z, P, T, eps):
    """
    Efficient KL divergence. See L-VAE paper.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param train_xt: auxiliary covariate information
    :param train_yt: latent embedding of samples
    :param z: inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: KL divergence between variational distribution and additive GP prior
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.double
    train_xt_st = torch.reshape(train_xt, [P, T, train_xt.shape[1]]).to(device)
    train_yt_st = torch.reshape(train_yt, [P, T, 1]).to(device)
    K0xz = covar_module0(train_xt, z).evaluate().to(device)
    K0zz = (covar_module0(z, z).evaluate() + eps * torch.eye(z.shape[0], dtype=torch_dtype).to(device)).to(device)
    LK0zz = torch.cholesky(K0zz).to(device)
    iK0zz = torch.cholesky_solve(torch.eye(z.shape[0], dtype=torch_dtype).to(device), LK0zz).to(device)
    K0_st = covar_module0(train_xt_st, train_xt_st).evaluate().to(device)
    K1_st = covar_module1(train_xt_st, train_xt_st).evaluate().to(device)
    B_st = K1_st + torch.eye(T, dtype=torch_dtype).to(device).to(device) * likelihood.noise_covar.noise.to(device)
    LB_st = torch.cholesky(B_st).to(device)
    iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch_dtype).to(device) , LB_st)
    K0xz_st = torch.reshape(K0xz, [P, T, K0xz.shape[1]]).to(device)
    iB_K0xz = torch.matmul(iB_st, K0xz_st).to(device)
    K0zx_iB_K0xz = torch.matmul(torch.transpose(K0xz, 0, 1), torch.reshape(iB_K0xz, [P * T, K0xz.shape[1]])).to(device)
    W = K0zz + K0zx_iB_K0xz
    W = (W + W.T) / 2
    LW = torch.cholesky(W).to(device)
    logDetK0zz = 2 * torch.sum(torch.log(torch.diagonal(LK0zz))).to(device)
    logDetB = 2 * torch.sum(torch.log(torch.diagonal(LB_st, dim1=-2, dim2=-1))).to(device)
    logDetW = 2 * torch.sum(torch.log(torch.diagonal(LW))).to(device)
    logDet = -logDetK0zz + logDetB + logDetW
    iB_y_st = torch.solve(train_yt_st, B_st)[0].to(device)
    qF1 = torch.sum(train_yt_st*iB_y_st).to(device)
    p = torch.matmul(K0xz.T, torch.reshape(iB_y_st, [P * T])).to(device)
    qF2 = torch.sum(torch.triangular_solve(p[:,None], LW, upper=False)[0] ** 2).to(device)
    qF = qF1 - qF2
    tr = torch.sum(iB_st * K0_st) - torch.sum(K0zx_iB_K0xz * iK0zz)
    constTerm = -0.5 * T * P * np.log(2 * np.pi)
    logLike = constTerm + -0.5 * (logDet + qF)
    el = (logLike - 0.5 * tr).to(device)
    return el

def deviance_upper_bound(covar_module0, covar_module1, likelihood, train_xt, m, log_v, z, P, T, eps):
    """
    Efficient KL divergence using the variational mean and variance instead of a sample from the latent space (DUBO).
    See L-VAE supplementary material.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param train_xt: auxiliary covariate information
    :param m: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P: number of unique instances
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: KL divergence between variational distribution and additive GP prior (DUBO)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.exp(log_v)
    torch_dtype = torch.double
    x_st = torch.reshape(train_xt, [P, T, train_xt.shape[1]]).to(device)
    m_st = torch.reshape(m, [P, T, 1]).to(device)
    v_st = torch.reshape(v, [P, T]).to(device)
    K0xz = covar_module0(train_xt, z).evaluate().to(device)
    K0zz = (covar_module0(z, z).evaluate() + eps * torch.eye(z.shape[0], dtype=torch_dtype).to(device)).to(device)
    LK0zz = torch.cholesky(K0zz).to(device)
    iK0zz = torch.cholesky_solve(torch.eye(z.shape[0], dtype=torch_dtype).to(device), LK0zz).to(device)
    K0_st = covar_module0(x_st, x_st).evaluate().to(device)
    K1_st = covar_module1(x_st, x_st).evaluate().to(device)
    B_st = K1_st + torch.eye(T, dtype=torch_dtype).to(device).to(device) * likelihood.noise_covar.noise.to(device)
    LB_st = torch.cholesky(B_st).to(device)
    iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch_dtype).to(device), LB_st)
    K0xz_st = torch.reshape(K0xz, [P, T, K0xz.shape[1]]).to(device)
    iB_K0xz = torch.matmul(iB_st, K0xz_st).to(device)
    K0zx_iB_K0xz = torch.matmul(torch.transpose(K0xz, 0, 1), torch.reshape(iB_K0xz, [P*T, K0xz.shape[1]])).to(device)
    W = K0zz + K0zx_iB_K0xz
    W = (W + W.T) / 2
    LW = torch.cholesky(W).to(device)
    logDetK0zz = 2 * torch.sum(torch.log(torch.diagonal(LK0zz))).to(device)
    logDetB = 2 * torch.sum(torch.log(torch.diagonal(LB_st, dim1=-2, dim2=-1))).to(device)
    logDetW = 2 * torch.sum(torch.log(torch.diagonal(LW))).to(device)
    logDetSigma = -logDetK0zz + logDetB + logDetW
    iB_m_st = torch.solve(m_st, B_st)[0].to(device)
    qF1 = torch.sum(m_st*iB_m_st).to(device)
    p = torch.matmul(K0xz.T, torch.reshape(iB_m_st, [P * T])).to(device)
    qF2 = torch.sum(torch.triangular_solve(p[:,None], LW, upper=False)[0] ** 2).to(device)
    qF = qF1 - qF2
    tr = torch.sum(iB_st * K0_st) - torch.sum(K0zx_iB_K0xz * iK0zz)
    logDetD = torch.sum(torch.log(v)).to(device)
    tr_iB_D = torch.sum(torch.diagonal(iB_st, dim1=-2, dim2=-1)*v_st).to(device)
    D05_iB_K0xz = torch.reshape(iB_K0xz*torch.sqrt(v_st)[:,:,None], [P*T, K0xz.shape[1]])
    K0zx_iB_D_iB_K0zx = torch.matmul(torch.transpose(D05_iB_K0xz,0,1), D05_iB_K0xz).to(device)
    tr_iB_K0xz_iW_K0zx_iB_D = torch.sum(torch.diagonal(torch.cholesky_solve(K0zx_iB_D_iB_K0zx, LW))).to(device)
    tr_iSigma_D = tr_iB_D - tr_iB_K0xz_iW_K0zx_iB_D
    dubo = 0.5*(tr_iSigma_D + qF - P*T + logDetSigma - logDetD + tr)
    return dubo

def minibatch_KLD_upper_bound(covar_module0, covar_module1, likelihood, latent_dim, m, H, train_xt, mu, log_v, z, P_tot, P_batch, T, natural_gradient, eps):
    """
    Efficient unbiased estimate of the KL-divergence upper bound that enables the use of mini-batching.
    See L-VAE supplementary material.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param m: mean of inducing values u
    :param H: covariate matrix of inducing values u
    :param train_xt: auxiliary covariate information
    :param mu: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P_tot: number of unique instances in the dataset
    :param P_batch: number of unique instances in the mini-batch
    :param T: number of longitudinal samples per individual
    :param eps: jitter
    :return: Unbiased estimate of the KL-divergence upper bound and its gradients w.r.t. m and H
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M = H.shape[-1] 
    x_st = torch.reshape(train_xt, [P_batch, T, train_xt.shape[1]])
    stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)

    K0xz = covar_module0(train_xt, z).evaluate()
    K0zz = covar_module0(z, z).evaluate()
    K0_st = covar_module0(stacked_x_st, stacked_x_st).evaluate().transpose(0, 1)
    B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + torch.eye(T, dtype=torch.double).to(device) * likelihood.noise_covar.noise.unsqueeze(dim=2)).transpose(0,1)

    K0zz = K0zz + eps * torch.eye(M, dtype=torch.double).to(device)
    LK0zz = torch.cholesky(K0zz)
    iK0zz = torch.cholesky_solve(torch.eye(M, dtype=torch.double).to(device), LK0zz)
    LB_st = torch.cholesky(B_st)
    iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch.double).to(device), LB_st).squeeze(dim=0)

    K0xz_st = torch.reshape(K0xz, [latent_dim, P_batch, T, M])
    iB_K0xz = torch.matmul(iB_st, K0xz_st)
    K0zx_iB_K0xz = torch.matmul(torch.transpose(K0xz, 1, 2), torch.reshape(iB_K0xz, [latent_dim, P_batch*T, M]))
    LH = torch.cholesky(H)
    iH = torch.cholesky_solve(torch.eye(M, dtype=torch.double).to(device), LH)

    #Compute the batch-wise partial sum
    _ = (torch.matmul(torch.matmul(K0xz, iK0zz), m).squeeze() - mu.T).reshape(latent_dim, P_batch, T, -1)
    A = torch.matmul(torch.matmul(_.transpose(2,3), iB_st), _).sum()
    B = torch.sum(torch.diagonal(iB_st, dim1=-1, dim2=-2).reshape(latent_dim, -1)*torch.exp(log_v.T))
    C = 2 * torch.sum(torch.log(torch.diagonal(LB_st, dim1=-2, dim2=-1)))
    D = torch.sum(iB_st * K0_st) - torch.sum(K0zx_iB_K0xz * iK0zz)
    _ = torch.matmul(torch.matmul(iK0zz, H), iK0zz)
    E = torch.sum(_.transpose(-1, -2) * K0zx_iB_K0xz)
    F = torch.sum(log_v)

    #Compute kld_qu_pu
    tr1 = torch.sum(iK0zz * H.transpose(-1, -2))
    qf1 = torch.sum(m * torch.matmul(iK0zz, m))
    logdetK = 2 * torch.sum(torch.log(torch.diagonal(LK0zz, dim1=-1, dim2=-2)))
    logdetH = 2 * torch.sum(torch.log(torch.diagonal(LH, dim1=-1, dim2=-2)))
    kld_qu_pu = 0.5 * (tr1 + qf1 - latent_dim*M + logdetK - logdetH)
    kld_total = P_tot/P_batch*0.5*(A + B + C + D + E - F) + kld_qu_pu - latent_dim*P_tot*T/2

    #Compute gradients of m and H w.r.t. kld_total
    grad_m, grad_H = None, None
    if natural_gradient:
        mu = mu.transpose(0,1).reshape(latent_dim, P_batch, T, -1)
        K0zx = K0xz.reshape(latent_dim, P_batch, T, -1).transpose(-1, -2)
        A = torch.matmul(torch.matmul(iK0zz.unsqueeze(dim=1), K0zx), torch.matmul(iB_st, mu)).sum(dim=1)
        B = torch.matmul(torch.matmul(iK0zz, K0zx_iB_K0xz), iK0zz) + iK0zz
        grad_m = -A + torch.matmul(B, m)
        grad_H = 0.5*(-iH + B)

    return kld_total, grad_m, grad_H


def minibatch_KLD_upper_bound_iter(covar_module0, covar_module1, likelihood, latent_dim, m, H, train_xt, mu, log_v, z, P, P_in_current_batch, N, natural_gradient, id_covariate, eps):
    """
    Efficient unbiased estimate of the KL-divergence upper bound that enables the use of mini-batching.
    See L-VAE supplementary material.

    :param covar_module0: additive kernel (sum of cross-covariances) without id covariate
    :param covar_module1: additive kernel (sum of cross-covariances) with id covariate
    :param likelihood: GPyTorch likelihood model
    :param m: mean of inducing values u
    :param H: covariate matrix of inducing values u
    :param train_xt: auxiliary covariate information
    :param mu: variational mean
    :param log_v: (log) variational variance
    :param z: inducing points
    :param P: total number of subjects
    :param P_in_current_batch: number of subjects in the batch
    :param N: total number of samples
    :param eps: jitter
    :return: Unbiased estimate of the KL-divergence upper bound and its gradients w.r.t. m and H
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M = H.shape[-1]

    K0xz = covar_module0(train_xt, z).evaluate()
    K0zz = covar_module0(z, z).evaluate()
    K0zz = K0zz + eps * torch.eye(M, dtype=torch.double).to(device)
    LK0zz = torch.cholesky(K0zz)
    iK0zz = torch.cholesky_solve(torch.eye(M, dtype=torch.double).to(device), LK0zz)
    LH = torch.cholesky(H)
    iH = torch.cholesky_solve(torch.eye(M, dtype=torch.double).to(device), LH)

    A_part = (torch.matmul(torch.matmul(K0xz, iK0zz), m).squeeze() - mu.T).unsqueeze(dim=2)
    E_part = torch.matmul(torch.matmul(iK0zz, H), iK0zz)

    A = torch.tensor([0.0], dtype=torch.double).to(device)
    B = torch.tensor([0.0], dtype=torch.double).to(device)
    C = torch.tensor([0.0], dtype=torch.double).to(device)
    D = torch.tensor([0.0], dtype=torch.double).to(device)
    E = torch.tensor([0.0], dtype=torch.double).to(device)
    if natural_gradient:
        ng_P1 = torch.zeros(latent_dim, M, 1, dtype=torch.double).to(device)
        ng_P2 = torch.zeros(latent_dim, M, M, dtype=torch.double).to(device)

    subjects = torch.unique(train_xt[:, id_covariate]).tolist()
    for s in subjects:
        indices = train_xt[:, id_covariate] == s
        tx = train_xt[indices]
        T = tx.shape[0] 
        stacked_tx = torch.stack([tx for i in range(latent_dim)], dim=0)
        K0_st = covar_module0(stacked_tx, stacked_tx).evaluate()
        B_st = covar_module1(stacked_tx, stacked_tx).evaluate() + torch.eye(T, dtype=torch.double).to(device) * likelihood.noise_covar.noise.unsqueeze(dim=2)

        LB_st = torch.cholesky(B_st)
        iB_st = torch.cholesky_solve(torch.eye(T, dtype=torch.double).to(device), LB_st).squeeze(dim=0)
        K0xz_st = K0xz[:, indices]
        K0zx_iB_K0xz = torch.einsum('bik,bij,bjl->bkl', K0xz_st, iB_st, K0xz_st)

        A = A + torch.einsum('bji,bjk,bkl->b', A_part[:, indices], iB_st, A_part[:, indices]).sum()
        B = B + torch.sum(torch.diagonal(iB_st, dim1=-1, dim2=-2).reshape(latent_dim, -1)*torch.exp(log_v[indices].T))
        C = C + 2 * torch.sum(torch.log(torch.diagonal(LB_st, dim1=-2, dim2=-1)))
        D = D + torch.sum(iB_st * K0_st) - torch.sum(K0zx_iB_K0xz * iK0zz)
        E = E + torch.sum(E_part * K0zx_iB_K0xz)

        if natural_gradient:
            mu_p = mu[indices].transpose(-1,-2).unsqueeze(dim=2)
            K0zx = K0xz_st.transpose(-1,-2)
            ng_P1 = ng_P1 + torch.matmul(K0zx, torch.matmul(iB_st, mu_p))
            ng_P2 = ng_P2 + K0zx_iB_K0xz
 
    F = torch.sum(log_v)

    #Compute kld_qu_pu
    tr1 = torch.sum(iK0zz * H.transpose(-1, -2))
    qf1 = torch.sum(m * torch.matmul(iK0zz, m))
    logdetK = 2 * torch.sum(torch.log(torch.diagonal(LK0zz, dim1=-1, dim2=-2)))
    logdetH = 2 * torch.sum(torch.log(torch.diagonal(LH, dim1=-1, dim2=-2)))
    kld_qu_pu = 0.5 * (tr1 + qf1 - latent_dim*M + logdetK - logdetH)

    kld_total = P/P_in_current_batch*0.5*(A + B + C + D + E - F) + kld_qu_pu - latent_dim*N/2

    grad_m, grad_H = None, None
    if natural_gradient:
        B = torch.matmul(iK0zz, torch.matmul(ng_P2, iK0zz)) + iK0zz
        grad_m = -torch.matmul(iK0zz, ng_P1) + torch.matmul(B, m)
        grad_H = 0.5*(-iH + B)

    return kld_total, grad_m, grad_H
