from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler

import torch
import numpy as np
import itertools
from collections import OrderedDict

class _RepeatSampler(object):
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class HensmanDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    Dataloader when using minibatching with Stochastic Variational Inference.

    """
    def __init__(self, dataset, batch_sampler, num_workers):
        super().__init__(dataset, batch_sampler=_RepeatSampler(batch_sampler), num_workers=num_workers)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class SubjectSampler(Sampler):
    """
    Perform individual-wise sampling
    
    """
    def __init__(self, data_source, P, T):
        super(SubjectSampler, self).__init__(data_source)
        self.data_source = data_source
        self.P = P
        self.T = T

    def __iter__(self):
        r = np.arange(self.P)
        np.random.shuffle(r)
        list_of_lists = list(map(lambda x: [i for i in range(self.T*x, self.T*(x+1))], r))
        res = list(itertools.chain.from_iterable(list_of_lists))
        return iter(res)

    def __len__(self):
        return len(self.data_source)

class VaryingLengthSubjectSampler(Sampler):
    """
    Perform individual-wise sampling when individuals have varying number of temporal samples.
    
    """
    def __init__(self, data_source, id_covariate):
        super(VaryingLengthSubjectSampler, self).__init__(data_source)
        self.data_source = data_source
        self.id_covariate = id_covariate

        def f(x):
            return int(x['label'][id_covariate].item())

        l = list(map(f, data_source))
        self.P = len(set(l))
        self.start_indices = [l.index(x) for x in list(OrderedDict.fromkeys(l))]
        self.end_indices = self.start_indices[1:] + [len(data_source)]

    def __iter__(self):
        r = np.arange(self.P) 
        np.random.shuffle(r)
        list_of_lists = list(map(lambda x: [(i, x) for i in range(self.start_indices[x], self.end_indices[x])], r))
        res = iter(itertools.chain.from_iterable(list_of_lists))
        return iter(res)

    def __len__(self):
        return self.P

class VaryingLengthBatchSampler(BatchSampler):
    """
    Perform batch sampling when individuals have varying number of temporal samples.
    
    """
    def __init__(self, sampler, batch_size):
        super(VaryingLengthBatchSampler, self).__init__(sampler, batch_size, False)
        assert isinstance(sampler, VaryingLengthSubjectSampler)
        self.sampler = sampler
        self.batch_size = batch_size

    #__len__ defined by the superclass

    def __iter__(self):
        batch = []
        batch_subjects = set()
        for idx, subj in self.sampler:
            if subj not in batch_subjects:
                if len(batch_subjects) == self.batch_size:
                    yield batch
                    batch = []
                    batch_subjects.clear()
                batch_subjects.add(subj)
            batch.append(idx)
        yield batch

def batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, 
                            test_x, mu, zt_list, id_covariate, eps):
    """
    Perform batch predictions when individuals have varying number of temporal samples.
    
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = prediction_x.shape[1]
    M = zt_list[0].shape[0]

    I_M = torch.eye(M, dtype=torch.double).to(device) 

    if isinstance(covar_module0, list):
        K0xz = torch.zeros(latent_dim, prediction_x.shape[0], M).double().to(device)
        K0zz = torch.zeros(latent_dim, M, M).double().to(device)
        K0Xz = torch.zeros(latent_dim, test_x.shape[0], M).double().to(device)

        for i in range(latent_dim):
            covar_module0[i].eval()
            covar_module1[i].eval()
            likelihoods[i].eval()
            z = zt_list[i].to(device)

            K0xz[i] = covar_module0[i](prediction_x, z).evaluate()
            K0zz[i] = covar_module0[i](z, z).evaluate()
            K0Xz[i] = covar_module0[i](test_x, z).evaluate()

    else:
        covar_module0.eval()
        covar_module1.eval()
        likelihoods.eval()

        K0xz = covar_module0(prediction_x, zt_list).evaluate()
        K0zz = covar_module0(zt_list, zt_list).evaluate()
        K0Xz = covar_module0(test_x, zt_list).evaluate()

    K0zz = K0zz + eps * I_M
    K0zx = K0xz.transpose(-1, -2)

    iB_st_list = []
    H = K0zz
    subjects = torch.unique(prediction_x[:, id_covariate]).tolist()
    iB_mu = torch.zeros(latent_dim, prediction_x.shape[0], 1, dtype=torch.double).to(device)
    for s in subjects:
        indices = prediction_x[:, id_covariate] == s
        x_st = prediction_x[indices]
        T = x_st.shape[0]
        I_T = torch.eye(T, dtype=torch.double).to(device)

        if isinstance(covar_module0, list):
            B_st = torch.zeros(latent_dim, T, T, dtype=torch.double).to(device)
            for i in range(latent_dim):
                B_st[i] = covar_module1[i](x_st, x_st).evaluate() + I_T * likelihoods[i].noise_covar.noise
        else:
            stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=0)
            B_st = covar_module1(stacked_x_st, stacked_x_st).evaluate() + I_T * likelihoods.noise_covar.noise.unsqueeze(dim=2)

        LB_st = torch.cholesky(B_st)
        iB_st = torch.cholesky_solve(I_T, LB_st)
        K0xz_st = K0xz[:, indices]
        K0zx_st = K0xz_st.transpose(-1, -2)
        iB_K0xz = torch.matmul(iB_st, K0xz_st)
        K0zx_iB_K0xz = torch.matmul(K0zx_st, iB_K0xz)
        H = H + K0zx_iB_K0xz
        iB_mu[:, indices] = torch.matmul(iB_st, mu[indices].T.unsqueeze(dim=2))
        iB_st_list.append(iB_st)

    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.solve(torch.matmul(K0zx, iB_mu), H)[0])
    iB_K0xz_iH_K0zx_iB_mu = torch.zeros(latent_dim, prediction_x.shape[0], 1, dtype=torch.double).to(device)
    for i, s in enumerate(subjects):
        indices = prediction_x[:, id_covariate] == s
        iB_K0xz_iH_K0zx_iB_mu[:, indices] = torch.matmul(iB_st_list[i], K0xz_iH_K0zx_iB_mu_st[:, indices])
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu

    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.solve(torch.matmul(K0zx, mu_tilde), K0zz)[0])

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(prediction_x[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(latent_dim, test_x.shape[0], 1, dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s

        if isinstance(covar_module0, list):
            K1Xx = torch.zeros(latent_dim, test_x[indices].shape[0], np.sum(mask)).double().to(device)
            for i in range(latent_dim):
                K1Xx[i] = covar_module1[i](test_x[indices], prediction_x[mask]).evaluate()
        else:
            stacked_test_x_indices = torch.stack([test_x[indices] for i in range(latent_dim)], dim=0)
            stacked_prediction_x_mask = torch.stack([prediction_x[mask] for i in range(latent_dim)], dim=0)
            K1Xx = covar_module1(stacked_test_x_indices, stacked_prediction_x_mask).evaluate()
        K1Xx_mu_tilde[:, indices] = torch.matmul(K1Xx, mu_tilde[:, mask])
    
    Z_pred = (K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde).squeeze(dim=2).T

    return Z_pred
 
def batch_predict(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, mu, 
                 zt_list, P, T, id_covariate, eps):
    """
    Perform batch-wise predictions
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = prediction_x.shape[1]
    M = zt_list[0].shape[0]
    I_M = torch.eye(M, dtype=torch.double).to(device)
    I_T = torch.eye(T, dtype=torch.double).to(device)

    x_st = torch.reshape(prediction_x, [P, T, Q])

    mu = mu.T
    mu_st = torch.reshape(mu, [latent_dim, P, T, 1])

    if isinstance(covar_module0, list):
        K0xz = torch.zeros(latent_dim, P*T, M).double().to(device)
        K0zz = torch.zeros(latent_dim, M, M).double().to(device)
        B_st = torch.zeros(latent_dim, P, T, T).double().to(device)
        K0Xz = torch.zeros(latent_dim, test_x.shape[0], M).double().to(device)

        for i in range(latent_dim):
            covar_module0[i].eval()
            covar_module1[i].eval()
            likelihoods[i].eval()
            z = zt_list[i].to(device)

            K0xz[i] = covar_module0[i](prediction_x, z).evaluate()
            K0zz[i] = covar_module0[i](z, z).evaluate()
            B_st[i] = covar_module1[i](x_st, x_st).evaluate() + I_T * likelihoods[i].noise_covar.noise
            K0Xz[i] = covar_module0[i](test_x, z).evaluate()

    else:
        covar_module0.eval()
        covar_module1.eval()
        likelihoods.eval()

        stacked_x_st = torch.stack([x_st for i in range(latent_dim)], dim=1)

        K0xz = covar_module0(prediction_x, zt_list).evaluate()
        K0zz = covar_module0(zt_list, zt_list).evaluate()
        B_st = (covar_module1(stacked_x_st, stacked_x_st).evaluate() + I_T * likelihoods.noise_covar.noise.unsqueeze(dim=2)).transpose(0, 1)
        K0Xz = covar_module0(test_x, zt_list).evaluate()

    K0zz = K0zz + eps * I_M
    LB_st = torch.cholesky(B_st)
    iB_st = torch.cholesky_solve(I_T, LB_st)
    K0xz_st = torch.reshape(K0xz, [latent_dim, P, T, M])
    K0zx_st = K0xz_st.transpose(-1, -2)
    K0zx = K0xz.transpose(-1, -2)

    iB_K0xz = torch.matmul(iB_st, K0xz_st)
    K0zx_iB_K0xz = torch.matmul(K0zx, torch.reshape(iB_K0xz, [latent_dim, P*T, M]))
    H = K0zz + K0zx_iB_K0xz
    iB_mu = torch.matmul(iB_st, mu_st).view(latent_dim, -1, 1)
    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.solve(torch.matmul(K0zx, iB_mu), H)[0]).reshape(latent_dim, P, T, -1)
    iB_K0xz_iH_K0zx_iB_mu = torch.matmul(iB_st, K0xz_iH_K0zx_iB_mu_st).view(latent_dim, -1, 1)
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu
    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.solve(torch.matmul(K0zx, mu_tilde), K0zz)[0])

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(prediction_x[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(latent_dim, test_x.shape[0], 1, dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s

        if isinstance(covar_module0, list):
            K1Xx = torch.zeros(latent_dim, test_x[indices].shape[0], np.sum(mask)).double().to(device)
            for i in range(latent_dim):
                K1Xx[i] = covar_module1[i](test_x[indices], prediction_x[mask]).evaluate()
        else:
            stacked_test_x_indices = torch.stack([test_x[indices] for i in range(latent_dim)], dim=0)
            stacked_prediction_x_mask = torch.stack([prediction_x[mask] for i in range(latent_dim)], dim=0)
            K1Xx = covar_module1(stacked_test_x_indices, stacked_prediction_x_mask).evaluate()

        K1Xx_mu_tilde[:, indices] = torch.matmul(K1Xx, mu_tilde[:, mask])
    
    Z_pred = (K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde).squeeze(dim=2).T

    return Z_pred

def predict(covar_module0, covar_module1, likelihood, train_xt, test_x, mu, z, P, T, id_covariate, eps):
    """
    Helper function to perform predictions.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = train_xt.shape[1]
    M = z.shape[0]
    I_M = torch.eye(M, dtype=torch.double).to(device)
    I_T = torch.eye(T, dtype=torch.double).to(device)

    x_st = torch.reshape(train_xt, [P, T, Q])
    mu_st = torch.reshape(mu, [P, T, 1])

    K0xz = covar_module0(train_xt, z).evaluate()
    K0zz = covar_module0(z, z).evaluate() + eps * I_M
    K1_st = covar_module1(x_st, x_st).evaluate()
    K0Xz = covar_module0(test_x, z).evaluate()

    B_st = K1_st + I_T * likelihood.noise_covar.noise
    LB_st = torch.cholesky(B_st)
    iB_st = torch.cholesky_solve(I_T, LB_st)
    K0xz_st = torch.reshape(K0xz, [P, T, M])
    K0zx_st = K0xz_st.transpose(-1, -2)

    iB_K0xz = torch.matmul(iB_st, K0xz_st)
    K0zx_iB_K0xz = torch.matmul(K0xz.T, torch.reshape(iB_K0xz, [P*T, M]))
    H = K0zz + K0zx_iB_K0xz

    iB_mu = torch.matmul(iB_st, mu_st).view(-1)
    K0xz_iH_K0zx_iB_mu_st = torch.matmul(K0xz, torch.solve(torch.matmul(K0xz.T, iB_mu).unsqueeze(dim=1), H)[0]).reshape(P, T, -1)
    iB_K0xz_iH_K0zx_iB_mu = torch.matmul(iB_st, K0xz_iH_K0zx_iB_mu_st).view(-1)
    mu_tilde = iB_mu - iB_K0xz_iH_K0zx_iB_mu
    K0Xz_iK0zz_K0zx_mu_tilde = torch.matmul(K0Xz, torch.solve(torch.matmul(K0xz.T, mu_tilde).unsqueeze(dim=1), K0zz)[0]).squeeze()

    test_subjects = torch.unique(test_x[:, id_covariate]).cpu().numpy()
    mask = np.isin(train_xt[:, id_covariate].cpu().numpy(), test_subjects)

    K1Xx_mu_tilde = torch.zeros(test_x.shape[0], dtype=torch.double).to(device)
    for s in test_subjects:
        indices = test_x[:, id_covariate] == s
        K1Xx = covar_module1(test_x[indices], train_xt[mask]).evaluate()
        K1Xx_mu_tilde[indices] = torch.matmul(K1Xx, mu_tilde[mask])

    Z_pred = K0Xz_iK0zz_K0zx_mu_tilde + K1Xx_mu_tilde

    return Z_pred

