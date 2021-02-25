import math

import torch
from torch import nn
from torch.nn import functional as F

class Likelihoods(nn.Module):
    """
    Likelihood definition

    """
    def __init__(self, latent_dim, noise, constrain=True):
        super(Likelihoods, self).__init__()
        self.latent_dim = latent_dim

        min_log_noise = torch.Tensor([-16.0])
        log_noise_init = torch.log(noise - torch.exp(min_log_noise))
        self._log_noise = nn.Parameter(torch.Tensor([log_noise_init] * latent_dim), requires_grad=constrain)

        self.register_buffer('min_log_noise', min_log_noise * torch.ones(1))

    @property
    def noise(self):
        return torch.exp(self.min_log_noise + F.softplus(self._log_noise - self.min_log_noise))

    @noise.setter
    def noise(self, noise):
        with torch.no_grad():
            self._log_noise.copy_(torch.log(noise - torch.exp(self.min_log_noise)))

class BinKernel(nn.Module):
    """
    Specification of binary kernel

    """
    def __init__(self, dim):
        super(BinKernel, self).__init__()
        self.dim = dim   
 
    def forward(self, x1, x2):
        return (x1[..., self.dim].unsqueeze(-1) + x2[..., self.dim].unsqueeze(-2) == 2).type(torch.double)

class CatKernel(nn.Module):
    """
    Specification of categorical kernel

    """
    def __init__(self, dim):
        super(CatKernel, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return (x1[..., self.dim].unsqueeze(-1) - x2[..., self.dim].unsqueeze(-2) == 0).type(torch.double)

class RbfKernel(nn.Module):
    """
    Specification of radial basis function kernel

    """
    def __init__(self, dim, latent_dim=1, lengthscale=2.5):
        super(RbfKernel, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        min_log_lengthscale = torch.Tensor([-16.0])
        log_lengthscale_init = torch.log(lengthscale - torch.exp(min_log_lengthscale))
        self._log_lengthscale = nn.Parameter(torch.Tensor([log_lengthscale_init] * latent_dim), requires_grad=True)

        self.register_buffer('min_log_lengthscale', min_log_lengthscale * torch.ones(1))

    @property
    def lengthscale(self):
        return torch.exp(self.min_log_lengthscale + F.softplus(self._log_lengthscale - self.min_log_lengthscale))

    @lengthscale.setter
    def lengthscale(self, lengthscale):
        with torch.no_grad():
            self._log_lengthscale.copy_(torch.log(lengthscale - torch.exp(self.min_log_lengthscale)))

    def forward(self, x1, x2):
        l = max(len(x1.shape), len(x2.shape))
        s = self.lengthscale
        while len(s.shape) < l:
            s = s.unsqueeze(dim=-1)
        return torch.exp(-1*((x1[..., self.dim].unsqueeze(-1) - x2[..., self.dim].unsqueeze(-2))**2)/(2*s**2).type(torch.double))

class ScaleKernel(nn.Module):
    """
    Specification of scale kernel

    """
    def __init__(self, kernel, latent_dim=1, scale=math.log(2)):
        super(ScaleKernel, self).__init__()
        self.latent_dim = latent_dim
        self.kernel = kernel

        min_log_scale = torch.Tensor([-16.0])
        log_scale_init = torch.log(scale - torch.exp(min_log_scale))
        self._log_scale = nn.Parameter(torch.Tensor([log_scale_init] * latent_dim), requires_grad=True)

        self.register_buffer('min_log_scale', min_log_scale * torch.ones(1))

    @property
    def scale(self):
        return torch.exp(self.min_log_scale + F.softplus(self._log_scale - self.min_log_scale))

    @scale.setter
    def scale(self, scale):
        with torch.no_grad():
            self._log_scale.copy_(torch.log(scale - torch.exp(self.min_log_scale)))

    def forward(self, x1, x2):
        l = max(len(x1.shape), len(x2.shape))
        s = self.scale
        while len(s.shape) < l:
            s = s.unsqueeze(dim=-1)
        return s * self.kernel(x1, x2)
       
class AdditiveKernel(nn.Module):
    """
    Specification of additive kernel

    """
    def __init__(self, kernels):
        super(AdditiveKernel, self).__init__()
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x1, x2):
        def f(k):
            return k(x1, x2)
        return sum(list(map(f, self.kernels)))

class ProductKernel(nn.Module):
    """
    Specification of product kernel

    """
    def __init__(self, kernel1, kernel2):
        super(ProductKernel, self).__init__()
        self.k1 = kernel1
        self.k2 = kernel2

    def forward(self, x1, x2):
        return self.k1(x1, x2) * self.k2(x1, x2)

def generate_kernel_batched(latent_dim, cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel, covariate_missing_val, id_covariate):
    """
    Generate two additive kernels. One with id covariate and the other without the id covariate

    :param latent_dim:  number of latent dimensions
    :param cat_kernel:  list of indices from the covariate matrix for the categorical kernel
    :param bin_kernel:  list of indices from the covariate matrix for the binary kernel
    :param sqexp_kernel:  list of indices from the covariate matrix for the squared exponential kernel
    :param cat_int_kernel:  list of dictionaries with indices for the interaction kernel between a categorical and continuous covariate
                            E.g.: [{'cont_covariate':0, 'cat_covariate':2}, {'cont_covariate':0, 'cat_covariate':3}]
    :param bin_int_kernel:  list of dictionaries with indices for the interaction kernel between a binary and continuous covariate
                            E.g.: [{'cont_covariate':1, 'bin_covariate':4}]
    :param covariate_missing_val:  list of dictionaries with indices from the covariate matrix with missing values
                            and their corresponding masks
                            E.g.: [{'covariate':0, 'mask': 3}]
    :param id_covariate:  index from the covariate matrix for the id covariate
    :return: AdditiveKernel without id covariate and AdditiveKernel with id covariate
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    covariate_missing = [dict_instance['covariate'] for dict_instance in covariate_missing_val]
    kernel0 = nn.ModuleList()
    kernel1 = nn.ModuleList()

    # categorical kernels
    for idx in cat_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            if idx == id_covariate:
                kernel1.append(ScaleKernel(ProductKernel(CatKernel(idx), BinKernel(dict_instance['mask'])), latent_dim).to(device))
            else:
                kernel0.append(ScaleKernel(ProductKernel(CatKernel(idx), BinKernel(dict_instance['mask'])), latent_dim).to(device))
        else:
            if idx == id_covariate:
                kernel1.append(ScaleKernel(CatKernel(idx), latent_dim).to(device))
            else:
                kernel0.append(ScaleKernel(CatKernel(idx), latent_dim).to(device))

    # continuous kernels
    for idx in sqexp_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            kernel0.append(ScaleKernel(ProductKernel(RbfKernel(idx, latent_dim), BinKernel(dict_instance['mask'])), latent_dim).to(device))
        else:
            kernel0.append(ScaleKernel(RbfKernel(idx, latent_dim), latent_dim).to(device))

    # binary kernels
    for idx in bin_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            kernel0.append(ScaleKernel(ProductKernel(BinKernel(idx), BinKernel(dict_instance['mask'])), latent_dim).to(device))
        else:
            kernel0.append(ScaleKernel(BinKernel(idx), latent_dim).to(device))

    # interaction kernels (categorical)
    for dict_instance_kernel in cat_int_kernel:
        if dict_instance_kernel['cat_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cat_covariate'])]
            k1 = ProductKernel(CatKernel(dict_instance_kernel['cat_covariate']), BinKernel(dict_instance['mask']))
        else:
            k1 = CatKernel(dict_instance_kernel['cat_covariate'])

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            k2 = ProductKernel(RbfKernel(dict_instance_kernel['cont_covariate'], latent_dim), BinKernel(dict_instance['mask']))
        else:
            k2 = RbfKernel(dict_instance_kernel['cont_covariate'], latent_dim)

        if dict_instance_kernel['cat_covariate'] == id_covariate:
            kernel1.append(ScaleKernel(ProductKernel(k1, k2), latent_dim).to(device))
        else:
            kernel0.append(ScaleKernel(ProductKernel(k1, k2), latent_dim).to(device))
    
    # interaction kernels (binary)
    for dict_instance_kernel in bin_int_kernel:
        if dict_instance_kernel['bin_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['bin_covariate'])]
            k1 = ProductKernel(BinKernel(dict_instance_kernel['bin_covariate']), BinKernel(dict_instance['mask']))
        else:
            k1 = BinKernel(dict_instance_kernel['bin_covariate'])

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            k2 = ProductKernel(RbfKernel(dict_instance_kernel['cont_covariate'], latent_dim), BinKernel(dict_instance['mask']))
        else:
            k2 = RbfKernel(dict_instance_kernel['cont_covariate'], latent_dim)

        kernel0.append(ScaleKernel(ProductKernel(k1, k2), latent_dim).to(device))

    return AdditiveKernel(kernel0).to(device), AdditiveKernel(kernel1).to(device)
