import torch
from gpytorch.kernels import Kernel
from gpytorch.kernels import RBFKernel


"""
Kernel specification
"""
class BinKernel(Kernel):
    """
    Binary kernel
    """

    def __init__(self, value, **kwargs):
        """
        :param value: positive value (i.e. value = 1)
        :param kwargs:
        """
        super(BinKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.value = value

    def forward(self, x1, x2, **params):
        return (x1.squeeze(-1).unsqueeze(-1)+x2.squeeze(-1).unsqueeze(-2) == 2).type(torch.double)


class CatKernel(Kernel):
    """
    Categorical kernel
    """

    def forward(self, x1, x2, last_dim_is_batch=False, **param):
        return (x1.squeeze(-1).unsqueeze(-1)-x2.squeeze(-1).unsqueeze(-2) == 0).type(torch.double)


class CatKernelMod(Kernel):
    """
    A modified categorical kernel that ensures that each instance is independent.
    See: https://arxiv.org/abs/1912.03549
    """

    def __init__(self, num, **kwargs):
        """
        :param num: number of unique instances
        :param kwargs:
        """
        super(CatKernelMod, self).__init__(has_lengthscale=False, **kwargs)
        self.num = num

    def forward(self, x1, x2, **params):
        x1_mesh, x2_mesh = torch.meshgrid([x1.view(-1), x2.view(-1)])
        id_kern = (x1_mesh - x2_mesh == 0).double()
        other_kern = (x1_mesh - x2_mesh != 0).double()
        other_kern = (-1/(self.num - 1)) * other_kern

        return id_kern + other_kern


def RbfKernel(active_dims, batch_shape=None):
    """
    RBF kernel specification
    :param active_dims: index of covariate from auxiliary covariate matrix
    :return: instance of GPyTorch RBFKernel
    """
    if batch_shape is None:
        rbfKernel = RBFKernel(active_dims=active_dims)
    else:
        rbfKernel = RBFKernel(active_dims=active_dims, batch_shape=batch_shape)
    rbfKernel.initialize(lengthscale=2.5)
    return rbfKernel
