import gpytorch
import torch


"""
GP model definitions
"""
class ExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP model definition
    """

    def __init__(self, train_x, train_y, likelihood, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ScaledExactGPModel(gpytorch.models.ExactGP):
    """
    Test for marginal likelihood (scale) constraining
    """

    def __init__(self, train_x, train_y, likelihood, covar_module):
        super(ScaledExactGPModel, self).__init__(train_x, train_y, likelihood)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = covar_module
        self.a = torch.nn.Parameter(torch.zeros(len(self.covar_module.kernels),
                                                dtype=torch.double), requires_grad=True)

    def evaluate_kernel(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        covar_x_list = [None] * len(self.covar_module.kernels)
        b = torch.nn.Softmax(dim=0)(torch.cat((self.a.to(device),
                                               torch.zeros(1, dtype=torch.double).to(device)))).to(device)
        for k in range(len(self.covar_module.kernels)):
            covar_x_list[k] = b[k] * (self.covar_module.kernels[k](x)).evaluate()
        covar_x = sum(covar_x_list) + b[-1] * torch.eye(x.shape[0]).to(device)
        return covar_x

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean_x = self.mean_module(x)
        covar_x_list = [None]*len(self.covar_module.kernels)
        b = torch.nn.Softmax(dim=0)(torch.cat((self.a.to(device),
                                               torch.zeros(1, dtype=torch.double).to(device)))).to(device)
        for k in range(len(self.covar_module.kernels)):
            covar_x_list[k] = b[k] * (self.covar_module.kernels[k](x)).evaluate()
        covar_x = sum(covar_x_list) + b[-1] * torch.eye(x.shape[0]).to(device)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
