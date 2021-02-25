import torch
from kernel_spec import BinKernel, CatKernel, RbfKernel
from gpytorch.kernels import AdditiveKernel, ProductKernel, ScaleKernel

"""
Helper functions for generating the additive kernels based on specification in config file.
"""

def generate_kernel(cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel, covariate_missing_val):
    """
    Generate the additive kernel

    :param cat_kernel: list of indices from the covariate matrix for the categorical kernel
    :param bin_kernel:  list of indices from the covariate matrix for the binary kernel
    :param sqexp_kernel:  list of indices from the covariate matrix for the squared exponential kernel
    :param cat_int_kernel:  list of dictionaries with indices for the interaction kernel between a categorical and continuous covariate
                            E.g.: [{'cont_covariate':0, 'cat_covariate':2}, {'cont_covariate':0, 'cat_covariate':3}]
    :param bin_int_kernel:  list of dictionaries with indices for the interaction kernel between a binary and continuous covariate
                            E.g.: [{'cont_covariate':1, 'bin_covariate':4}]
    :param covariate_missing_val:  list of dictionaries with indices from the covariate matrix with missing values
                            and their corresponding masks
                            E.g.: [{'covariate':0, 'mask': 3}]
    :return: an instance of GPyTorch AdditiveKernel
    """
    covariate_missing = [dict_instance['covariate'] for dict_instance in covariate_missing_val]
    additive_kernel = AdditiveKernel()

    # categorical kernels
    for idx in cat_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            additive_kernel.kernels.append(ScaleKernel(
                CatKernel(active_dims=idx) *
                BinKernel(active_dims=dict_instance['mask'], value=1)))
        else:
            additive_kernel.kernels.append(ScaleKernel(CatKernel(active_dims=idx)))

    # continuous kernels
    for idx in sqexp_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            additive_kernel.kernels.append(ScaleKernel(
                RbfKernel(active_dims=idx) *
                BinKernel(active_dims=dict_instance['mask'], value=1)))
        else:
            additive_kernel.kernels.append(ScaleKernel(RbfKernel(active_dims=idx)))

    # binary kernels
    for idx in bin_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            additive_kernel.kernels.append(ScaleKernel(
                BinKernel(active_dims=idx, value=1) *
                BinKernel(active_dims=dict_instance['mask'], value=1)))
        else:
            additive_kernel.kernels.append(ScaleKernel(BinKernel(active_dims=idx, value=1)))

    # interaction kernels (categorical)
    for dict_instance_kernel in cat_int_kernel:
        if dict_instance_kernel['cat_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cat_covariate'])]
            masked_kernel1 = CatKernel(active_dims=dict_instance_kernel['cat_covariate']) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel1 = CatKernel(active_dims=dict_instance_kernel['cat_covariate'])

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate']) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'])

        additive_kernel.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2)))

    # interaction kernels (binary)
    for dict_instance_kernel in bin_int_kernel:
        if dict_instance_kernel['bin_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['bin_covariate'])]
            masked_kernel1 = BinKernel(active_dims=dict_instance_kernel['bin_covariate'], value=1) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel1 = BinKernel(active_dims=dict_instance_kernel['bin_covariate'], value=1)

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate']) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'])

        additive_kernel.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2)))

    return additive_kernel


def generate_kernel_approx(cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel, covariate_missing_val,
                           id_covariate):
    """
    Generate two sets of additive kernels. One with id covariate and the other without the id covariate

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

    covariate_missing = [dict_instance['covariate'] for dict_instance in covariate_missing_val]
    additive_kernel0 = AdditiveKernel()
    additive_kernel1 = AdditiveKernel()

    # categorical kernels
    for idx in cat_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            if idx == id_covariate:
                additive_kernel1.kernels.append(ScaleKernel(
                    CatKernel(active_dims=idx) *
                    BinKernel(active_dims=dict_instance['mask'], value=1)))
            else:
                additive_kernel0.kernels.append(ScaleKernel(
                    CatKernel(active_dims=idx) *
                    BinKernel(active_dims=dict_instance['mask'], value=1)))
        else:
            if idx == id_covariate:
                additive_kernel1.kernels.append(ScaleKernel(CatKernel(active_dims=idx)))
            else:
                additive_kernel0.kernels.append(ScaleKernel(CatKernel(active_dims=idx)))

    # continuous kernels
    for idx in sqexp_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            additive_kernel0.kernels.append(ScaleKernel(
                RbfKernel(active_dims=idx) *
                BinKernel(active_dims=dict_instance['mask'], value=1)))
        else:
            additive_kernel0.kernels.append(ScaleKernel(RbfKernel(active_dims=idx)))

    # binary kernels
    for idx in bin_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            additive_kernel0.kernels.append(ScaleKernel(
                BinKernel(active_dims=idx, value=1) *
                BinKernel(active_dims=dict_instance['mask'], value=1)))
        else:
            additive_kernel0.kernels.append(ScaleKernel(BinKernel(active_dims=idx, value=1)))

    # interaction kernels (categorical)
    for dict_instance_kernel in cat_int_kernel:
        if dict_instance_kernel['cat_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cat_covariate'])]
            masked_kernel1 = CatKernel(active_dims=dict_instance_kernel['cat_covariate']) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel1 = CatKernel(active_dims=dict_instance_kernel['cat_covariate'])

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate']) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'])

        if dict_instance_kernel['cat_covariate'] == id_covariate:
            additive_kernel1.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2)))
        else:
            additive_kernel0.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2)))

    # interaction kernels (binary)
    for dict_instance_kernel in bin_int_kernel:
        if dict_instance_kernel['bin_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['bin_covariate'])]
            masked_kernel1 = BinKernel(active_dims=dict_instance_kernel['bin_covariate'], value=1) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel1 = BinKernel(active_dims=dict_instance_kernel['bin_covariate'], value=1)

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate']) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'])

        additive_kernel0.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2)))

    return additive_kernel0, additive_kernel1

def generate_kernel_batched(latent_dim, cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel, covariate_missing_val,
                            id_covariate):
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
    additive_kernel0 = AdditiveKernel()
    additive_kernel1 = AdditiveKernel()

    # categorical kernels
    for idx in cat_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            if idx == id_covariate:
                additive_kernel1.kernels.append(ScaleKernel(
                    CatKernel(active_dims=idx) *
                    BinKernel(active_dims=dict_instance['mask'], value=1),
                    batch_shape=torch.Size([latent_dim])))
            else:
                additive_kernel0.kernels.append(ScaleKernel(
                    CatKernel(active_dims=idx) *
                    BinKernel(active_dims=dict_instance['mask'], value=1),
                    batch_shape=torch.Size([latent_dim])))
        else:
            if idx == id_covariate:
                additive_kernel1.kernels.append(ScaleKernel(CatKernel(active_dims=idx), batch_shape=torch.Size([latent_dim])))
            else:
                additive_kernel0.kernels.append(Scalekernel(CatKernel(active_dims=idx), batch_shape=torch.Size([latent_dim])))

    # continuous kernels
    for idx in sqexp_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            additive_kernel0.kernels.append(ScaleKernel(
                RbfKernel(active_dims=idx, batch_shape=torch.Size([latent_dim])) *
                BinKernel(active_dims=dict_instance['mask'], value=1),
                batch_shape=torch.Size([latent_dim])))
        else:
            additive_kernel0.kernels.append(ScaleKernel(RbfKernel(active_dims=idx, batch_shape=torch.Size([latent_dim])),
                                                        batch_shape=torch.Size([latent_dim])))

    # binary kernels
    for idx in bin_kernel:
        if idx in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(idx)]
            additive_kernel0.kernels.append(ScaleKernel(
                BinKernel(active_dims=idx, value=1) *
                BinKernel(active_dims=dict_instance['mask'], value=1),
                batch_shape=torch.Size([latent_dim])))
        else:
            additive_kernel0.kernels.append(ScaleKernel(BinKernel(active_dims=idx, value=1),
                                                        batch_shape=torch.Size([latent_dim])))

    # interaction kernels (categorical)
    for dict_instance_kernel in cat_int_kernel:
        if dict_instance_kernel['cat_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cat_covariate'])]
            masked_kernel1 = CatKernel(active_dims=dict_instance_kernel['cat_covariate']) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel1 = CatKernel(active_dims=dict_instance_kernel['cat_covariate'])

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'], batch_shape=torch.Size([latent_dim])) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'], batch_shape=torch.Size([latent_dim]))

        if dict_instance_kernel['cat_covariate'] == id_covariate:
            additive_kernel1.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2),
                                                        batch_shape=torch.Size([latent_dim])))
        else:
            additive_kernel0.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2),
                                                        batch_shape=torch.Size([latent_dim])))

    # interaction kernels (binary)
    for dict_instance_kernel in bin_int_kernel:
        if dict_instance_kernel['bin_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['bin_covariate'])]
            masked_kernel1 = BinKernel(active_dims=dict_instance_kernel['bin_covariate'], value=1) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel1 = BinKernel(active_dims=dict_instance_kernel['bin_covariate'], value=1)

        if dict_instance_kernel['cont_covariate'] in covariate_missing:
            dict_instance = covariate_missing_val[covariate_missing.index(dict_instance_kernel['cont_covariate'])]
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'], batch_shape=torch.Size([latent_dim])) * BinKernel(
                active_dims=dict_instance['mask'], value=1)
        else:
            masked_kernel2 = RbfKernel(active_dims=dict_instance_kernel['cont_covariate'], batch_shape=torch.Size([latent_dim]))

        additive_kernel0.kernels.append(ScaleKernel(ProductKernel(masked_kernel1, masked_kernel2),
                                                    batch_shape=torch.Size([latent_dim])))

    return additive_kernel0.to(device), additive_kernel1.to(device)
