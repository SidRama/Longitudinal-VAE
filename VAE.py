import os, sys, torch, argparse
import pandas as pd
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math

from predict_HealthMNIST import VAEoutput
from dataset_def import HealthMNISTDatasetConv, RotatedMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDataset, \
    PhysionetDataset
from parse_model_args import VAEArgs
from model_test import VAEtest

class ConvVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False, p_input=0.2, p=0.5):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim
        self.p_input = p_input
        self.p = p

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_1 = nn.Dropout2d(p=self.p)  # spatial dropout

        # second convolution layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d_2 = nn.Dropout2d(p=self.p)

        self.fc1 = nn.Linear(32 * 9 * 9, 300)
        self.dropout1 = nn.Dropout(p=self.p)
        self.fc21 = nn.Linear(300, 30)
        self.dropout2 = nn.Dropout(p=self.p)
        self.fc211 = nn.Linear(30, self.latent_dim)
        self.fc221 = nn.Linear(30, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(latent_dim, 30)
        self.dropout3 = nn.Dropout(p=self.p)
        self.fc31 = nn.Linear(30, 300)
        self.dropout4 = nn.Dropout(p=self.p)
        self.fc4 = nn.Linear(300, 32 * 9 * 9)

        self.dropout2d_3 = nn.Dropout2d(p=self.p)
        # first transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        self.dropout2d_4 = nn.Dropout2d(p=self.p)
        # second transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        # convolution
        z = F.relu(self.conv1(x))
        z = self.dropout2d_1(self.pool1(z))
        z = F.relu(self.conv2(z))
        z = self.dropout2d_2(self.pool2(z))

        # MLP
        z = z.view(-1, 32 * 9 * 9)
        h1 = self.dropout1(F.relu(self.fc1(z)))
        h2 = self.dropout2(F.relu(self.fc21(h1)))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        # MLP
        x = self.dropout3(F.relu(self.fc3(z)))
        x = self.dropout4(F.relu(self.fc31(x)))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = self.dropout2d_3(x.view(-1, 32, 9, 9))
        x = self.dropout2d_4(F.relu(self.deconv1(x)))
        return torch.sigmoid(self.deconv2(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample_latent(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)


class SimpleVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with simple multi-layered perceptrons.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False):
        super(SimpleVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        self.fc1 = nn.Linear(num_dim, 300)
        self.fc21 = nn.Linear(300, 30)
        self.fc211 = nn.Linear(30, latent_dim)
        self.fc221 = nn.Linear(30, latent_dim)

        # decoder network
        self.fc3 = nn.Linear(latent_dim, 30)
        self.fc31 = nn.Linear(30, 300)
        self.fc4 = nn.Linear(300, num_dim)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc21(h1))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc31(h3))
        return torch.sigmoid(self.fc4(h4))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.num_dim))
        z = self.sample_latent(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)


if __name__ == "__main__":
    """
    This is used for pre-training.
    """

    # create parser and set variables
    opt = VAEArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    assert loss_function=='mse' or loss_function=='nll', ("Unknown loss function " + loss_function)
    assert ('T' in locals() and T is not None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))
    
    # set up dataset
    if type_nnet == 'conv':
        if dataset_type == 'HealthMNIST':
            dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                             mask_file=mask_file, root_dir=data_source_path,
                                             transform=transforms.ToTensor())
        elif dataset_type == 'RotatedMNIST':
            dataset = RotatedMNISTDatasetConv(data_file=csv_file_data,
                                              label_file=csv_file_label,
                                              mask_file=mask_file, root_dir=data_source_path,
                                              transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        if dataset_type == 'HealthMNIST':
            dataset = HealthMNISTDataset(csv_file_data=csv_file_data, csv_file_label=csv_file_label,
                                         mask_file=mask_file, root_dir=data_source_path,
                                         transform=transforms.ToTensor())
        elif dataset_type == 'RotatedMNIST':
            dataset = RotatedMNISTDataset(data_file=csv_file_data,
                                          label_file=csv_file_label,
                                          mask_file=mask_file, root_dir=data_source_path,
                                          transform=transforms.ToTensor())
        elif dataset_type == 'Physionet':
            dataset = PhysionetDataset(data_file=csv_file_data, root_dir=data_source_path)


    print('Length of dataset:  {}'.format(len(dataset)))
    Q = len(dataset[0]['label'])

    # set up Data Loader
    dataloader = DataLoader(dataset, min(len(dataset),256), shuffle=True, num_workers=4)

    vy = torch.Tensor(np.ones(num_dim) * vy_init)

    # set up model and send to GPU if available
    if type_nnet == 'conv':
        print('Using convolutional neural network')
        nnet_model = ConvVAE(latent_dim, num_dim, vy, vy_fixed).to(device)
    elif type_nnet == 'simple':
        print('Using standard MLP')
        nnet_model = SimpleVAE(latent_dim, num_dim, vy, vy_fixed).to(device)

    optimiser = torch.optim.Adam(nnet_model.parameters(), lr=1e-3)

    print(nnet_model.vy)

    net_train_loss = np.empty((0, 1))
    for epoch in range(1, epochs + 1):

        # start training VAE
        nnet_model.train()
        train_loss = 0
        recon_loss_sum = 0
        nll_loss = 0
        kld_loss = 0

        for batch_idx, sample_batched in enumerate(dataloader):
            data = sample_batched['digit']
            data = data.to(device)                                  # send to GPU
            mask = sample_batched['mask']
            mask = mask.to(device)
            label = sample_batched['label'].to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1)

            optimiser.zero_grad()                                   # clear gradients

            recon_batch, mu, log_var = nnet_model(data)

            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            if loss_function == 'nll':
                loss = torch.sum(nll + KLD)
            elif loss_function == 'mse':
                loss = torch.sum(recon_loss + KLD)

            loss.backward()                                         # compute gradients
            train_loss += loss.item()
            recon_loss_sum += recon_loss.sum().item()
            nll_loss += nll.sum().item()
            kld_loss += KLD.sum().item()
            
            optimiser.step()                                        # update parameters

        print('====> Epoch: {} - Average loss: {:.4f}  - KLD loss: {:.3f}  - NLL loss: {:.3f}  - Recon loss: {:.3f}'.format(epoch, train_loss, kld_loss, nll_loss, recon_loss_sum))
        net_train_loss = np.append(net_train_loss, train_loss)
        if epoch % 25 == 0:
            print(nnet_model.vy)
            if run_tests:
                VAEtest(test_dataset, nnet_model, type_nnet, id_covariate)
                VAEoutput(nnet_model, dataset, epoch, save_path, type_nnet, id_covariate)
            torch.save(nnet_model.state_dict(), os.path.join(save_path, 'model_params_vae_' + str(epoch) + '.pth'))
    
    print(nnet_model.vy)
    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'model_params_vae.pth'))
