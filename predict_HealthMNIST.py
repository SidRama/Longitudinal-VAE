import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset_def import HealthMNISTDatasetConv, HealthMNISTDataset
from utils import batch_predict, batch_predict_varying_T

def gen_rotated_mnist_plot(X, recon_X, labels, seq_length=16, num_sets=3, save_file='recon.pdf'):
    """
    Function to generate rotated MNIST digits plots.
    
    """
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    fig.set_size_inches(9, 1.5 * num_sets)
    for j in range(num_sets):
        begin = seq_length * j
        end = seq_length * (j + 1)
        time_steps = labels[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray')
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file)
    plt.close('all')

def gen_rotated_mnist_seqrecon_plot_old(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate rotated MNIST digits.
    
    """
    num_sets = 4
    fig, ax = plt.subplots(2 * num_sets, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(3 * num_sets, 3 * num_sets)

    for j in range(num_sets):
        begin = seq_length_train * j
        end = seq_length_train * (j + 1)
        time_steps = labels_train[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j, int(t)].imshow(np.reshape(X[begin + i, :], [36, 36]), cmap='gray')

        begin = seq_length_full * j
        end = seq_length_full * (j + 1)
        time_steps = labels_recon[begin:end, 0]
        for i, t in enumerate(time_steps):
            ax[2 * j + 1, int(t)].imshow(np.reshape(recon_X[begin + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file)
    plt.close('all')


def gen_rotated_mnist_seqrecon_plot(X, recon_X, labels_recon, labels_train, save_file='recon_complete.pdf'):
    """
    Function to generate Health MNIST digits.
    
    """    
    num_sets = 8
    fig, ax = plt.subplots(4 * num_sets - 1, 20)
    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks([])
            ax__.set_yticks([])
            ax__.axis('off')
    plt.axis('off')
    seq_length_train = 20
    seq_length_full = 20
    fig.set_size_inches(12, 20)

    for j in range(num_sets):
        begin_data = seq_length_train*j
        end_data = seq_length_train*(j+1)

        begin_label = seq_length_full*2*j
        mid_label = seq_length_full*(2*j+1)
        end_label = seq_length_full*2*(j+1)
        
        time_steps = labels_train[begin_data:end_data, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j, int(t)].imshow(np.reshape(X[begin_data + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[begin_label:mid_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 1, int(t)].imshow(np.reshape(recon_X[begin_label + i, :], [36, 36]), cmap='gray')
        
        time_steps = labels_train[mid_label:end_label, 0]
        for i, t in enumerate(time_steps):
            ax[4 * j + 2, int(t)].imshow(np.reshape(recon_X[mid_label + i, :], [36, 36]), cmap='gray')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close('all')

def recon_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                       covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, 
                       prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Generating images - length of dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)

            filename = 'recon_complete.pdf' if epoch == -1 else 'recon_complete_best.pdf'

            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, filename))

def variational_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, 
                             covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, 
                             prediction_mu, epoch, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to generate rotated MNIST digits.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Length of generation dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            print('Prediction size: ' + str(Z_pred.shape))
            recon_Z = nnet_model.decode(Z_pred)
            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, 'recon_complete_' + str(epoch) + '.pdf'))

def VAEoutput(nnet_model, dataset, epoch, save_path, type_nnet, id_covariate):
    """
    Function to obtain output of VAE.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Batch size must be a multiple of T
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
            # no mini-batching. Instead get a mini-batch of size 4000
            label = sample_batched['label'].to(device)
            data = sample_batched['digit'].to(device)

            recon_batch, mu, log_var = nnet_model(data)

            gen_rotated_mnist_plot(data[40:200, :].cpu(), recon_batch[40:200, :].cpu(), label[40:200, :].cpu(), seq_length=20, num_sets=8,
                                   save_file=os.path.join(save_path, 'recon_VAE_' + str(epoch) + '.pdf'))
            break

def predict_generate(csv_file_test_data, csv_file_test_label, csv_file_test_mask, dataset, generation_dataset, nnet_model, results_path, covar_module0, covar_module1, likelihoods, type_nnet, latent_dim, data_source_path, prediction_x, prediction_mu, zt_list, P, T, id_covariate, varying_T=False):
    """
    Function to perform prediction and visualise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    print('Length of dataset:  {}'.format(len(dataset)))

    # set up Data Loader
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)

    # Get values for GP initialisation:
    Z = torch.tensor([]).to(device)
    mu = torch.tensor([]).to(device)
    log_var = torch.tensor([]).to(device)
    data_train = torch.tensor([]).to(device)
    label_train = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader):
            # no mini-batching. Instead get a mini-batch of size 4000
            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            label_train = label
            data_train = data
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)

            gen_rotated_mnist_plot(data[40:100, :].cpu(), recon_batch[40:100, :].cpu(), label[40:100, :].cpu(), seq_length=20,
                                   save_file=os.path.join(results_path, 'recon_train.pdf'))
            break

    if type_nnet == 'conv':
        test_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_test_data,
                                              csv_file_label=csv_file_test_label,
                                              mask_file=csv_file_test_mask, root_dir=data_source_path,
                                              transform=transforms.ToTensor())

    elif type_nnet == 'simple':
        test_dataset = HealthMNISTDataset(csv_file_data=csv_file_test_data,
                                          csv_file_label=csv_file_test_label,
                                          mask_file=csv_file_test_mask, root_dir=data_source_path,
                                          transform=transforms.ToTensor())

    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            
            Z = nnet_model.sample_latent(mu, log_var)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)  # reconstruction loss
            print('Decoder loss: ' + str(torch.mean(recon_loss)))

            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            recon_Z = nnet_model.decode(Z_pred)
            [recon_loss_GP, nll] = nnet_model.loss_function(recon_Z, data, mask)  # reconstruction loss
            print('Decoder loss (GP): ' + str(torch.mean(recon_loss_GP)))
            pred_results = np.array([torch.mean(recon_loss).cpu().numpy(), torch.mean(recon_loss_GP).cpu().numpy()])
            np.savetxt(os.path.join(results_path, 'result_error.csv'), pred_results)

    print('Length of generation dataset:  {}'.format(len(generation_dataset)))
    dataloader_full = DataLoader(generation_dataset, batch_size=len(generation_dataset), shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_full):
            # no mini-batching. Instead get a mini-batch of size 4000

            label_id = sample_batched['idx']
            label = sample_batched['label']
            data = sample_batched['digit']
            data = data.double().to(device)
            mask = sample_batched['mask']
            mask = mask.to(device)
            covariates = torch.cat((label[:, :id_covariate], label[:, id_covariate+1:]), dim=1).double().to(device)

            recon_batch, mu, log_var = nnet_model(data)
            Z = nnet_model.sample_latent(mu, log_var)

            Z_pred = torch.tensor([], dtype=torch.double).to(device)
            test_x = label.type(torch.DoubleTensor).to(device)
            Z_pred = batch_predict_varying_T(latent_dim, covar_module0, covar_module1, likelihoods, prediction_x, test_x, prediction_mu, zt_list, id_covariate, eps=1e-6)

            print('Prediction size: ' + str(Z_pred.shape))
            recon_Z = nnet_model.decode(Z_pred)

            gen_rotated_mnist_seqrecon_plot(data[0:160, :].cpu(), recon_Z[0:320, :].cpu(), label[0:320, :].cpu(), label[0:320, :].cpu(),
                                            save_file=os.path.join(results_path, 'recon_complete.pdf'))

