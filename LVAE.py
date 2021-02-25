import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import gpytorch

from timeit import default_timer as timer

from GP_def import ExactGPModel
from VAE import ConvVAE, SimpleVAE
from dataset_def import HealthMNISTDatasetConv, RotatedMNISTDatasetConv, HealthMNISTDataset, RotatedMNISTDataset, \
    PhysionetDataset
from elbo_functions import elbo, KL_closed, deviance_upper_bound
from kernel_gen import generate_kernel, generate_kernel_approx, generate_kernel_batched
from model_test import MSE_test, MSE_test_GPapprox
from predict_HealthMNIST import recon_complete_gen
from parse_model_args import ModelArgs
from training import hensman_training, minibatch_training, standard_training, variational_inference_optimization
from validation import validate
from utils import SubjectSampler, VaryingLengthSubjectSampler, VaryingLengthBatchSampler, HensmanDataLoader

eps = 1e-6


if __name__ == "__main__":
    """
    Root file for running L-VAE.
    
    Run command: python LVAE.py --f=path_to_config-file.txt 
    """

    # create parser and set variables
    opt = ModelArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    assert not(hensman and mini_batch)
    assert loss_function=='mse' or loss_function=='nll', ("Unknown loss function " + loss_function)
    assert not varying_T or hensman, "varying_T can't be used without hensman"

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


    #Set up prediction dataset
    if run_tests or generate_images:
        if dataset_type == 'HealthMNIST' and type_nnet == 'conv':
            prediction_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_prediction_data,
                                                        csv_file_label=csv_file_prediction_label,
                                                        mask_file=prediction_mask_file, root_dir=data_source_path,
                                                        transform=transforms.ToTensor())
            print('Length of prediction dataset:  {}'.format(len(prediction_dataset)))
        elif dataset_type == 'RotatedMNIST':
            prediction_dataset = RotatedMNISTDatasetConv(data_file=csv_file_prediction_data,
                                                         label_file=csv_file_prediction_label,
                                                         mask_file=prediction_mask_file, root_dir=data_source_path,
                                                         transform=transforms.ToTensor())
            print('Length of prediction dataset:  {}'.format(len(prediction_dataset)))
    else:
        prediction_dataset = None

    #Set up dataset for image generation
    if generate_images:
        if type_nnet == 'conv':
            generation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_generation_data,
                                                        csv_file_label=csv_file_generation_label,
                                                        mask_file=generation_mask_file,
                                                        root_dir=data_source_path,
                                                        transform=transforms.ToTensor())

        elif type_nnet == 'simple':
            generation_dataset = HealthMNISTDataset(csv_file_data=csv_file_generation_data,
                                                    csv_file_label=csv_file_generation_label,
                                                    mask_file=generation_mask_file,
                                                    root_dir=data_source_path,
                                                    transform=transforms.ToTensor())

    else:
        generation_dataset = None

    #Set up validation dataset
    if run_validation:
        if dataset_type == 'HealthMNIST':
            validation_dataset = HealthMNISTDatasetConv(csv_file_data=csv_file_validation_data,
                                                        csv_file_label=csv_file_validation_label,
                                                        mask_file=validation_mask_file, root_dir=data_source_path,
                                                        transform=transforms.ToTensor())
            print('Length of validation dataset:  {}'.format(len(validation_dataset)))
        elif dataset_type == 'RotatedMNIST':
            validation_dataset = RotatedMNISTDatasetConv(data_file=csv_file_validation_data,
                                                         label_file=csv_file_validation_label,
                                                         mask_file=validation_mask_file, root_dir=data_source_path,
                                                         transform=transforms.ToTensor())
            print('Length of prediction dataset:  {}'.format(len(validation_dataset)))

    else:
        validation_dataset = None

    print('Length of dataset:  {}'.format(len(dataset)))
    N = len(dataset)

    if not N:
        print("ERROR: Dataset is empty")
        exit(1)

    Q = len(dataset[0]['label'])

    # set up model and send to GPU if available
    if type_nnet == 'conv':
        print('Using convolutional neural network')
        nnet_model = ConvVAE(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed,
                             p_input=dropout_input, p=dropout).double().to(device)
    elif type_nnet == 'simple':
        print('Using standard MLP')
        nnet_model = SimpleVAE(latent_dim, num_dim, vy_init, vy_fixed).to(device)

    # Load pre-trained encoder/decoder parameters if present
    try:
        nnet_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))
        print('Loaded pre-trained values.')
    except:
        print('Did not load pre-trained values.')

    nnet_model = nnet_model.double().to(device)

    # set up Data Loader for GP initialisation
    # Kalle: Hard-coded batch size 1000
    setup_dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4)

    # Get values for GP initialisation:
    Z = torch.zeros(N, latent_dim, dtype=torch.double).to(device)
    train_x = torch.zeros(N, Q, dtype=torch.double).to(device)
    nnet_model.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(setup_dataloader):
            # no mini-batching. Instead get a batch of dataset size
            label_id = sample_batched['idx']
            train_x[label_id] = sample_batched['label'].double().to(device)
            data = sample_batched['digit'].double().to(device)

            covariates = torch.cat((train_x[label_id, :id_covariate], train_x[label_id, id_covariate+1:]), dim=1)
            
            mu, log_var = nnet_model.encode(data)
            Z[label_id] = nnet_model.sample_latent(mu, log_var)

    covar_module = []
    covar_module0 = []
    covar_module1 = []
    zt_list = []
    likelihoods = []
    gp_models = []
    adam_param_list = []

    if hensman:
        likelihoods = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([latent_dim]),
            noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device)

        if constrain_scales:
            likelihoods.noise = 1
            likelihoods.raw_noise.requires_grad = False

        covar_module0, covar_module1 = generate_kernel_batched(latent_dim,
                                                               cat_kernel, bin_kernel, sqexp_kernel,
                                                               cat_int_kernel, bin_int_kernel,
                                                               covariate_missing_val, id_covariate)

        gp_model = ExactGPModel(train_x, Z.type(torch.DoubleTensor), likelihoods,
                                covar_module0 + covar_module1).to(device)

        # initialise inducing points
        zt_list = torch.zeros(latent_dim, M, Q, dtype=torch.double).to(device)
        for i in range(latent_dim):
            #zt_list[i] = train_x[np.random.choice(N, M, replace=False)].clone().detach()
            #zt_list[i]=torch.cat((train_x[0:22], train_x[110:132]), dim=0).clone().detach()
            zt_list[i]=torch.cat((train_x[0:60], train_x[2000:2060]), dim=0).clone().detach()
        #zt_list.requires_grad_(True)

        adam_param_list.append({'params': covar_module0.parameters()})
        adam_param_list.append({'params': covar_module1.parameters()})
        #adam_param_list.append({'params': zt_list})

        covar_module0.train().double()
        covar_module1.train().double()
        likelihoods.train().double()

        try:
            gp_model.load_state_dict(torch.load(os.path.join(gp_model_folder, 'gp_model.pth'), map_location=torch.device(device)))
            zt_list = torch.load(os.path.join(gp_model_folder, 'zt_list.pth'), map_location=torch.device(device))
            print('Loaded GP models')
        except:
            print('GP model loading failed!')
            pass

        m = torch.randn(latent_dim, M, 1).double().to(device).detach()
        H = (torch.randn(latent_dim, M, M)/10).double().to(device).detach()
    
        if natural_gradient:
            H = torch.matmul(H, H.transpose(-1, -2)).detach().requires_grad_(False)
        
        try:
            m = torch.load(os.path.join(gp_model_folder,'m.pth'), map_location=torch.device(device)).detach()
            H = torch.load(os.path.join(gp_model_folder,'H.pth'), map_location=torch.device(device)).detach()
            print('Loaded natural gradient values')
        except:
            print('Loading natural gradient values failed!')
            pass

        if not natural_gradient:
            adam_param_list.append({'params': m})
            adam_param_list.append({'params': H})
            m.requires_grad_(True)
            H.requires_grad_(True)

    else:
        for i in range(0, latent_dim):
            likelihoods.append(gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1.000E-08)).to(device))

            if constrain_scales:
                likelihoods[i].noise = 1
                likelihoods[i].raw_noise.requires_grad = False

            # set up additive GP prior
            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                additive_kernel0, additive_kernel1 = generate_kernel_approx(cat_kernel, bin_kernel, sqexp_kernel,
                                                                            cat_int_kernel, bin_int_kernel,
                                                                            covariate_missing_val, id_covariate)
                covar_module0.append(additive_kernel0.to(device))           # additive kernel without id covariate
                covar_module1.append(additive_kernel1.to(device))           # additive kernel with id covariate
                gp_models.append(ExactGPModel(train_x, Z[:, i].view(-1).type(torch.DoubleTensor), likelihoods[i],
                                              covar_module0[i] + covar_module1[i]).to(device))
                #z_init = train_x[np.random.choice(N, M, replace=False)]     # initialise inducing points
                #Hardcoded for generation_test3
                z_init=torch.cat((train_x[20:60], train_x[10000:10040]), dim=0)
                #Hardcoded for generation_test
                #z_init=torch.cat((train_x[0:40], train_x[2000:2040]), dim=0)
                zt = torch.nn.Parameter(z_init.clone().cpu().double().detach(), requires_grad=False)
                zt_list.append(zt)
                adam_param_list.append({'params': covar_module0[i].parameters()})
                adam_param_list.append({'params': covar_module1[i].parameters()})
                #adam_param_list.append({'params': zt_list[i]})
            else:
                additive_kernel = generate_kernel(cat_kernel, bin_kernel, sqexp_kernel, cat_int_kernel, bin_int_kernel,
                                                  covariate_missing_val)
                covar_module.append(additive_kernel.to(device))             # additive kernel GP prior
                gp_models.append(ExactGPModel(train_x, Z[:, i].view(-1).type(torch.DoubleTensor), likelihoods[i],
                                              covar_module[i]).to(device))
                adam_param_list.append({'params': gp_models[i].parameters()})

            gp_models[i].train().double()
            likelihoods[i].train().double()

        for i in range(0, latent_dim):
            gp_model_name = 'gp_model' + str(i) + '.pth'
            zt_list_name = 'zt_list' + str(i) + '.pth'
            try:
                gp_models[i].load_state_dict(torch.load(os.path.join(gp_model_folder,gp_model_name), map_location=torch.device(device)))
                zt_list[i] = torch.load(os.path.join(gp_model_folder,zt_list_name), map_location=torch.device('cpu'))
            except:
                pass

    nnet_model.train()
    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)


    if memory_dbg:
        print("Max memory allocated during initialisation: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)

    if type_KL == 'closed':
        covar_modules = [covar_module]
    elif type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
        covar_modules = [covar_module0, covar_module1]

    start = timer()
    if hensman:
        _ = hensman_training(nnet_model, type_nnet, epochs, dataset,
                             optimiser, type_KL, num_samples, latent_dim,
                             covar_module0, covar_module1, likelihoods, m,
                             H, zt_list, P, T, varying_T, Q, weight,
                             id_covariate, loss_function, natural_gradient, natural_gradient_lr,
                             subjects_per_batch, memory_dbg, eps,
                             results_path, validation_dataset,
                             generation_dataset, prediction_dataset, gp_model, csv_file_test_data=csv_file_test_data,
                             csv_file_test_label=csv_file_test_label, test_mask_file=test_mask_file,
                             data_source_path=data_source_path)
        m, H = _[5], _[6]
    elif mini_batch:
        _ = minibatch_training(nnet_model, type_nnet, epochs, dataset,
                               optimiser, type_KL, num_samples, latent_dim,
                               covar_module0, covar_module1, likelihoods,
                               zt_list, P, T, Q, weight,
                               id_covariate, loss_function, memory_dbg, eps, results_path,
                               validation_dataset, generation_dataset, prediction_dataset)
    elif variational_inference_training:
        variational_inference_optimization(nnet_model, type_nnet, epochs, dataset, prediction_dataset,
                                           optimiser, latent_dim, covar_module0, covar_module1,
                                           likelihoods, zt_list, P, T, Q, weight, constrain_scales,
                                           id_covariate, loss_function, memory_dbg, eps,
                                           results_path, save_path, gp_model_folder, generation_dataset)
    else:
        _ = standard_training(nnet_model, type_nnet, epochs, dataset,
                              optimiser, type_KL, num_samples, latent_dim,
                              covar_modules, likelihoods, zt_list,
                              id_covariate, P, T, Q, weight, constrain_scales,
                              loss_function, memory_dbg, eps, validation_dataset,
                              generation_dataset, prediction_dataset)
    print("Duration of training: {:.2f} seconds".format(timer()-start))
    
    if memory_dbg:
        print("Max memory allocated during training: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)

    penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr = _[0], _[1], _[2], _[3], _[4]
    print('Best results in epoch: ' + str(_[7]))
    # saving
    print('Saving')
    pd.to_pickle([penalty_term_arr, net_train_loss_arr, nll_loss_arr, recon_loss_arr, gp_loss_arr],
                 os.path.join(save_path, 'diagnostics.pkl'))

    pd.to_pickle([train_x, mu, log_var, Z, label_id], os.path.join(save_path, 'plot_values.pkl'))
    torch.save(nnet_model.state_dict(), os.path.join(save_path, 'final-vae_model.pth'))

    if hensman:
        try:
            torch.save(gp_model.state_dict(), os.path.join(save_path, 'gp_model.pth'))
            torch.save(zt_list, os.path.join(save_path, 'zt_list.pth'))
            torch.save(m, os.path.join(save_path, 'm.pth'))
            torch.save(H, os.path.join(save_path, 'H.pth'))
        except:
            pass

    else:
        for i in range(0, latent_dim):
            gp_model_name = 'gp_model' + str(i) + '.pth'
            zt_list_name = 'zt_list' + str(i) + '.pth'
            torch.save(gp_models[i].state_dict(), os.path.join(save_path, gp_model_name))
            try:
                torch.save(zt_list[i], os.path.join(save_path, zt_list_name))
            except:
                pass
    
    if memory_dbg:
        print("Max memory allocated during saving and post-processing: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)   

    nnet_model.eval()

    if run_validation:
        dataloader = DataLoader(dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(dataset, id_covariate), subjects_per_batch), num_workers=4)
        full_mu = torch.zeros(len(dataset), latent_dim, dtype=torch.double).to(device)
        prediction_x = torch.zeros(len(dataset), Q, dtype=torch.double).to(device)
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(dataloader):
                label_id = sample_batched['idx']
                prediction_x[label_id] = sample_batched['label'].double().to(device)
                data = sample_batched['digit'].double().to(device)
                covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)

                mu, log_var = nnet_model.encode(data)
                full_mu[label_id] = mu
            covar_module0.eval()
            covar_module1.eval()
            validate(nnet_model, type_nnet, validation_dataset, type_KL, num_samples, latent_dim, covar_module0, covar_module1, likelihoods, zt_list, T, weight, full_mu, prediction_x, id_covariate, loss_function, eps=1e-6)

    if run_tests or generate_images:
        prediction_dataloader = DataLoader(prediction_dataset, batch_sampler=VaryingLengthBatchSampler(VaryingLengthSubjectSampler(prediction_dataset, id_covariate), subjects_per_batch), num_workers=4)
        full_mu = torch.zeros(len(prediction_dataset), latent_dim, dtype=torch.double).to(device)
        prediction_x = torch.zeros(len(prediction_dataset), Q, dtype=torch.double).to(device)

        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(prediction_dataloader):
                label_id = sample_batched['idx']
                prediction_x[label_id] = sample_batched['label'].double().to(device)
                data = sample_batched['digit'].double().to(device)
                covariates = torch.cat((prediction_x[label_id, :id_covariate], prediction_x[label_id, id_covariate+1:]), dim=1)

                mu, log_var = nnet_model.encode(data)
                full_mu[label_id] = mu

    # MSE test
    if run_tests:
        with torch.no_grad():
            covar_module0.eval()
            covar_module1.eval()
            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                MSE_test_GPapprox(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet,
                                  nnet_model, covar_module0, covar_module1, likelihoods,  results_path, latent_dim, prediction_x,
                                  full_mu, zt_list, P, T, id_covariate, varying_T)
            else:
                MSE_test(csv_file_test_data, csv_file_test_label, test_mask_file, data_source_path, type_nnet,
                         nnet_model, covar_module, likelihoods, results_path, latent_dim, prediction_x, full_mu)
    
    if generate_images:
        covar_module0.eval()
        covar_module1.eval()
        with torch.no_grad():
            if type_KL == 'GPapprox' or type_KL == 'GPapprox_closed':
                recon_complete_gen(generation_dataset, nnet_model, type_nnet, results_path, covar_module0, covar_module1, likelihoods, latent_dim, data_source_path, prediction_x, full_mu, -1, zt_list, P, T, id_covariate, varying_T)


    if memory_dbg:
        print("Max memory allocated during tests: {:.2f} MBs".format(torch.cuda.max_memory_allocated(device)/(1024**2)))
        torch.cuda.reset_max_memory_allocated(device)

