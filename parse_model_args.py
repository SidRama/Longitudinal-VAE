import argparse
import ast


"""
Helper code for loading parameters from parameter file or from command line
"""

class LoadFromFile (argparse.Action):
    """
    Read parameters from config file
    """
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().splitlines(), namespace)


class ModelArgs:
    """
    Runtime parameters for the L-VAE model
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the model')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=str, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_test_data', type=str, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_label', type=str, help='Name of label file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=str, help='Name of test label file', required=False)
        self.parser.add_argument('--csv_file_prediction_data', type=str, help='Name of prediction data file', required=False)
        self.parser.add_argument('--csv_file_prediction_label', type=str, help='Name of prediction label file', required=False)
        self.parser.add_argument('--csv_file_validation_data', type=str, help='Name of validation data file', required=False)
        self.parser.add_argument('--csv_file_validation_label', type=str, help='Name of validation label file', required=False)
        self.parser.add_argument('--csv_file_generation_data', type=str, help='Name of data file for image generation', required=False)
        self.parser.add_argument('--csv_file_generation_label', type=str, help='Name of label file for image generation', required=False)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--test_mask_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--prediction_mask_file', type=str, help='Name of prediction mask file', default=None)
        self.parser.add_argument('--validation_mask_file', type=str, help='Name of validation mask file', default=None)
        self.parser.add_argument('--generation_mask_file', type=str, help='Name of mask file for image generation', default=None)
        self.parser.add_argument('--dataset_type', required=False, choices=['RotatedMNIST', 'HealthMNIST', 'Physionet'],
                                 help='Type of dataset being used.')
        self.parser.add_argument('--latent_dim', type=int, default=2, help='Number of latent dimensions')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for RNN')
        self.parser.add_argument('--id_covariate', type=int, help='Index of ID (unique identifier) covariate')
        self.parser.add_argument('--M', type=int, help='Number of inducing points')
        self.parser.add_argument('--P', type=int, help='Number of unique instances')
        self.parser.add_argument('--T', type=int, help='Number of longitudinal samples per instance')
        self.parser.add_argument('--varying_T', type=str2bool, default=False, help='Varying number of samples per instance')
        self.parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
        self.parser.add_argument('--weight', type=float, default=1,
                                 help='Trade-off parameter balancing data reconstruction and latent space prior' +
                                      ' regularisation')
        self.parser.add_argument('--num_dim', type=int, help='Number of input dimensions', required=False)
        self.parser.add_argument('--num_samples', type=int, default=1, help='Number of Monte Carlo samples')
        self.parser.add_argument('--loss_function', type=str, default='mse', help='LVAE loss function for training (mse/nll)')
        self.parser.add_argument('--type_nnet', required=False, choices=['rnn', 'conv', 'simple'],
                                 help='Type of neural network for the encoder and decoder')
        self.parser.add_argument('--type_rnn', required=False, choices=['lstm', 'gru'],
                                 help='Type of rnn for rnn-encoder')
        self.parser.add_argument('--type_KL', required=False, choices=['closed', 'other', 'GPapprox', 'GPapprox_closed'],
                                 help='Type of loss computation')
        self.parser.add_argument('--constrain_scales', type=str2bool, default=False, required=False,
                                 help='Constrain the marginal variances')
        self.parser.add_argument('--model_params', type=str, default='model_params.pth',
                                 help='Pre-trained VAE parameters')
        self.parser.add_argument('--gp_model_folder', type=str, default='./pretrainedVAE',
                                 help='Pre-trained GP model parameters')
        self.parser.add_argument('--generate_plots', type=str2bool, default=False, help='Generate plots')
        self.parser.add_argument('--iter_num', type=int, default=1, help='Iteration number. Useful for multiple runs.')
        self.parser.add_argument('--test_freq', type=int, default=50, help='Period of computing test MSE.')
        self.parser.add_argument('--cat_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_kernel', type=ast.literal_eval)
        self.parser.add_argument('--sqexp_kernel', type=ast.literal_eval)
        self.parser.add_argument('--cat_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--covariate_missing_val', type=ast.literal_eval)
        self.parser.add_argument('--run_tests', type=str2bool, default=False,
                                 help='Perform tests using the trained model')
        self.parser.add_argument('--run_validation', type=str2bool, default=False,
                                 help='Test the model using a validation set')
        self.parser.add_argument('--generate_images', type=str2bool, default=False,
                                 help='Generate images of unseen individuals')
        self.parser.add_argument('--results_path', type=str, required=False, help='Path to results')
        self.parser.add_argument('--f', type=open, action=LoadFromFile)
        self.parser.add_argument('--mini_batch', type=str2bool, default=False, help='Use mini-batching for training.')
        self.parser.add_argument('--hensman', type=str2bool, default=False, help='Use true mini-batch training.')
        self.parser.add_argument('--variational_inference_training', type=str2bool, default=False, help='Use variational inference training.')
        self.parser.add_argument('--memory_dbg', type=str2bool, default=False, help='Debug memory usage in training')
        self.parser.add_argument('--natural_gradient', type=str2bool, default=True, help='Use natural gradients for parameters m and H')
        self.parser.add_argument('--natural_gradient_lr', type=float, default=0.01, help='Learning rate for variational parameters m and H if natural gradient is used')
        self.parser.add_argument('--subjects_per_batch', type=int, default=20, help='Number of subjects per batch in mini-batching.')
        self.parser.add_argument('--vy_fixed', type=str2bool, default=False, help='Use fixed variance for y in VAE')
        self.parser.add_argument('--vy_init', type=float, default=1.0, help='Initial variance for y in VAE')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Probability for dropout')
        self.parser.add_argument('--dropout_input', type=float, default=0.2, help='Probability for dropout at input layer')

    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt


class VAEArgs:
    """
    Runtime parameters for the VAE model
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the VAE pre-training')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=str, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_label', type=str, help='Name of label file', required=False)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--csv_file_test_data', type=str, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=str, help='Name of test label file', required=False)
        self.parser.add_argument('--test_mask_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--dataset_type', required=False, choices=['RotatedMNIST', 'HealthMNIST', 'Physionet'],
                                 help='Type of dataset being used.')
        self.parser.add_argument('--latent_dim', type=int, default=2, help='Number of latent dimensions')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for RNN')
        self.parser.add_argument('--id_covariate', type=int, help='Index of ID (unique identifier) covariate')
        self.parser.add_argument('--T', type=int, help='Number of longitudinal samples per instance')
        self.parser.add_argument('--varying_T', type=str2bool, default=False, help='Varying number of samples per instance')
        self.parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
        self.parser.add_argument('--num_dim', type=int, help='Number of input dimensions', required=False)
        self.parser.add_argument('--type_nnet', required=False, choices=['rnn', 'conv', 'simple'],
                                 help='Type of neural network for the encoder and decoder')
        self.parser.add_argument('--type_rnn', required=False, choices=['lstm', 'gru'],
                                 help='Type of rnn for rnn-encoder')
        self.parser.add_argument('--loss_function', type=str, default='nll', help='VAE loss function for training (mse/nll)')
        self.parser.add_argument('--iter_num', type=int, default=1, help='Iteration number. Useful for multiple runs')
        self.parser.add_argument('--vy_fixed', type=str2bool, default=False, help='Use fixed variance for y in VAE')
        self.parser.add_argument('--vy_init', type=float, default=1.0, help='Initial variance for y in VAE')
        self.parser.add_argument('--run_tests', type=str2bool, default=False,
                                 help='Perform tests using the trained model')
        self.parser.add_argument('--f', type=open, action=LoadFromFile)

    def parse_options(self):
        opt = vars(self.parser.parse_args())
        return opt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
