import argparse
from cmath import nan
import enum
import logging
from re import L
import sys
import os
import copy 
from configparser import ConfigParser

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import warnings
#warnings.filterwarnings('error')

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import UTIL_LOSSES, LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS, UTILTIIES 
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining

from torch.utils.data import Dataset, DataLoader

from models.vision.cnn import CNN
from models.vision.cnn import Trainer as CNN_Trainer  

import torch 
from torch import optim
import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 



from os import listdir
from os.path import isfile, join

import pandas as pd 
import numpy as np 
import ast 
import matplotlib.pyplot as plt
import scipy.optimize as optimize

from sklearn.linear_model import LinearRegression

import seaborn as sns 

import statsmodels.api as sm

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

from sklearn.metrics import brier_score_loss

def softmax(utilities, tau):
    try:
        distribution = np.exp(utilities * tau) / np.sum(np.exp(utilities * tau))
    except Exception as e: 
        deterministic = np.zeros_like(utilities)
        max_idx = np.argmax(utilities)
        deterministic[max_idx] = 1
        return deterministic
    
    if np.isnan(distribution).any():
        deterministic = np.zeros_like(utilities)
        max_idx = np.argmax(utilities)
        deterministic[max_idx] = 1
        return deterministic
    
    return distribution


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('--bvae_folder', type=str, default='./marbles/bvae/',
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-ut', '--utility-type',
                       default=default_config['utility'], choices=UTILTIIES,
                       help='Type of utility prediction model to use.')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-ul', '--util-loss',
                       default=default_config['util_loss'], choices=UTIL_LOSSES,
                       help="Type of Utility loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    utility = parser.add_argument_group('BetaH specific parameters')
    utility.add_argument('-u', '--upsilon', type=float,
                       default=default_config['upsilon'],
                       help="Weight of the utility loss parameter.")
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Eval options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')
    
    modelling = parser.add_argument_group('L&DM modelling specific options')
    modelling.add_argument('--model-epochs', type=int,
                            default=default_config['model_epochs'],
                            help='Number of epochs to train utility prediction model.')
    modelling.add_argument('--pretrain-epochs', type=int,
                            default=default_config['pretrain_epochs'],
                            help='Number of pretraining epochs to train utility prediction model.')
    
    modelling.add_argument('--trial-update', type=str,
                            default=default_config['trial_update'],
                            help='Source for util predictions.')
    
    modelling.add_argument('--cnn_folder', type=str, default="./results/marbles/cnn/",
                         help='Random seed. Can be `None` for stochastic behavior.')

    modelling.add_argument('--feature_labels', type=int,
                            default=default_config['feature_labels'],
                            help='Weight for utility loss in training.')

    modelling.add_argument('--dropout_percent', type=int,
                            default=0,
                            help='Dropout for CNN')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

all_marble_colors = pd.read_csv("./data/marbles/source/colors.csv")
all_marble_colors = all_marble_colors['colors']

stimuli_mean_utilities = []
stimuli_deviations = []
stimuli_marble_values = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))
    stimuli_marble_values.append(marble_values)

def bvae_risk_aversion():
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.bvae_folder)

    X = []
    y = []

    predictive_accuracy = []

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = None
    stimuli_set = 0
    args.img_size = get_img_size(args.dataset)

    setattr(args, 'betaH_B', 1000)

    
    model = load_model(exp_dir + "/set" + str(stimuli_set))
    model.to(device)
    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=logger,
                                    set=stimuli_set)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    
    trainer = Trainer(model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir + "/set" + str(stimuli_set),
                        is_progress_bar=not args.no_progress_bar)

    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    data = pd.DataFrame()

    base_utilities = np.ones(100) * 2
    utilities = np.array(base_utilities) # 0 initially or mean? 
    utilities = torch.from_numpy(utilities.astype(np.float64)).float()
    trainer(train_loader,
            utilities=utilities, 
            epochs=10,
            checkpoint_every=1000000)
    
    for training_step in range(0,10):
        for stimuli1_idx, stimulus1 in enumerate(stimuli):
            for stimuli2_idx, stimulus2 in enumerate(stimuli):
                if(stimuli1_idx == stimuli2_idx): continue

                _ , _, sample_1, pred_util1  = model(stimulus1.unsqueeze(0))
                pred_util1 = pred_util1.item()
                sample_1 = sample_1.detach().numpy()

                _ , _, sample_2, pred_util2 = model(stimulus2.unsqueeze(0))
                pred_util2 = pred_util2.item()
                sample_2 = sample_2.detach().numpy()

                stim1_mean = stimuli_mean_utilities[stimuli1_idx]
                stim1_std = stimuli_deviations[stimuli1_idx]

                stim2_mean = stimuli_mean_utilities[stimuli2_idx]
                stim2_std = stimuli_deviations[stimuli2_idx]

                softmax_utilities = np.array([pred_util1, pred_util2])

                prob_select = np.exp(softmax_utilities * 7) / np.sum(np.exp(softmax_utilities * 7))
                std_diff = round(stim1_std - stim2_std, 2)

                util_parameter = 0.5

                tau = 6

                utility_diff = util_parameter * (np.sqrt(np.sum((pred_util1 - pred_util2)**2)) / 2)
                latent_diff = (1/util_parameter) * (np.sqrt(np.sum((sample_1 - sample_2)**2)) / len(sample_2[0]))

                change_detection_prediction = (latent_diff + utility_diff) / 2
                change_detection = np.array([(1-change_detection_prediction), (0+change_detection_prediction)])
                prob_detect = np.exp(change_detection * tau) / np.sum(np.exp(change_detection * tau))

                util_diff = round(stim1_mean - stim2_mean, 2)

                representation_similarity = (np.sqrt(np.sum((sample_1 - sample_2)**2)))

                #visual_diff = np.sqrt(np.square(np.subtract(stimulus1.detach().numpy(),stimulus2.detach().numpy())).mean())

                #d = {"Visual Difference": visual_diff, "Utility Difference": util_diff}
                d = {"Representation Similarity": representation_similarity, "Probability of Detecting Change": prob_detect[1], "Utility Difference": util_diff, "Training Step": training_step}
                data = data.append(d, ignore_index=True)
        
        #data = data.loc[data['Utility Difference'] > 0]
        #ata = data.loc[data['Utility Difference'] < 1]

        #sns.lineplot(data=data, x="Utility Difference", y="Probability of Detecting Change")
        #sns.regplot(data=data, x="Utility Difference", y="Probability of Detecting Change", scatter=False)

        #plt.show()
        
        #data = data.loc[data["Number of Utility Observations"] == num_utility_obs]
        """utility_differences = data['Utility Difference'].unique()
        y = np.array([])
        for utility_difference in utility_differences:
            diff_data = data.loc[data['Utility Difference'] == utility_difference]
            y = np.append(y, diff_data['Probability of Detecting Change'].mean())
        linRegModel = LinearRegression()
        utility_differences = np.array(utility_differences)
        utility_differences = utility_differences.reshape(-1, 1)
        reg = linRegModel.fit(utility_differences, y)

        ratio = reg.coef_[0]
        print("training_step: ", training_step, " slope ", reg.coef_[0])
        """

        utilities = np.array(stimuli_mean_utilities)
        utilities = torch.from_numpy(utilities.astype(np.float64)).float()
        trainer(train_loader,
                utilities=utilities, 
                epochs=30,
                checkpoint_every=1000000)

        #data = pd.DataFrame()

        data.to_pickle("representationAnalysis_e3.pkl")



    assert(False)
    #sns.lineplot(data=data, x="Visual Difference", y="Utility Difference")
    #plt.show()
    #assert(False)

    #data = data.loc[data['Utility Standard Deviation Difference'] > -0.5]
    #data = data.loc[data['Utility Standard Deviation Difference'] < 0.5]

    data = data.loc[data['Utility Difference'] > 0]
    data = data.loc[data['Utility Difference'] < 1]

    #data.to_pickle("./bvae_risk_aversion.pkl")
    print(data)
    data.to_pickle("./bvae_change_detection_learning.pkl")

    for num_utility_obs in range(1,9):
        data = data.loc[data["Number of Utility Observations"] == num_utility_obs]
        utility_differences = data['Utility Difference'].unique()
        y = np.array([])
        for utility_difference in utility_differences:
            diff_data = data.loc[data['Utility Difference'] == utility_difference]
            y = np.append(y, diff_data['Probability of Detecting Change'].mean())
        model = LinearRegression()
        utility_differences = np.array(utility_differences)
        utility_differences = utility_differences.reshape(-1, 1)
        reg = model.fit(utility_differences, y)

    ratio = reg.coef_[0]
    Slopes = np.append(Slopes, ratio)
    
    sns.lineplot(data=data, x="Utility Difference", y="Probability of Detecting Change")
    sns.regplot(data=data, x="Utility Difference", y="Probability of Detecting Change", scatter=False)

    plt.xlabel('Stimuli Utility Difference', fontsize=14)
    plt.ylabel('Probability of Detecting Change', fontsize=14)
    plt.title("UB-VAE Model Probability of Detecting Change by Stimuli Utility Difference", fontsize=16)
    plt.show()

def main(args):
    predictive_accuracies = []
    inv_temps = []
    all_data = pd.DataFrame()

    bvae_ra = bvae_risk_aversion()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
