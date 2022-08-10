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

from models.vision.cnn import CNN
from models.vision.cnn import Trainer as CNN_Trainer   

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import UTIL_LOSSES, LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS, UTILTIIES 
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining

from torch.utils.data import Dataset, DataLoader

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

import seaborn as sns 

import statsmodels.api as sm

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

def log_loss_score(predicted, actual, eps=1e-14):
        """
        :param predicted:   The predicted probabilities as floats between 0-1
        :param actual:      The binary labels. Either 0 or 1.
        :param eps:         Log(0) is equal to infinity, so we need to offset our predicted values slightly by eps from 0 or 1
        :return:            The logarithmic loss between between the predicted probability assigned to the possible outcomes for item i, and the actual outcome.
        """
        predicted = np.clip(predicted, eps, 1-eps)
        loss = -1 * np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1-predicted))

        return loss

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
    modelling.add_argument('--trial-update', type=str,
                            default=default_config['trial_update'],
                            help='Source for util predictions.')
    
    modelling.add_argument('--bvae_folder', type=str, default="./marbles/bvae/",
                         help="BVAE folder.")

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

folder = './data/marbles/decisions/data2'
all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]
good_participants = ['1088359975_20220708.csv', '1384981370_20220710.csv', '1748395787_20220709.csv', '1832380163_20220710.csv', '1996454642_20220710.csv', '2285081095_20220709.csv', '3072823518_20220709.csv', '3209482804_20220710.csv', '3280341755_20220709.csv', '3437307782_20220709.csv', '3684971091_20220710.csv', '4192753508_20220710.csv', '4617021112_20220709.csv', '4984990593_20220710.csv', '5649795488_20220711.csv', '6261906642_20220709.csv', '6967768109_20220708.csv', '7036685623_20220709.csv', '7361812709_20220709.csv', '7714472260_20220710.csv', '7763967651_20220710.csv', '7781888656_20220709.csv', '8056959514_20220709.csv', '8114269562_20220709.csv', '8214654421_20220710.csv', '8242903913_20220710.csv', '8466633972_20220709.csv', '8473787759_20220709.csv', '8854732576_20220710.csv', '8893453676_20220710.csv', '8988448256_20220710.csv', '9201972787_20220709.csv', '9375774875_20220710.csv', '9553285857_20220709.csv', '9852782779_20220709.csv']
#all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]
#all_participant_data = [participant_data for participant_data in all_participant_data if participant_data not in bad_participants]
all_participant_data = good_participants
print(len(all_participant_data))

def softmax(utilities, tau):
    distribution = np.exp(utilities * tau) / np.sum(np.exp(utilities * tau))
    return distribution

def meu_predictive_accuracy(parameters, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])

    data = data.tail(180)
    
    inv_temp = parameters[0]
    change_temp = parameters[1]

    predictive_accuracy = []
    X = []
    y = []

    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=None,
                                    set=stimuli_set)
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            stim_1 = int(data['stim_1'][ind])
            stim_2 = int(data['stim_2'][ind])

            if(key_press != 'd' and key_press != 'f'):
                continue

            #print("stim 1: ", stim_1, " stim 2: ", stim_2)
            #print("stim 1 mean: ", stimuli_mean_utilities[stim_1], " stim 2: ", stimuli_mean_utilities[stim_2])
            #print("stim 1 mean: ", stimuli_deviations[stim_1], " stim 2: ", stimuli_deviations[stim_2])

            utilties = np.array([stimuli_mean_utilities[stim_1], stimuli_mean_utilities[stim_2]])
            meu_softmax = softmax(utilties, inv_temp)
            X.append(meu_softmax)
            if(key_press == 'd'):
                predictive_accuracy.append(meu_softmax[0])
                y.append([1,0])
            elif(key_press == 'f'):
                predictive_accuracy.append(meu_softmax[1])
                y.append([1,0])
            else:
                assert(False)
        
        if(data['type'][ind] == 0.0):
            key_press = data['key_press'][ind]
            if(key_press != 'k' and key_press != 'j'):
                continue
            
            stim_1 = int(data['stim_1'][ind])
            stim_2 = int(data['stim_2'][ind])

            stim_1_util = stimuli_mean_utilities[stim_1] 
            stim_2_util = stimuli_mean_utilities[stim_2] 
            
            changed = data['changed'][ind]
            change_index = int(data['change_index'][ind])
            changed_stim_idx = int(data['stim_1'][ind]) if change_index == 0 else int(data['stim_2'][ind])
            changed_stim_util = stimuli_mean_utilities[changed_stim_idx]

            first_stim = stimuli[changed_stim_idx].unsqueeze(0).detach().numpy()

            if(changed):
                new_stim_ind = int(data['changed'][ind])
                new_stim = stimuli[change_index].unsqueeze(0)
                new_stim_util = stimuli_mean_utilities[new_stim_ind] 
                
                visual_diff = np.sqrt(np.square(np.subtract(first_stim, new_stim)).mean())
                util_diff = np.sqrt((new_stim_util - changed_stim_util) ** 2) / 4
                diff = (visual_diff + util_diff) / 2
            else: 
                new_stim = np.zeros_like(first_stim)
                visual_diff = np.sqrt(np.square(np.subtract(first_stim, new_stim)).mean())
                util_diff = np.sqrt((0 - changed_stim_util) ** 2) / 4
                diff = (visual_diff + util_diff) / 2

            
            change_detection_prediction = diff
            change_detection = np.array([(1-change_detection_prediction), (0+change_detection_prediction)])
            change_detection_softmax = softmax(change_detection, change_temp)

            if(key_press == 'j'): # predict same 
                predictive_accuracy.append(change_detection_softmax[1])
            elif(key_press == 'k'):
                predictive_accuracy.append(change_detection_softmax[0])

    if(len(predictive_accuracy) == 0): return 0
    return -1 * np.mean(predictive_accuracy)

def cpt_predictive_accuracy(params, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])
    data = data.tail(180)

    inv_temp = params[0]
    change_temp = params[1]
    cpt_scale = params[2]
    
    predictive_accuracy = []
    X = []
    y = []

    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=None,
                                    set=stimuli_set)
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            stim_1 = int(data['stim_1'][ind])
            stim_2 = int(data['stim_2'][ind])

            if(key_press != 'd' and key_press != 'f'):
                continue

            stim_1_util = stimuli_mean_utilities[stim_1] + (cpt_scale * stimuli_deviations[stim_1])
            stim_2_util = stimuli_mean_utilities[stim_2] + (cpt_scale * stimuli_deviations[stim_2])

            #print("updated: ", stim_1_util, stim_2_util)
            #print("original: ", stimuli_mean_utilities[stim_1] , stimuli_mean_utilities[stim_2])

            utilties = np.array([stim_1_util, stim_2_util])
            cpt_softmax = softmax(utilties, inv_temp)
            X.append(cpt_softmax)

            if(key_press == 'd'):
                predictive_accuracy.append(cpt_softmax[0])
                y.append([1,0])
            elif(key_press == 'f'):
                predictive_accuracy.append(cpt_softmax[1])
                y.append([0,1])
            else:
                assert(False)
        
        if(data['type'][ind] == 0.0):
            key_press = data['key_press'][ind]
            if(key_press != 'k' and key_press != 'j'):
                continue

            changed = data['changed'][ind]
            change_index = int(data['change_index'][ind])
            changed_stim_idx = int(data['stim_1'][ind]) if change_index == 0 else int(data['stim_2'][ind])

            stim_1 = int(data['stim_1'][ind])
            stim_2 = int(data['stim_2'][ind])

            stim_1_util = stimuli_mean_utilities[stim_1] + (cpt_scale * stimuli_deviations[stim_1])
            stim_2_util = stimuli_mean_utilities[stim_2] + (cpt_scale * stimuli_deviations[stim_2])

            changed = data['changed'][ind]
            change_index = int(data['change_index'][ind])
            changed_stim_idx = int(data['stim_1'][ind]) if change_index == 0 else int(data['stim_2'][ind])
            changed_stim_util = stimuli_mean_utilities[changed_stim_idx] + (cpt_scale * stimuli_deviations[changed_stim_idx])
            first_stim = stimuli[changed_stim_idx].unsqueeze(0).detach().numpy()

            if(changed):
                new_stim_ind = int(data['changed'][ind])
                new_stim = stimuli[change_index].unsqueeze(0)
                new_stim_util = stimuli_mean_utilities[new_stim_ind] + (cpt_scale * stimuli_deviations[new_stim_ind])
                
                visual_diff = np.sqrt(np.square(np.subtract(first_stim, new_stim)).mean())
                util_diff = np.sqrt((new_stim_util - changed_stim_util) ** 2) / 4
                diff = (visual_diff + util_diff) / 2
            else: 
                new_stim = np.zeros_like(first_stim)
                visual_diff = np.sqrt(np.square(np.subtract(first_stim,new_stim)).mean())
                util_diff = np.sqrt((0 - changed_stim_util) ** 2) / 4
                diff = (visual_diff + util_diff) / 2

            
            change_detection_prediction = diff#.detach().numpy()
            change_detection = np.array([(1-change_detection_prediction), (0+change_detection_prediction)])
            change_detection_softmax = softmax(change_detection, change_temp)

            if(key_press == 'j'): # predict same 
                predictive_accuracy.append(change_detection_softmax[1])
            elif(key_press == 'k'):
                predictive_accuracy.append(change_detection_softmax[0])
    
    if(len(predictive_accuracy) == 0): return 0
    return -1 * np.mean(predictive_accuracy)

def logic_predictive_accuracy(params, participant_data):
    print("Not updated")
    assert(False)
    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])
    data = data.tail(180)

    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=None,
                                    set=stimuli_set)

    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    twos_value = params[0]
    threes_value = params[1]
    fours_value = params[2]
    inv_temp = params[3]
    change_random = params[4]
    change_weight = params[5]

    predictive_accuracy = []
    X = []
    y = []

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            stim_1 = int(data['stim_1'][ind])
            stim_2 = int(data['stim_2'][ind])

            if(key_press != 'd' and key_press != 'f'):
                continue
            
            stim_1_2s = np.sum(stimuli_marble_values[stim_1] == 2)
            stim_2_2s = np.sum(stimuli_marble_values[stim_2] == 2)

            stim_1_3s = np.sum(stimuli_marble_values[stim_1] == 3)
            stim_2_3s = np.sum(stimuli_marble_values[stim_2] == 3)

            stim_1_4s = np.sum(stimuli_marble_values[stim_1] == 4)
            stim_2_4s = np.sum(stimuli_marble_values[stim_2] == 4)

            stim_1_util = (twos_value * stim_1_2s) + (threes_value * stim_1_3s) + (fours_value * stim_1_4s) 
            stim_2_util = (twos_value * stim_2_2s) + (threes_value * stim_2_3s) + (fours_value * stim_2_4s) 

            utilties = np.array([stim_1_util, stim_2_util])
            cpt_softmax = softmax(utilties, inv_temp)
            X.append(cpt_softmax)

            if(key_press == 'd'):
                predictive_accuracy.append(cpt_softmax[0])
                y.append([1,0])
            elif(key_press == 'f'):
                predictive_accuracy.append(cpt_softmax[1])
                y.append([1,0])
            else:
                assert(False)
    
        if(data['type'][ind] == 0.0):
            key_press = data['key_press'][ind]
            if(key_press != 'k' and key_press != 'j'):
                continue

            stim_1 = int(data['stim_1'][ind])
            stim_2 = int(data['stim_2'][ind])

            changed = data['changed'][ind]
            change_index = int(data['change_index'][ind])
            changed_stim_idx = int(data['stim_1'][ind]) if change_index == 0 else int(data['stim_2'][ind])

            stim_1_2s = np.sum(stimuli_marble_values[stim_1] == 2)
            stim_2_2s = np.sum(stimuli_marble_values[stim_2] == 2)

            stim_1_3s = np.sum(stimuli_marble_values[stim_1] == 3)
            stim_2_3s = np.sum(stimuli_marble_values[stim_2] == 3)

            stim_1_4s = np.sum(stimuli_marble_values[stim_1] == 4)
            stim_2_4s = np.sum(stimuli_marble_values[stim_2] == 4)

            stim_1_util = (twos_value * stim_1_2s) + (threes_value * stim_1_3s) + (fours_value * stim_1_4s) 
            stim_2_util = (twos_value * stim_2_2s) + (threes_value * stim_2_3s) + (fours_value * stim_2_4s) 

            util_diff = np.sqrt((stim_1_util - stim_2_util) ** 2)

            first_stim = stimuli[changed_stim_idx].unsqueeze(0)

            if(changed):
                new_stim = stimuli[int(data['new_stim'][ind])].unsqueeze(0)
                visual_diff = np.sqrt(np.square(np.subtract(first_stim,new_stim)).mean())
                diff = (util_diff + visual_diff) / 2
                change_detection_prediction = change_weight * diff
            else: 
                new_stim = np.zeros_like(first_stim)
                visual_diff = np.sqrt(np.square(np.subtract(first_stim,new_stim)).mean())
                diff = (util_diff + visual_diff) / 2
                change_detection_prediction = change_random * diff

            if(key_press == 'j'): # predict same 
                predictive_accuracy.append(1 - change_detection_prediction)
            elif(key_press == 'k'):
                predictive_accuracy.append(change_detection_prediction)

    if(len(predictive_accuracy) == 0): return 0

    return -1 * np.mean(predictive_accuracy)

def bvae_mse(optim_args, participant_data):
    
    inv_temp = optim_args[0]
    change_temp = optim_args[1]
    beta = optim_args[2]

    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.bvae_folder)

    predictive_accuracy = []

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = None

    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])

    data = data.tail(180)
    args.img_size = get_img_size(args.dataset)

    
    model = load_model(exp_dir + "/set" + str(stimuli_set))
    #model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
    model.to(device)
    # set beta and  upsilon
    #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=logger,
                                    set=stimuli_set)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #print("setting beta: ", beta)
    setattr(args, 'betaH_B', beta)
    #setattr(args, 'upsilon', 1e6)
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    
    trainer = Trainer(model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir + "/set" + str(stimuli_set),
                        is_progress_bar=not args.no_progress_bar)
    
    utilities = np.array(stimuli_mean_utilities)
    utilities = torch.from_numpy(utilities.astype(np.float64)).float()
    trainer(train_loader,
            utilities=utilities, 
            epochs=50,
            checkpoint_every=1000000)

    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            if(key_press != 'd' and key_press != 'f'):
                continue 
            
            stim_1 = stimuli[int(data['stim_1'][ind])].unsqueeze(0)
            stim_1_recon , _, _, stim_1_pred_util = model(stim_1)

            stim_2 = stimuli[int(data['stim_2'][ind])].unsqueeze(0)
            stim_2_recon , _, _, stim_2_pred_util = model(stim_2)
        
            pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
            bvae_softmax = np.exp(pred_utils / inv_temp) / np.sum(np.exp(pred_utils / inv_temp), axis=0)

            stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
            stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

            #print("predicted stim util: ", stim_1_pred_util.item(), " true util is: ", stim_1_true_util)

            if(key_press == 'd'):
                predictive_accuracy.append(bvae_softmax[0])
            elif(key_press == 'f'):
                predictive_accuracy.append(bvae_softmax[1])

        if(data['type'][ind] == 0.0):
            key_press = data['key_press'][ind]
            if(key_press != 'k' and key_press != 'j'):
                continue

            changed = data['changed'][ind]
            change_index = int(data['change_index'][ind])
            changed_stim_idx = int(data['stim_1'][ind]) if change_index == 0 else int(data['stim_2'][ind])

            first_stim = stimuli[changed_stim_idx].unsqueeze(0)
            first_stim_recon, first_stim_latent, first_stim_sample, first_stim_pred_util = model(first_stim)

            if(changed):
                new_stim = stimuli[int(data['new_stim'][ind])].unsqueeze(0)
                new_stim_recon, new_stim_latent, new_stim_sample, new_stim_pred_util = model(new_stim)
            else: 
                new_stim_recon, new_stim_latent, new_stim_sample, new_stim_pred_util = model(first_stim_recon)

            first_stim_sample = first_stim_sample.detach().numpy()
            new_stim_sample = new_stim_sample.detach().numpy()

            utility_diff = (np.sqrt(np.sum((first_stim_sample - new_stim_sample)**2)) / 2)
            latent_diff = (np.sqrt(np.sum((first_stim_sample - new_stim_sample)**2)) / len(new_stim_sample))

            change_detection_prediction = (latent_diff + utility_diff) / 2
            change_detection = np.array([(1-change_detection_prediction), (0+change_detection_prediction)])
            change_detection_softmax = softmax(change_detection, change_temp)

            if(key_press == 'j'): # predict same 
                predictive_accuracy.append(change_detection_softmax[1])
            elif(key_press == 'k'):
                predictive_accuracy.append(change_detection_softmax[0])

    return -1 * np.mean(predictive_accuracy)

def cnn_mse(optim_args, participant_data):
    inv_temp = optim_args[0]
    change_temp = optim_args[1]

    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.cnn_folder)

    feature_labels = [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]]

    predictive_accuracy = []

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = None

    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])

    data = data.tail(200)
    args.img_size = get_img_size(args.dataset)

    model = CNN(utility_type=args.utility_type, img_size=args.img_size, latent_dim=2*args.latent_dim,  kwargs=vars(args))
    model.load(args.cnn_folder + "/set" + str(stimuli_set), args)

    #model = load_model(exp_dir + "/set" + str(stimuli_set))
    #model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
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
    
    
    trainer = CNN_Trainer(model, optimizer,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir + "/set" + str(stimuli_set),
                        is_progress_bar=not args.no_progress_bar)
    
    feature_labels = np.array(feature_labels)
    feature_labels = torch.from_numpy(feature_labels).to(device).float()

    utilities = np.array(stimuli_mean_utilities)
    utilities = torch.from_numpy(utilities.astype(np.float64)).float()
    trainer(train_loader,
            utilities=utilities, 
            epochs=50,
            feature_labels=feature_labels,
            checkpoint_every=1000000)
            
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            if(key_press != 'd' and key_press != 'f'):
                continue 
            
            stim_1 = stimuli[int(data['stim_1'][ind])].unsqueeze(0)
            _, stim_1_pred_util, _ = model(stim_1)

            stim_2 = stimuli[int(data['stim_2'][ind])].unsqueeze(0)
            _, stim_2_pred_util, _ = model(stim_2)
        
            pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
            bvae_softmax = np.exp(pred_utils / inv_temp) / np.sum(np.exp(pred_utils / inv_temp), axis=0)

            stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
            stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

            #print("predicted stim util: ", stim_1_pred_util.item(), " true util is: ", stim_1_true_util)

            if(key_press == 'd'):
                predictive_accuracy.append(bvae_softmax[0])
            elif(key_press == 'f'):
                predictive_accuracy.append(bvae_softmax[1])

        if(data['type'][ind] == 0.0):
            key_press = data['key_press'][ind]
            if(key_press != 'k' and key_press != 'j'):
                continue

            changed = data['changed'][ind]
            change_index = int(data['change_index'][ind])
            changed_stim_idx = int(data['stim_1'][ind]) if change_index == 0 else int(data['stim_2'][ind])

            activation = {}

            first_stim = stimuli[changed_stim_idx].unsqueeze(0)
            first_stim_cat, first_stim_pred_util, first_stim_activation = model(first_stim)
            first_stim_activation = first_stim_activation['fcbn1'].cpu().numpy()

            if(changed):
                new_stim = stimuli[int(data['new_stim'][ind])].unsqueeze(0)
                new_stim_cat, new_stim_pred_util, new_stim_activation = model(new_stim)
                new_stim_activation = new_stim_activation['fcbn1'].cpu().numpy()
            else: 
                #new_stim = stimuli[int(data['new_stim'][ind])].unsqueeze(0)
                new_stim = np.zeros_like(first_stim)
                new_stim_cat, new_stim_pred_util = model(new_stim)

            first_stim_sample = first_stim_sample.detach().numpy()
            new_stim_sample = new_stim_sample.detach().numpy()

            utility_diff =  (np.sqrt(np.sum((first_stim_pred_util - new_stim_pred_util)**2)) / 2)
            latent_diff =  (np.sqrt(np.sum((new_stim_activation - first_stim_activation)**2)) / len(new_stim_sample))

            change_detection_prediction = (latent_diff + utility_diff) / 2

            change_detection = np.array([(1-change_detection_prediction), (0+change_detection_prediction)])
            change_detection_softmax = softmax(change_detection, change_temp)

            if(key_press == 'j'): # predict same 
                predictive_accuracy.append(change_detection_softmax[1])
            elif(key_press == 'k'):
                predictive_accuracy.append(change_detection_softmax[0])

    return -1 * np.mean(predictive_accuracy)

def main(args):
    all_data = pd.DataFrame()
    for participant_data in all_participant_data:
    #for participant_data in ['1088359975_20220708.csv']:
        #res = optimize.minimize(meu_predictive_accuracy, (20, 20), args=(participant_data), bounds=((1e-6, 100),(1e-6, 100)), options={"gtol":1e-12})
        #meu_data_accuracy = {"Model": "MEU", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        #all_data = all_data.append(meu_data_accuracy, ignore_index=True)
        
        #res = optimize.minimize(cpt_predictive_accuracy, (20, 20, 1), args=(participant_data), bounds=((1e-6, 100),(1e-6, 100),(1e-6, 100)), options={"gtol":1e-12})
        #cpt_data_accuracy = {"Model": "CPT", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        #all_data = all_data.append(cpt_data_accuracy, ignore_index=True)

        #res = optimize.minimize(logic_predictive_accuracy, (2,3,4,20, 0.5, 0.5), args=(participant_data), bounds=((.1,10),(.1,10),(.1,10),(1e-6, 100),(1e-6,1),(1e-6,1)), options={"gtol":1e-12})
        #logic_data_accuracy = {"Model": "Logic", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        #all_data = all_data.append(logic_data_accuracy, ignore_index=True)
        
        #res = optimize.minimize(bvae_mse, (20, 20, 10), args=(participant_data), bounds=((1e-6, 100),(1e-6, 100),(1e-6, 100)), options={"gtol":1e-12})
        #bvae_data_accuracy = {"Model": "BVAE", "Predictive Accuracy": res['fun']}
        #all_data = all_data.append(bvae_data_accuracy, ignore_index=True)

        #print(cnn_mse((10, 0.5, 0.2), participant_data))
        #assert(False)
        res = optimize.minimize(cnn_mse, (10, 0.5, 0.2), args=(participant_data), bounds=((-100, 100), (1e-6,1), (1e-6,1)), options={"gtol":1e-12})
        bvae_data_accuracy = {"Model": "CNN", "Predictive Accuracy": res['fun']}
        all_data = all_data.append(bvae_data_accuracy, ignore_index=True)

        print("Bvae predictive accuracy: ", -1 * res['fun'])

    print(all_data)
    ax = sns.violinplot(x="Model", y="Predictive Accuracy", data=all_data)
    plt.title("Experiment 2 Model Predictive Accuracy by Participant")
    plt.show()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
