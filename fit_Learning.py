import argparse
from cmath import nan
import enum
from inspect import Parameter
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

import torch 
from torch import optim
import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 

from models.vision.cnn import CNN
from models.vision.cnn import Trainer as CNN_Trainer  


from os import listdir
from os.path import isfile, join

import pandas as pd 
import numpy as np 
import ast 
import matplotlib.pyplot as plt
import scipy.optimize as optimize

import seaborn as sns 

import statsmodels.api as sm

from sklearn.metrics import brier_score_loss

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

def log_loss_score(predicted, actual, eps=1e-14):
        score = 0
        for (pred, act) in zip(predicted, actual):
            score += brier_score_loss(act, pred)
        return score
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
    modelling.add_argument('--bvae_folder', type=str, default='./marbles/bvae/',
                         help="Name of the model for storing and loading purposes.")

    modelling.add_argument('--cnn_folder', type=str, default="./results/marbles/cnn/",
                         help='Random seed. Can be `None` for stochastic behavior.')
    
    modelling.add_argument('--sparse_folder', type=str, default="./results/marbles/sparse/",
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



folder = './data/marbles/learning/data/'
all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]

good_participants = ['1049540426_20220809.csv', '106381449_20220815.csv', '1158286381_20220812.csv', '1204911692_20220816.csv', '125257521_20220809.csv', '1424208338_20220815.csv', '1472936997_20220812.csv', '1480381967_20220815.csv', '1599165604_20220816.csv', '1683383581_20220809.csv', '1789205864_20220815.csv', '1804789041_20220812.csv', '2160484470_20220809.csv', '2269165132_20220809.csv', '2473448161_20220816.csv', '2485825004_20220812.csv', '2551137904_20220812.csv', '2608388359_20220816.csv', '2734423660_20220816.csv', '285880003_20220815.csv', '2975869345_20220812.csv', '3113381684_20220816.csv', '3169950911_20220805.csv', '317608556_20220805.csv', '3549067443_20220815.csv', '3633378519_20220812.csv', '3693949122_20220809.csv', '3758324478_20220816.csv', '376286187_20220809.csv', '3865498490_20220815.csv', '390755978_20220816.csv', '4074885344_20220816.csv', '4081939425_20220816.csv', '4132587080_20220815.csv', '4154479176_20220815.csv', '4258732026_20220812.csv', '440344663_20220813.csv', '4506016898_20220812.csv', '4573741990_20220815.csv', '4647530528_20220815.csv', '4758191482_20220815.csv', '4799514765_20220812.csv', '481425382_20220816.csv', '4819188505_20220812.csv', '4934510627_20220816.csv', '501044799_20220816.csv', '5027782038_20220816.csv', '5176739543_20220812.csv', '5265534006_20220816.csv', '5559758084_20220815.csv', '5892509075_20220816.csv', '5904028522_20220816.csv', '5906483058_20220815.csv', '6074039749_20220812.csv', '6190914712_20220817.csv', '6314725237_20220816.csv', '6499217974_20220816.csv', '6506762788_20220815.csv', '6652616958_20220816.csv', '6764555397_20220812.csv', '6945026478_20220805.csv', '7013814928_20220816.csv', '7125339922_20220815.csv', '7178847280_20220815.csv', '7198253621_20220816.csv', '7211046746_20220809.csv', '7291120861_20220815.csv', '7509475451_20220815.csv', '7711000091_20220816.csv', '7869458961_20220812.csv', '7972392719_20220812.csv', '8483879839_20220815.csv', '8499283501_20220816.csv', '8557939177_20220816.csv', '8759020784_20220812.csv', '9312196920_20220816.csv', '9348576762_20220815.csv', '9547662512_20220816.csv', '9748880425_20220815.csv', '978150698_20220815.csv', '9840397114_20220815.csv']
#good_participants = ['1049540426_20220809.csv', '1158286381_20220812.csv', '125257521_20220809.csv', '1472936997_20220812.csv', '1480381967_20220815.csv', '1683383581_20220809.csv', '1789205864_20220815.csv', '1804789041_20220812.csv', '2160484470_20220809.csv', '2485825004_20220812.csv', '2551137904_20220812.csv', '285880003_20220815.csv', '2975869345_20220812.csv', '3549067443_20220815.csv', '3633378519_20220812.csv', '3758324478_20220816.csv', '376286187_20220809.csv', '3865498490_20220815.csv', '390755978_20220816.csv', '4081939425_20220816.csv', '4154479176_20220815.csv', '440344663_20220813.csv', '4573741990_20220815.csv', '4647530528_20220815.csv', '4758191482_20220815.csv', '4799514765_20220812.csv', '4934510627_20220816.csv', '501044799_20220816.csv', '5027782038_20220816.csv', '5176739543_20220812.csv', '5265534006_20220816.csv', '5559758084_20220815.csv', '5892509075_20220816.csv', '5906483058_20220815.csv', '6074039749_20220812.csv', '6314725237_20220816.csv', '6506762788_20220815.csv', '6652616958_20220816.csv', '6764555397_20220812.csv', '6945026478_20220805.csv', '7198253621_20220816.csv', '7509475451_20220815.csv', '7711000091_20220816.csv', '7972392719_20220812.csv', '8483879839_20220815.csv', '8557939177_20220816.csv', '8759020784_20220812.csv', '9312196920_20220816.csv', '9348576762_20220815.csv', '9547662512_20220816.csv', '9748880425_20220815.csv']
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

def frl_predictive_accuracy(parameters, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])

    all_color_rewards = [
        [45,30,25],
        [45,35,25],
        [50,35,30],
        [50,35,25],
        [50,40,25]
    ]

    dataFrame = pd.DataFrame()


    data = data.tail(200)
    
    inv_temp = parameters[0]
    change_temp = parameters[1]
    learning_rate = parameters[2]

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

    feature_values = np.zeros(3)
    game = 0
    for i, ind in enumerate(data.index):
        if i % 20 == 0: 
            game += 1
            feature_values = np.zeros(3)
            utility_example_ind = 0
        
        color_reward_index = int(data['color_reward_index'][ind])
        color_rewards = all_color_rewards[color_reward_index]

        stimuli_mean_utilities = []
        stimuli_deviations = []
        stimuli_marble_values = []
        stimuli_marble_colors = []
        for marble_colors in all_marble_colors:
            marble_colors = np.array(ast.literal_eval(marble_colors))
            marble_values = np.select([marble_colors == 0, marble_colors == 1, marble_colors == 2], color_rewards, marble_colors)
            stimuli_deviations.append(np.std(marble_values))
            stimuli_mean_utilities.append(np.mean(marble_values))
            stimuli_marble_values.append(marble_values)
            stimuli_marble_colors.append(marble_colors)
        
        stim_1 = int(data['stim_1'][ind])
        stim_2 = int(data['stim_2'][ind])
        new_stim = int(data['new_stim'][ind])

        utilities = []
        for stim in [stim_1, stim_2, new_stim]:
            stim_values = stimuli_marble_colors[stim]
            stim_features = np.array([np.count_nonzero(stim_values == 0), np.count_nonzero(stim_values == 1), np.count_nonzero(stim_values == 2)])
            stim_utility = np.sum(np.multiply(feature_values, stim_features))
            utilities.append(stim_utility / 90) # divide by 9 to get means same size as decision making task 

        meu_softmax = softmax(np.array([utilities[0] , utilities[1]]), inv_temp)

        #stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
        #stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]
        #new_stim_true_util = stimuli_mean_utilities[int(data['new_stim'][ind])]
        #utilities = np.array([stim_1_true_util / 10, stim_2_true_util / 10, new_stim_true_util / 10])
        #meu_softmax = softmax(np.array([stim_1_true_util / 10, stim_2_true_util / 10]), inv_temp)

        if(np.isnan(meu_softmax).any()):
            print(utilities)
            print(meu_softmax)
            print(inv_temp)
            assert(False)

        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            
            reward = int(data['reward'][ind])

            if(key_press != 'd' and key_press != 'f'):
                continue
            
            
            X.append(meu_softmax)
            if(key_press == 'd'):
                predictive_accuracy.append(meu_softmax[0])
                y.append([1,0])
                dataFrame = dataFrame.append({"Accuracy":meu_softmax[0], "Trial": utility_example_ind, "Game": game}, ignore_index=True)
            elif(key_press == 'f'):
                predictive_accuracy.append(meu_softmax[1])
                y.append([0,1])
                dataFrame = dataFrame.append({"Accuracy":meu_softmax[1], "Trial": utility_example_ind, "Game": game}, ignore_index=True)
            else:
                assert(False)
            
            utility_example_ind += 1
            
            # update feature values 
            chosen_stim = stim_1 if key_press == 'd' else stim_2
            chosen_stim_values = stimuli_marble_values[chosen_stim]
            chosen_stim_colors = stimuli_marble_colors[chosen_stim]

            chosen_0s = np.count_nonzero(chosen_stim_colors == 0) # least frequent color
            chosen_1s = np.count_nonzero(chosen_stim_colors == 1) # middle frequent color
            chosen_2s = np.count_nonzero(chosen_stim_colors == 2) # most frequent color

            chosen = [chosen_0s, chosen_1s, chosen_2s]

            true_utility = np.sum(np.multiply(color_rewards, chosen)) 
            feature_reward_diff = ((reward * 9) - true_utility) / 9
            feature_reward_diff = feature_reward_diff / 9

            #print(participant_data)
            #print(" chosen ", chosen, " reward ", reward, " chosen_stim_values ", chosen_stim_values, " color rewards ", color_rewards, " feature values ", feature_values, " ind ", ind, " color_reward_index, ", color_reward_index)
            for feature_index, feature_value in enumerate(feature_values): 
                if(chosen[feature_index] == 0): continue 
                feature_reward = (feature_reward_diff * chosen[feature_index] ) + color_rewards[feature_index]
                feature_values[feature_index] = feature_value + learning_rate * (feature_reward - feature_value)
            
            #print(feature_values)
        
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
            changed_stim_util = utilities[change_index]

            first_stim = stimuli[changed_stim_idx].unsqueeze(0).detach().numpy()

            if(changed):
                new_stim_ind = int(data['changed'][ind])
                new_stim = stimuli[change_index].unsqueeze(0)
                new_stim_util = utilities[2] 
                
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
            
            X.append(change_detection_softmax)
            if(key_press == 'j'): # predict same 
                y.append([1,0])
                predictive_accuracy.append(change_detection_softmax[0])
            elif(key_press == 'k'):
                y.append([0,1])
                predictive_accuracy.append(change_detection_softmax[1])

    
    #if(len(predictive_accuracy) == 0): return 0
    #return -1 * np.mean(predictive_accuracy)
    return log_loss_score(X,y)


def bvae_predictive_accuracy(parameters, participant_data):
    inv_temp = parameters[0]
    change_temp = parameters[1]
    learning_rate = parameters[2]
    beta = parameters[3]

    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.bvae_folder)

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = None

    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])

    all_color_rewards = [
        [45,30,25],
        [45,35,25],
        [50,35,30],
        [50,35,25],
        [50,40,25]
    ]

    dataFrame = pd.DataFrame()
    data = data.tail(200)

    predictive_accuracy = []
    X = []
    y = []

    model = load_model(exp_dir + "/set" + str(stimuli_set))
    #model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
    model.to(device)

    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=None,
                                    set=stimuli_set)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

    base_utilities = np.ones(100) * 2
    utilities = np.array(base_utilities) # 0 initially or mean? 
    utilities = torch.from_numpy(utilities.astype(np.float64)).float()
    trainer(train_loader,
            utilities=utilities, 
            epochs=50,
            checkpoint_every=1000000)

    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    feature_values = np.zeros(3)
    game = 0
    for i, ind in enumerate(data.index):
        if i % 20 == 0: 
            game += 1
            feature_values = np.zeros(3)
            utility_example_ind = 0
        
        color_reward_index = int(data['color_reward_index'][ind])
        color_rewards = all_color_rewards[color_reward_index]

        stimuli_mean_utilities = []
        stimuli_deviations = []
        stimuli_marble_values = []
        stimuli_marble_colors = []
        for marble_colors in all_marble_colors:
            marble_colors = np.array(ast.literal_eval(marble_colors))
            marble_values = np.select([marble_colors == 0, marble_colors == 1, marble_colors == 2], color_rewards, marble_colors)
            stimuli_deviations.append(np.std(marble_values))
            stimuli_mean_utilities.append(np.mean(marble_values))
            stimuli_marble_values.append(marble_values)
            stimuli_marble_colors.append(marble_colors)
        
        stim_1_index = int(data['stim_1'][ind])
        stim_2_index = int(data['stim_2'][ind])
        new_stim_index = int(data['new_stim'][ind])

        utilities = []
        for stim in [stim_1_index, stim_2_index, new_stim_index]:
            stim_values = stimuli_marble_colors[stim]
            stim_features = np.array([np.count_nonzero(stim_values == 0), np.count_nonzero(stim_values == 1), np.count_nonzero(stim_values == 2)])
            stim_utility = np.sum(np.multiply(feature_values, stim_features))
            utilities.append(stim_utility / 90) # divide by 9 to get means same size as decision making task 

        stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
        stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]
        new_stim_true_util = stimuli_mean_utilities[int(data['new_stim'][ind])]

        # first_stim_recon, first_stim_latent, first_stim_sample, first_stim_pred_util
        stim_1 = stimuli[int(data['stim_1'][ind])].unsqueeze(0)
        stim_1_recon , stim_1_latent, stim_1_sample, stim_1_pred_util = model(stim_1)

        stim_2 = stimuli[int(data['stim_2'][ind])].unsqueeze(0)
        stim_2_recon , stim_2_latent, stim_2_sample, stim_2_pred_util = model(stim_2)

        new_stim = stimuli[int(data['new_stim'][ind])].unsqueeze(0)
        new_stim_recon , new_stim_latent, new_stim_sample, new_stim_pred_util = model(new_stim)
    
        pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
        #bvae_softmax = np.exp(pred_utils / inv_temp) / np.sum(np.exp(pred_utils / inv_temp), axis=0)
        bvae_softmax = softmax(pred_utils, inv_temp)

        if(np.isnan(bvae_softmax).any()):
            print(utilities)
            print(bvae_softmax)
            print(inv_temp)
            assert(False)

        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            
            reward = int(data['reward'][ind])

            if(key_press != 'd' and key_press != 'f'):
                continue
            
            
            X.append(bvae_softmax)
            if(key_press == 'd'):
                predictive_accuracy.append(bvae_softmax[0])
                y.append([1,0])
                dataFrame = dataFrame.append({"Accuracy":bvae_softmax[0], "Trial": utility_example_ind, "Game": game}, ignore_index=True)
            elif(key_press == 'f'):
                predictive_accuracy.append(bvae_softmax[1])
                y.append([0,1])
                dataFrame = dataFrame.append({"Accuracy":bvae_softmax[1], "Trial": utility_example_ind, "Game": game}, ignore_index=True)
            else:
                assert(False)
            
            utility_example_ind += 1
            
            # update feature values 
            chosen_stim_index = stim_1_index if key_press == 'd' else stim_2_index
            chosen_stim_values = stimuli_marble_values[chosen_stim_index]
            chosen_stim_colors = stimuli_marble_colors[chosen_stim_index]

            chosen_0s = np.count_nonzero(chosen_stim_colors == 0) # least frequent color
            chosen_1s = np.count_nonzero(chosen_stim_colors == 1) # middle frequent color
            chosen_2s = np.count_nonzero(chosen_stim_colors == 2) # most frequent color

            chosen = [chosen_0s, chosen_1s, chosen_2s]

            true_utility = np.sum(np.multiply(color_rewards, chosen)) 
            feature_reward_diff = ((reward * 9) - true_utility) / 9
            feature_reward_diff = feature_reward_diff / 9

            #print(participant_data)
            #print(" chosen ", chosen, " reward ", reward, " chosen_stim_values ", chosen_stim_values, " color rewards ", color_rewards, " feature values ", feature_values, " ind ", ind, " color_reward_index, ", color_reward_index)
            for feature_index, feature_value in enumerate(feature_values): 
                if(chosen[feature_index] == 0): continue 
                feature_reward = (feature_reward_diff * chosen[feature_index] ) + color_rewards[feature_index]
                #feature_reward = color_rewards[feature_index]
                feature_values[feature_index] = feature_value + learning_rate * (feature_reward - feature_value)

            updated_stimuli_means = []

            for stim in range(100):
                stim_values = stimuli_marble_colors[stim]
                stim_features = np.array([np.count_nonzero(stim_values == 0), np.count_nonzero(stim_values == 1), np.count_nonzero(stim_values == 2)])
                stim_utility = np.sum(np.multiply(feature_values, stim_features))
                updated_stimuli_means.append(stim_utility / 90)

            utilities = np.array(updated_stimuli_means) 
            utilities = torch.from_numpy(utilities.astype(np.float64)).float()
            trainer(train_loader,
                    utilities=utilities, 
                    epochs=args.model_epochs,
                    checkpoint_every=1000000)
        
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
            changed_stim_util = utilities[change_index]

            first_stim = stimuli[changed_stim_idx].unsqueeze(0).detach().numpy()

            if(not changed): continue 

            new_stim_ind = int(data['changed'][ind])
            new_stim = stimuli[change_index].unsqueeze(0)
            new_stim_util = utilities[2] 

            first_stim_sample = stim_1_sample if change_index == 0 else stim_2_sample

            new_stim_sample = new_stim_sample.detach().numpy()
            first_stim_sample = first_stim_sample.detach().numpy()
            
            latent_diff = (np.sqrt(np.sum((first_stim_sample - new_stim_sample)**2)) / len(new_stim_sample[0]))
            #util_diff = np.sqrt((new_stim_util - changed_stim_util) ** 2) / 4
            #diff = (visual_diff + util_diff) / 2

            #change_detection_prediction = diff
            change_detection_prediction = latent_diff
            change_detection = np.array([(1-change_detection_prediction), (0+change_detection_prediction)])
            change_detection_softmax = softmax(change_detection, change_temp)
            
            X.append(change_detection_softmax)
            if(key_press == 'j'): # predict same 
                y.append([1,0])
                predictive_accuracy.append(change_detection_softmax[0])
            elif(key_press == 'k'):
                y.append([0,1])
                predictive_accuracy.append(change_detection_softmax[1])

    
    #if(len(predictive_accuracy) == 0): return 0
    #return -1 * np.mean(predictive_accuracy)
    return log_loss_score(X,y)

def cnn_predictive_accuracy(parameters, participant_data, model_folder, latent_dims):
    inv_temp = parameters[0]
    change_temp = parameters[1]
    learning_rate = parameters[2]
    setattr(args, 'latent_dim', latent_dims)

    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, model_folder)

    feature_labels = [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]]

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = None

    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])

    all_color_rewards = [
        [45,30,25],
        [45,35,25],
        [50,35,30],
        [50,35,25],
        [50,40,25]
    ]

    dataFrame = pd.DataFrame()
    data = data.tail(200)

    predictive_accuracy = []
    X = []
    y = []

    args.img_size = get_img_size(args.dataset)

    model = CNN(utility_type=args.utility_type, img_size=args.img_size, latent_dim=2*args.latent_dim,  kwargs=vars(args))
    model.load(model_folder + "/set" + str(stimuli_set), args)
    model.to(device)

    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=None,
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

    base_utilities = np.ones(100) * 2
    utilities = np.array(base_utilities) # 0 initially or mean? 
    utilities = torch.from_numpy(utilities.astype(np.float64)).float()
    trainer(train_loader,
            utilities=utilities, 
            epochs=50,
            feature_labels=feature_labels,
            checkpoint_every=1000000)

    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 

    feature_values = np.zeros(3)
    game = 0
    for i, ind in enumerate(data.index):
        if i % 20 == 0: 
            game += 1
            feature_values = np.zeros(3)
            utility_example_ind = 0
        
        color_reward_index = int(data['color_reward_index'][ind])
        color_rewards = all_color_rewards[color_reward_index]

        stimuli_mean_utilities = []
        stimuli_deviations = []
        stimuli_marble_values = []
        stimuli_marble_colors = []
        for marble_colors in all_marble_colors:
            marble_colors = np.array(ast.literal_eval(marble_colors))
            marble_values = np.select([marble_colors == 0, marble_colors == 1, marble_colors == 2], color_rewards, marble_colors)
            stimuli_deviations.append(np.std(marble_values))
            stimuli_mean_utilities.append(np.mean(marble_values))
            stimuli_marble_values.append(marble_values)
            stimuli_marble_colors.append(marble_colors)
        
        stim_1_index = int(data['stim_1'][ind])
        stim_2_index = int(data['stim_2'][ind])
        new_stim_index = int(data['new_stim'][ind])

        utilities = []
        for stim in [stim_1_index, stim_2_index, new_stim_index]:
            stim_values = stimuli_marble_colors[stim]
            stim_features = np.array([np.count_nonzero(stim_values == 0), np.count_nonzero(stim_values == 1), np.count_nonzero(stim_values == 2)])
            stim_utility = np.sum(np.multiply(feature_values, stim_features))
            utilities.append(stim_utility / 90) # divide by 9 to get means same size as decision making task 

        stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
        stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]
        new_stim_true_util = stimuli_mean_utilities[int(data['new_stim'][ind])]

        stim_1 = stimuli[int(data['stim_1'][ind])].unsqueeze(0)
        _, stim_1_pred_util, _ = model(stim_1)

        stim_2 = stimuli[int(data['stim_2'][ind])].unsqueeze(0)
        _, stim_2_pred_util, _ = model(stim_2)

        new_stim = stimuli[int(data['new_stim'][ind])].unsqueeze(0)
        _, new_stim_pred_util, _= model(new_stim)
    
        pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
        #bvae_softmax = np.exp(pred_utils / inv_temp) / np.sum(np.exp(pred_utils / inv_temp), axis=0)
        cnn_softmax = softmax(pred_utils, inv_temp)

        if(np.isnan(cnn_softmax).any()):
            print(utilities)
            print(cnn_softmax)
            print(inv_temp)
            assert(False)

        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            
            reward = int(data['reward'][ind])

            if(key_press != 'd' and key_press != 'f'):
                continue
            
            
            X.append(cnn_softmax)
            if(key_press == 'd'):
                predictive_accuracy.append(cnn_softmax[0])
                y.append([1,0])
                dataFrame = dataFrame.append({"Predictive Accuracy":cnn_softmax[0], "Trial": utility_example_ind, "Game": game, "Type": "utility", "Participant": participant_data, "Model": model_folder}, ignore_index=True)
            elif(key_press == 'f'):
                predictive_accuracy.append(cnn_softmax[1])
                y.append([0,1])
                dataFrame = dataFrame.append({"Predictive Accuracy":cnn_softmax[1], "Trial": utility_example_ind, "Game": game, "Type": "utility", "Participant": participant_data, "Model": model_folder}, ignore_index=True)
            else:
                assert(False)
            
            utility_example_ind += 1
            
            # update feature values 
            chosen_stim_index = stim_1_index if key_press == 'd' else stim_2_index
            chosen_stim_values = stimuli_marble_values[chosen_stim_index]
            chosen_stim_colors = stimuli_marble_colors[chosen_stim_index]

            chosen_0s = np.count_nonzero(chosen_stim_colors == 0) # least frequent color
            chosen_1s = np.count_nonzero(chosen_stim_colors == 1) # middle frequent color
            chosen_2s = np.count_nonzero(chosen_stim_colors == 2) # most frequent color

            chosen = [chosen_0s, chosen_1s, chosen_2s]

            true_utility = np.sum(np.multiply(color_rewards, chosen)) 
            feature_reward_diff = ((reward * 9) - true_utility) / 9
            feature_reward_diff = feature_reward_diff / 9

            #print(participant_data)
            #print(" chosen ", chosen, " reward ", reward, " chosen_stim_values ", chosen_stim_values, " color rewards ", color_rewards, " feature values ", feature_values, " ind ", ind, " color_reward_index, ", color_reward_index)
            for feature_index, feature_value in enumerate(feature_values): 
                if(chosen[feature_index] == 0): continue 
                feature_reward = (feature_reward_diff * chosen[feature_index] ) + color_rewards[feature_index]
                #feature_reward = color_rewards[feature_index]
                feature_values[feature_index] = feature_value + learning_rate * (feature_reward - feature_value)

            updated_stimuli_means = []

            for stim in range(100):
                stim_values = stimuli_marble_colors[stim]
                stim_features = np.array([np.count_nonzero(stim_values == 0), np.count_nonzero(stim_values == 1), np.count_nonzero(stim_values == 2)])
                stim_utility = np.sum(np.multiply(feature_values, stim_features))
                updated_stimuli_means.append(stim_utility / 90)

            utilities = np.array(updated_stimuli_means) # 0 initially or mean? 
            utilities = torch.from_numpy(utilities.astype(np.float64)).float()
            trainer(train_loader,
                    utilities=utilities, 
                    epochs=args.model_epochs,
                    feature_labels=feature_labels,
                    checkpoint_every=1000000)
        
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
            else: 
                new_stim = torch.from_numpy(np.zeros_like(first_stim)).float()
            
            new_stim_cat, new_stim_pred_util, new_stim_activation = model(new_stim)
            new_stim_activation = new_stim_activation['fcbn1'].cpu().numpy()

            new_stim_cat = new_stim_cat.detach().numpy()
            first_stim_cat = first_stim_cat.detach().numpy()

            first_stim_pred_util = first_stim_pred_util.detach().numpy()
            new_stim_pred_util = new_stim_pred_util.detach().numpy()

            utility_diff =  (np.sqrt((first_stim_pred_util - new_stim_pred_util)**2)) / 2
            latent_diff =  (np.sqrt(np.sum((new_stim_activation - first_stim_activation)**2)) / len(new_stim_activation))
            cat_diff = (np.sqrt(np.sum((new_stim_cat - first_stim_cat)**2)) / len(new_stim_activation))

            #change_detection_prediction = (latent_diff + utility_diff) / 2
            change_detection_prediction = latent_diff
            #change_detection_prediction = change_detection_prediction[0]

            change_detection = np.array([(1-change_detection_prediction), (0+change_detection_prediction)])
            change_detection_softmax = softmax(change_detection, change_temp)

            X.append(change_detection_softmax)

            if(key_press == 'j'): # predict same 
                y.append([1,0])
                predictive_accuracy.append(change_detection_softmax[0])
                dataFrame = dataFrame.append({"Predictive Accuracy":change_detection_softmax[1], "Trial": utility_example_ind, "Game": game, "Type": "change", "Participant": participant_data, "Model": model_folder}, ignore_index=True)
            elif(key_press == 'k'):
                y.append([0,1])
                predictive_accuracy.append(change_detection_softmax[1])
                dataFrame = dataFrame.append({"Predictive Accuracy":change_detection_softmax[0], "Trial": utility_example_ind, "Game": game, "Type": "change", "Participant": participant_data, "Model": model_folder}, ignore_index=True)

    #return dataFrame
    #return -1 * np.mean(predictive_accuracy)
    return log_loss_score(X,y)


def accuracy(participant_data):
    data = pd.read_csv(join(folder, participant_data))  

    all_color_rewards = [
        [45,30,25],
        [45,35,25],
        [50,35,30],
        [50,35,25],
        [50,40,25]
    ]

    dataFrame = pd.DataFrame()


    data = data.tail(200)
    game = 0
    utility_example_ind = 0
    
    for i, ind in enumerate(data.index):
        if i % 20 == 0: 
            game += 1
            feature_values = np.zeros(3)
            utility_example_ind = 0

        color_reward_index = int(data['color_reward_index'][ind])
        color_rewards = all_color_rewards[color_reward_index]

        stimuli_mean_utilities = []
        stimuli_deviations = []
        stimuli_marble_values = []
        stimuli_marble_colors = []
        for marble_colors in all_marble_colors:
            marble_colors = np.array(ast.literal_eval(marble_colors))
            marble_values = np.select([marble_colors == 0, marble_colors == 1, marble_colors == 2], color_rewards, marble_colors)
            stimuli_deviations.append(np.std(marble_values))
            stimuli_mean_utilities.append(np.mean(marble_values))
            stimuli_marble_values.append(marble_values)
            stimuli_marble_colors.append(marble_colors)
        
        stim_1 = int(data['stim_1'][ind])
        stim_2 = int(data['stim_2'][ind])

        utilities = []
        for stim in [stim_1, stim_2]:
            stim_values = stimuli_marble_colors[stim]
            stim_features = np.array([np.count_nonzero(stim_values == 0), np.count_nonzero(stim_values == 1), np.count_nonzero(stim_values == 2)])
            stim_utility = np.sum(np.multiply(feature_values, stim_features))
            utilities.append(stim_utility / 90) # divide by 9 to get means

        stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
        stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]
        #utilities = [stim_1_true_util, stim_2_true_util]
        reward = int(data['reward'][ind])

        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]

            if(key_press != 'd' and key_press != 'f'):
                continue
            
            if(key_press == 'd'):
                correct = 1 if stim_1_true_util >= stim_2_true_util else 0
                dataFrame = dataFrame.append({"Accuracy":correct, "Reward":reward, "Trial": utility_example_ind, "Game": game}, ignore_index=True)
            elif(key_press == 'f'):
                correct = 0 if stim_1_true_util >= stim_2_true_util else 1
                dataFrame = dataFrame.append({"Accuracy":correct, "Reward":reward, "Trial": utility_example_ind, "Game": game}, ignore_index=True)
            else:
                assert(False)
        
            utility_example_ind += 1

    return dataFrame  

def main(args):
    all_data = pd.DataFrame()
    #for participant_data in all_participant_data:
    for participant_data in good_participants:
        res = optimize.minimize(frl_predictive_accuracy, (5, 5, 0.8), args=(participant_data), bounds=((1e-1, 50), (1e-1, 50), (1e-3,1)), options={"gtol":1e-12})
        data_accuracy = {"Model": "FRL", "Log Loss": res['fun'], "Params": res['x']}
        all_data = all_data.append(data_accuracy, ignore_index=True)

        """print(" frl_predictive_accuracy: ", res['fun'])

        res = optimize.minimize(bvae_predictive_accuracy, (5, 5, 0.8, 10), args=(participant_data), bounds=((1e-1, 50), (1e-1, 50), (1e-3,1), (1e-3, 100)), options={"gtol":1e-12})
        data_accuracy = {"Model": "BVAE", "Log Loss": res['fun'], "Params": res['x']}
        all_data = all_data.append(data_accuracy, ignore_index=True)

        print(" bvae_predictive_accuracy: ", res['fun'])

        res = optimize.minimize(cnn_predictive_accuracy, (5, 5, 0.8), args=(participant_data, args.cnn_folder, 128), bounds=((1e-1, 50), (1e-1, 50), (1e-3,1)), options={"gtol":1e-12})
        data_accuracy = {"Model": "CNN", "Log Loss": res['fun'], "Params": res['x']}
        all_data = all_data.append(data_accuracy, ignore_index=True)

        print(" cnn_predictive_accuracy: ", res['fun'])

        res = optimize.minimize(cnn_predictive_accuracy, (5, 5, 0.8), args=(participant_data, args.sparse_folder, 9), bounds=((1e-1, 50), (1e-1, 50), (1e-3,1)), options={"gtol":1e-12})
        data_accuracy = {"Model": "Sparse", "Log Loss": res['fun'], "Params": res['x']}
        all_data = all_data.append(data_accuracy, ignore_index=True)

        print(" sparse_predictive_accuracy: ", res['fun'])"""

    #print(all_data)
    #sns.lineplot(data=all_data, x="Trial", y="Reward")
    #plt.title("Experiment 3 Participant Utility Observed by Block Trial")
    #plt.show()

    all_data.to_pickle("./learningParameters.pkl")

    print(all_data)
    ax = sns.barplot(x="Model", y="Predictive Accuracy", data=all_data)
    plt.show()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
