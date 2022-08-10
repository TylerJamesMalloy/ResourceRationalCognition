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
    general.add_argument('-name', type=str, default='',
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
    modelling.add_argument('--trial-update', type=str,
                            default=default_config['trial_update'],
                            help='Source for util predictions.')

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



folder = './data/marbles/learning/data'
all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]

def softmax(utilities, tau):
    if(np.sum(utilities) == 0): return [0.5,0.5]
    distribution = np.exp(utilities * tau) / np.sum(np.exp(utilities * tau))
    return distribution

def frl_mse(parameters, participant_data):
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

        stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
        stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]
        new_stim_true_util = stimuli_mean_utilities[int(data['new_stim'][ind])]
        #utilities = [stim_1_true_util, stim_2_true_util]

        #utilities = np.array(utilities)
        utilities = np.array([stim_1_true_util / 10, stim_2_true_util / 10, new_stim_true_util / 10])
        meu_softmax = softmax(utilities, inv_temp)

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
                y.append([1,0])
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
                #feature_reward = color_rewards[feature_index]
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

            if(key_press == 'j'): # predict same 
                predictive_accuracy.append(change_detection_softmax[1])
            elif(key_press == 'k'):
                predictive_accuracy.append(change_detection_softmax[0])

    
    if(len(predictive_accuracy) == 0): return 0
    return -1 * np.mean(predictive_accuracy)


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


    data = data.tail(100)
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
    for participant_data in ["6945026478_20220805.csv", "3169950911_20220805.csv", "317608556_20220805.csv", "5580217541_20220805.csv", "1049540426_20220809.csv", "125257521_20220809.csv", "1683383581_20220809.csv", "376286187_20220809.csv"]:
        
        #accuracyData = accuracy(participant_data)
        #all_data = all_data.append(accuracyData, ignore_index=True)
        #print(-1 * frl_mse((10, 0.2, 0.2, 0.8), participant_data))
        res = optimize.minimize(frl_mse, (20, 20, 0.8), args=(participant_data), bounds=((1e-6, 100), (1e-6, 100), (1e-6,1)), options={"gtol":1e-12})
        #res = optimize.minimize(frl_mse, (10, 0.8), args=(participant_data), bounds=((1e-6, 100), (1e-6,1)), options={"gtol":1e-12})
        frl_data_accuracy = {"Model": "FRL", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        all_data = all_data.append(frl_data_accuracy, ignore_index=True)

        #print("FRL predictive accuracy: ", -1 * res['fun'])

    #print(all_data)
    #sns.lineplot(data=all_data, x="Trial", y="Reward")
    #plt.title("Experiment 3 Participant Utility Observed by Block Trial")
    #plt.show()

    print(all_data)
    ax = sns.violinplot(x="Model", y="Predictive Accuracy", data=all_data)
    plt.show()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
