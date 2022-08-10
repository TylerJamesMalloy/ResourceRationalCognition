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

stimuli_mean_utilities = []
stimuli_deviations = []
stimuli_marble_values = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))
    stimuli_marble_values.append(marble_values)

folder = './data/marbles/decisions/data'
#good_participants = ['1206472879_20220607.csv', '13896051_20220623.csv', '1481432285_20220607.csv', '1571956186_20220623.csv', '1655417347_20220607.csv', '1670585779_20220607.csv', '169075273_20220607.csv', '1917494030_20220701.csv', '2024508277_20220622.csv', '2307452822_20220602.csv', '2429104214_20220607.csv', '2467485070_20220701.csv', '2616009388_20220607.csv', '2669911017_20220607.csv', '282458648_20220622.csv', '2969042685_20220622.csv', '3010097611_20220622.csv', '3016456232_20220622.csv', '302208162_20220607.csv', '3050445599_20220701.csv', '3072452500_20220623.csv', '3231145639_20220622.csv', '3310926888_20220622.csv', '3437000926_20220623.csv', '3453896152_20220518.csv', '3525719084_20220607.csv', '3545302544_20220623.csv', '3627110067_20220623.csv', '3709436102_20220622.csv', '3774486973_20220702.csv', '3777558888_20220602.csv', '3868544605_20220622.csv', '424296399_20220622.csv', '4302545825_20220607.csv', '4309499885_20220624.csv', '4424042522_20220623.csv', '4522484535_20220602.csv', '4604314752_20220625.csv', '4717805082_20220622.csv', '4737559307_20220623.csv', '4758284626_20220607.csv', '4784817211_20220623.csv', '4786413128_20220623.csv', '4833293935_20220607.csv', '5138618294_20220607.csv', '5144493038_20220602.csv', '5347559166_20220701.csv', '5534437613_20220701.csv', '5552993317_20220602.csv', '5878990705_20220607.csv', '5979681843_20220623.csv', '6130248011_20220622.csv', '6174180168_20220602.csv', '6176365135_20220602.csv', '6247410167_20220607.csv', '6737332423_20220607.csv', '6745644970_20220623.csv', '682320948_20220701.csv', '685851185_20220701.csv', '6948208686_20220602.csv', '6969137467_20220622.csv', '7043291063_20220622.csv', '7056217438_20220622.csv', '7075207841_20220622.csv', '7243344067_20220701.csv', '7351329913_20220701.csv', '748797646_20220518.csv', '7489651562_20220701.csv', '7560795788_20220624.csv', '7708514735_20220701.csv', '7729591288_20220607.csv', '7811512263_20220623.csv', '7839131207_20220623.csv', '7916373955_20220622.csv', '8198410857_20220622.csv', '8254485902_20220623.csv', '8488980532_20220602.csv', '851978686_20220701.csv', '8762245299_20220622.csv', '8880742555_20220623.csv', '8894686670_20220622.csv', '9023291296_20220602.csv', '908333355_20220623.csv', '908986223_20220622.csv', '9162481065_20220607.csv', '9177013872_20220518.csv', '9195187466_20220607.csv', '9262283903_20220623.csv', '934906418_20220623.csv', '9410194125_20220623.csv', '9412783563_20220607.csv', '9796178986_20220623.csv', '9824877929_20220518.csv']
good_participants = ['1206472879_20220607.csv', '1481432285_20220607.csv', '1571956186_20220623.csv', '1655417347_20220607.csv', '169075273_20220607.csv', '2307452822_20220602.csv', '2429104214_20220607.csv', '2669911017_20220607.csv', '282458648_20220622.csv', '2969042685_20220622.csv', '3010097611_20220622.csv', '302208162_20220607.csv', '3050445599_20220701.csv', '3072452500_20220623.csv', '3310926888_20220622.csv', '3437000926_20220623.csv', '3525719084_20220607.csv', '3545302544_20220623.csv', '3709436102_20220622.csv', '3774486973_20220702.csv', '3777558888_20220602.csv', '4302545825_20220607.csv', '4309499885_20220624.csv', '4424042522_20220623.csv', '4604314752_20220625.csv', '4717805082_20220622.csv', '4784817211_20220623.csv', '4833293935_20220607.csv', '5138618294_20220607.csv', '5144493038_20220602.csv', '5534437613_20220701.csv', '5878990705_20220607.csv', '6130248011_20220622.csv', '6174180168_20220602.csv', '6176365135_20220602.csv', '6247410167_20220607.csv', '682320948_20220701.csv', '685851185_20220701.csv', '6969137467_20220622.csv', '7056217438_20220622.csv', '748797646_20220518.csv', '7489651562_20220701.csv', '7708514735_20220701.csv', '7729591288_20220607.csv', '7811512263_20220623.csv', '8198410857_20220622.csv', '8254485902_20220623.csv', '8488980532_20220602.csv', '8762245299_20220622.csv', '8880742555_20220623.csv', '8894686670_20220622.csv', '908986223_20220622.csv', '9162481065_20220607.csv', '9177013872_20220518.csv', '9195187466_20220607.csv', '9262283903_20220623.csv', '934906418_20220623.csv', '9412783563_20220607.csv', '9796178986_20220623.csv', '9824877929_20220518.csv']
#all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]
#all_participant_data = [participant_data for participant_data in all_participant_data if participant_data not in bad_participants]
all_participant_data = good_participants

def softmax(utilities, tau):
    distribution = np.exp(utilities * tau) / np.sum(np.exp(utilities * tau))
    return distribution

def meu_predictive_accuracy(inv_temp, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)

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

    if(len(predictive_accuracy) == 0): return 0
    return -1 * np.mean(predictive_accuracy)

def cpt_predictive_accuracy(params, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)

    inv_temp = params[0]
    cpt_scale = params[1]

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
    
    if(len(predictive_accuracy) == 0): return 0
    return -1 * np.mean(predictive_accuracy)

def logic_predictive_accuracy(params, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)

    twos_value = params[0]
    threes_value = params[1]
    fours_value = params[2]
    inv_temp = params[3]

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
    
    if(len(predictive_accuracy) == 0): return 0

    return -1 * np.mean(predictive_accuracy)

def bvae_mse(optim_args, participant_data):
    beta = optim_args[0]
    inv_temp = optim_args[1]

    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)

    predictive_accuracy = []

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = None

    data = pd.read_csv(join(folder, participant_data))  
    stimuli_set = int(data['marble_set'][0])

    data = data.tail(200)
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
            else:
                assert(False)

    return -1 * np.mean(predictive_accuracy)



def main(args):
    predictive_accuracies = []
    inv_temps = []
    all_data = pd.DataFrame()
    for participant_data in all_participant_data:
    #for participant_data in ['1206472879_20220607.csv']:
        #print(meu_predictive_accuracy(5, participant_data))
        #assert(False)
        
        res = optimize.minimize(meu_predictive_accuracy, (20), args=(participant_data), bounds=((1e-6, 100),), options={"gtol":1e-12})
        meu_data_accuracy = {"Model": "MEU", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        all_data = all_data.append(meu_data_accuracy, ignore_index=True)

        res = optimize.minimize(cpt_predictive_accuracy, (20,1), args=(participant_data), bounds=((1e-6, 100),(1e-3, 3)), options={"gtol":1e-12})
        cpt_data_accuracy = {"Model": "CPT", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        all_data = all_data.append(cpt_data_accuracy, ignore_index=True)

        res = optimize.minimize(logic_predictive_accuracy, (2,3,4,20), args=(participant_data), bounds=((.1,10),(.1,10),(.1,10),(1e-6, 100)), options={"gtol":1e-12})
        meu_data_accuracy = {"Model": "Logic", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        all_data = all_data.append(meu_data_accuracy, ignore_index=True)
        
        res = optimize.minimize(bvae_mse, (10, 10), args=(participant_data), bounds=((0, 1000),(-100, 100)), options={"gtol":1e-12})
        meu_data_accuracy = {"Model": "BVAE", "Predictive Accuracy":-1 * res['fun'], "Params": res['x']}
        all_data = all_data.append(meu_data_accuracy, ignore_index=True)

        #inv_temps.append(res['x'][0])
        #predictive_accuracies.append(-1 * res['fun'])

    print(all_data)
    all_data.to_csv("./fit.pd")
    ax = sns.violinplot(x="Model", y="Predictive Accuracy", data=all_data)
    plt.title("Model Predictive Accuracy")
    plt.show()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
