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

import math, random 
from models.learning.frl import Feature_RL, frl_env

from os import listdir
from os.path import isfile, join

from sklearn.metrics import log_loss

import ast 

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]


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
    general.add_argument('name', type=str,
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
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))

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

def softmax(utilities, inv_temp=10):
    print(utilities)
    soft_percentages = []
    for utility in utilities: 
        soft_percentages.append(np.exp(utility * inv_temp) / np.sum(np.exp(utilities * inv_temp), axis=0))
    
    return soft_percentages


def main(args):
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    all_participant_data = [f for f in listdir('./data/marbles/decisions/data/') if isfile(join('./data/marbles/decisions/data/', f))]
    args.img_size = get_img_size(args.dataset)

    models = []
    if(not args.is_eval_only):
        for stimuli_set in [0,1,2,3,4,5]:

            if(not os.path.exists(exp_dir + "/set" + str(stimuli_set))):
                os.makedirs(exp_dir + "/set" + str(stimuli_set))

            model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
            model = model.to(device)  # make sure trainer and viz on same device
            #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
            train_loader = get_dataloaders(args.dataset,
                                            batch_size=args.batch_size,
                                            logger=logger,
                                            set=stimuli_set)
            
            logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
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
            
            #utilities = np.array(stimuli_mean_utilities)
            #utilities = np.ones_like(stimuli_mean_utilities) * np.mean(stimuli_mean_utilities) #2.5733
            utilities = np.zeros_like(stimuli_mean_utilities)
            utilities = torch.from_numpy(utilities.astype(np.float64)).float()
            trainer(train_loader,
                    utilities=utilities, 
                    epochs=args.epochs,
                    checkpoint_every=args.checkpoint_every,)
            
            models.append(model)

            # SAVE MODEL AND EXPERIMENT INFORMATION
            save_model(trainer.model, exp_dir + "/set" + str(stimuli_set), metadata=vars(args))
    
    
    """models = []
    for stimuli_set in [0,1,2,3,4,5]:
        model = load_model(exp_dir + "/set" + str(stimuli_set))
        model.to(device)

        #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger,
                                        set=stimuli_set)
        
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
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
        
        # fit utilities to participant? 
        # LOO based? 
        utilities = np.array(stimuli_mean_utilities)
        utilities = torch.from_numpy(utilities.astype(np.float64)).float()
        trainer(train_loader,
                utilities=utilities, 
                epochs=1,
                checkpoint_every=args.checkpoint_every,)

        models.append(model)
    
    participant_risk_aversions = risk_avers_parameters
    predictive_accuracy = pd.DataFrame()
    
    for participant_idx, participant_data in enumerate(all_participant_data):
        bvae_participant_accuracies = []
        meu_participant_accuracies = []
        cpt_participant_accuracies = []
        
        inv_temp = inv_temp_parameters[participant_idx]

        #if(participant_idx == 0): continue
        data = pd.read_csv(join('./data/marbles/decisions/data/', participant_data))  
        marble_set = int(data['marble_set'][0])

        participant_model = models[marble_set]

        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger,
                                        set=str(marble_set))

        stimuli = None 
        for _, stimuli in enumerate(train_loader):
            stimuli = stimuli 

        all_bvae_nll = []
        all_meu_nll = []
        all_cpt_nll = []

        #print(data.head)
        for ind in data.index:
            if(data['type'][ind] == 1.0):
                stim_1 = stimuli[int(data['stim_1'][ind])].unsqueeze(0)
                stim_1_recon , _, _, stim_1_pred_util = participant_model(stim_1)

                stim_2 = stimuli[int(data['stim_2'][ind])].unsqueeze(0)
                stim_2_recon , _, _, stim_2_pred_util = participant_model(stim_2)

                #print(stim_1_util.item(), stim_2_util.item())
                key_press = data['key_press'][ind]
                if(key_press != 'd' and key_press != 'f'):
                    break 

                pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
                bvae_softmax = np.exp(pred_utils * inv_temp) / np.sum(np.exp(pred_utils * inv_temp), axis=0)

                stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
                stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

                true_utils = np.array([stim_1_true_util, stim_2_true_util])
                meu_softmax = np.exp(true_utils * inv_temp) / np.sum(np.exp(true_utils * inv_temp), axis=0)
                
                risk_aversion = participant_risk_aversions[participant_idx]
                stim_1_deviation = stimuli_deviations[int(data['stim_1'][ind])]
                stim_2_deviation = stimuli_deviations[int(data['stim_2'][ind])]

                ra_stim_1 = stim_1_true_util
                ra_stim_2 = stim_2_true_util

                if(key_press == 'd'):
                    sd_diff = (stim_1_deviation - stim_2_deviation)
                    ra_stim_1 = (sd_diff * risk_aversion[0]) + risk_aversion[1]
                else:
                    sd_diff = (stim_2_deviation - stim_1_deviation)
                    ra_stim_2 = (sd_diff * risk_aversion[0]) + risk_aversion[1]

                cpt_inv_temp = cpt_temp_params[participant_idx]
                cpt_utils = np.array([ra_stim_1, ra_stim_2])
                cpt_softmax = np.exp(cpt_utils * cpt_inv_temp) / np.sum(np.exp(cpt_utils * cpt_inv_temp), axis=0)

                if(key_press == 'd'):
                    y = np.array([1,0])
                else:
                    y = np.array([0,1])

                bvae_softmax = np.array(bvae_softmax)
                meu_softmax = np.array(meu_softmax)
                cpt_softmax = np.array(cpt_softmax)
                import scipy 
                #bvae_mse = (np.square(bvae_softmax - y)).mean() #** -2
                #meu_mse = (np.square(meu_softmax - y)).mean()  #** -2
                #cpt_mse = (np.square(cpt_softmax - y)).mean() #** -2
                bvae_nll = log_loss(y,bvae_softmax )
                meu_nll = log_loss(y,meu_softmax)
                cpt_nll = log_loss(y,cpt_softmax)

                all_bvae_nll.append(bvae_nll)
                all_meu_nll.append(meu_nll)
                all_cpt_nll.append(cpt_nll)

        bvae_nlls = np.mean(all_bvae_nll)
        meu_nlls = np.mean(all_meu_nll)
        cpt_nlls = np.mean(all_cpt_nll)

        #assert(False)
        predictive_accuracy = predictive_accuracy.append({"Model Type":"B-BVAE", "Negative Log Loss":bvae_nlls, "Inverse Temperature":inv_temp, "Participant":participant_idx}, ignore_index=True)
        predictive_accuracy = predictive_accuracy.append({"Model Type":"Soft-Max", "Negative Log Loss":meu_nlls, "Inverse Temperature":inv_temp, "Participant":participant_idx}, ignore_index=True)
        predictive_accuracy = predictive_accuracy.append({"Model Type":"Risk Averse", "Negative Log Loss":cpt_nlls, "Inverse Temperature":inv_temp, "Participant":participant_idx}, ignore_index=True)
        
        bvae_participant_accuracies.append(bvae_nlls)
        meu_participant_accuracies.append(meu_nlls)
        cpt_participant_accuracies.append(cpt_nlls)

        bvae_participant_accuracies = np.array(bvae_participant_accuracies)
        #print(betas[np.argmin(bvae_participant_accuracies)], ",")

    fig, axis = plt.subplots(1, 1, sharex=True, sharey=True)
    #print(predictive_accuracy)
    #p = sns.lineplot(data=predictive_accuracy, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[1])
    #p = sns.lineplot(data=predictive_accuracy, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axis)
    p = sns.barplot(data=predictive_accuracy, x="Model Type", y="Negative Log Loss")
    p.set_title("All Trials Negative Log Loss")

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #print(predictive_accuracy)
    correct_trials = predictive_accuracy.loc[predictive_accuracy["Correct"] == True]
    incorrect_trials = predictive_accuracy.loc[predictive_accuracy["Correct"] == False]

    #p = sns.lineplot(data=predictive_accuracy, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[1])
    p = sns.lineplot(data=correct_trials, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[0])
    p.set_title("Correct Trials Predictive Accuracy")
    p = sns.lineplot(data=incorrect_trials, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[1])
    p.set_title("Incorrect Trials Predictive Accuracy")

    plt.show()

    print(predictive_accuracy)
    """

    return 

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


