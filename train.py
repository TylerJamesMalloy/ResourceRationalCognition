import argparse
import logging
import sys
import os
import copy 
from configparser import ConfigParser

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

    # Learning options
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

STIMULI_IMAGES = np.load("./data/niv/train.npy").reshape((3,3,3,64,64,3))

def predict_utilities(stimuli, model, train_loader):
    stim_indices = []
    utilities = []
    for stim in stimuli: 
        index = (1 * (stim[2]-1)) + (3 * (stim[1]-1)) + (6 * (stim[0]-1))
        stim_indices.append(index)
    
    for _, data in enumerate(train_loader):
        #print(data)
        data_subset = []
        for stim_index in stim_indices:
            stim = torch.unsqueeze(data[stim_index], 0) 
            _, _, _, util = model(stim)
            utilities.append(util.detach().numpy())
    
    return np.array(utilities)
    assert(False)

    stim1 = np.transpose(STIMULI_IMAGES[stimuli[0][0]-1, stimuli[0][1]-1, stimuli[0][2]-1, :,:,:], [2, 0, 1]) / 255
    stim2 = np.transpose(STIMULI_IMAGES[stimuli[1][0]-1, stimuli[1][1]-1, stimuli[1][2]-1, :,:,:], [2, 0, 1]) / 255
    stim3 = np.transpose(STIMULI_IMAGES[stimuli[2][0]-1, stimuli[2][1]-1, stimuli[2][2]-1, :,:,:], [2, 0, 1]) / 255
    model_input = np.stack((stim1, stim2, stim3))
    im = torch.Tensor(model_input)
    recons, latent_dists, latent_samples, utilities = model(im)

    #im = Image.fromarray(STIMULI_IMAGES[stimuli[0][0]-1, stimuli[0][1]-1, stimuli[0][2]-1, :,:,:].astype(np.uint8))
    #im.show()

    means = utilities[0].detach().numpy()
    vars = np.exp(utilities[1].detach().numpy())

    # reparam trick 
    #std = torch.exp(0.5 * logvar)
    #eps = torch.randn_like(std)
    #return mean + std * eps

    return means, vars

def get_utilities(choice, outcome):
    indices = np.ones((27)) * .5

    if(choice[2] == 1): # red
        indices[0:9] = outcome
    if(choice[2] == 2): # green
        indices[9:18] = outcome
    if(choice[2] == 3): #blue 
        indices[18:27] = outcome
    
    if(choice[1] == 1): # square
        for i in [0,1,2,9,10,11,18,19,20]:
            indices[i] = outcome
    if(choice[1] == 2): # circle
        for i in [3,4,5,12,13,14,21,22,23]:
            indices[i] = outcome
    if(choice[1] == 3): # triangle 
        for i in [6,7,8,15,16,17,24,25,26]:
            indices[i] = outcome
    
    if(choice[0] == 1): # hatched
        for i in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
            indices[i] = outcome
    if(choice[0] == 2): # wave
        for i in [1, 4, 7, 10, 13, 16, 19, 22, 25]:
            indices[i] = outcome
    if(choice[0] == 3): # dotted 
        for i in [2, 5, 8, 11, 14, 17, 20, 23, 26]:
            indices[i] = outcome

    return (torch.from_numpy(indices)).float() 

def get_updated_utilities(features):
    updated_utilities = np.zeros(27)
    utility_maps = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
    updated_utilities = np.array([np.sum((features[utility_map[0]] , features[utility_map[1]], features[utility_map[2]])) for utility_map in utility_maps])
    return (torch.from_numpy(updated_utilities)).float() 

def main(args):
    args.img_size = get_img_size(args.dataset)
    mat = io.loadmat('./data/niv/responses/BehavioralDataOnline.mat')
    """
    Choices - choices (800 trials x 3 features x 22 subjects), NaN for missed trials 
    Outcomes - outcomes, 1 or 0 (800 trials x 22 subjects), NaN for missed trials
    Stimuli - stimuli (800 trials x 3 dimensions x 3 features x 22 subjects; the first dimension is colors, then shapes, then textures)
    RelevantDim - relevant dimension in each trial (800 trials x 22 subjects) 
    CorrectFeature - correct feature in each trial (800 trials x 22 subjects)
    ReactionTimes - reaction times (800 trials x 22 subjects), NaN for missed trials
    """
    data = mat["DimTaskData"][0,0]

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    
    if(not args.is_eval_only):
        # pretrain model or load pretrained on reconstructing stimuli set
        model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
        model = model.to(device)  # make sure trainer and viz on same device
        gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger)
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                            device=device,
                            logger=logger,
                            save_dir=exp_dir,
                            is_progress_bar=not args.no_progress_bar,
                            gif_visualizer=gif_visualizer)
        utilities = torch.from_numpy(np.zeros((27)).astype(np.float64)).float()
        trainer(train_loader,
                utilities=utilities, 
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))
    
    model = load_model(exp_dir, is_gpu=not args.no_cuda)
    model.to(device)

    # retrain and predict 
    ResponseAccuracy = pd.DataFrame()
    rl_lr = 0.122
    sft_inv_temp = 10.33
    decay = 0.466


    for participant_id in range(0,22):
        Choices = data[0][:,:,participant_id]
        Outcomes = data[1][:,participant_id]
        Stimuli = data[2][:,:,:,participant_id]
        RelevantDim = data[3][:,participant_id]
        CorrectFeature = data[4][:,participant_id]
        ReactionTimes = data[5][:,participant_id]

        trial_index = 0
        old_relevant = None 
        old_correct = None 
        base_model = copy.deepcopy(model)
        feature_values = np.zeros(9)

        for trial in range(800):
            choice = Choices[trial]
            if(np.isnan(choice).any()): 
                trial_index += 1
                continue 
            choice = Choices[trial].astype(int)
            outcome = Outcomes[trial]
            stimulus = Stimuli[trial]
            relevant = RelevantDim[trial]
            correct = CorrectFeature[trial]
            reaction = ReactionTimes[trial]

            if( (old_relevant!= None and relevant != old_relevant) or (old_correct != None and old_correct != correct)):
                trial_index = 0
                model = base_model

            old_relevant = relevant
            old_correct = correct
            trial_index += 1

            train_loader = get_dataloaders(args.dataset, batch_size=args.batch_size)

            # predict all utilities based on model 
            utils = predict_utilities(stimulus, model, train_loader)

            for option_index, option in enumerate(stimulus):
                if (option == choice).all: choice_index = option_index

            chosen_probability = np.exp(utils[choice_index] * sft_inv_temp) / np.sum(np.exp(utils * sft_inv_temp))
            #print(chosen_probability)
            
            ResponseAccuracy = ResponseAccuracy.append({"EpisodeTrial": trial_index, "ParticipantId":  participant_id, "Accuracy":chosen_probability[0]}, ignore_index=True)
            utilities = get_utilities(choice, outcome)

            # features (red, green, blue, square, circle, triangle, hatch, wave, dotted)
            color = choice[0] - 1
            shape = choice[1] + 2
            textr = choice[2] + 5
            for feature_index in range(9):
                if(feature_index in [color, shape, textr]):
                    feature_values[feature_index] = feature_values[feature_index] + rl_lr * (outcome - utils[choice_index])
                else:
                    feature_values[feature_index] = (1 - decay) * feature_values[feature_index] 
            
            updated_utilities = get_updated_utilities(feature_values)
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            loss_f = get_loss_f(args.loss,
                                    n_data=len(train_loader.dataset),
                                    device=device,
                                    **vars(args))
            trainer = Trainer(model, optimizer, loss_f,
                                    device=device,
                                    logger=None,
                                    save_dir=exp_dir,
                                    is_progress_bar=False)
            
            trainer(train_loader,
                utilities=updated_utilities, 
                epochs=100, 
                checkpoint_every=1)
            
            utils = predict_utilities(stimulus, model, train_loader)

    print(ResponseAccuracy)
    #ResponseAccuracy.plot(x='EpisodeTrial', y='Accuracy')
    sns.lineplot(data=ResponseAccuracy, x="EpisodeTrial", y="Accuracy")
    plt.show()
    #print(np.mean(all_probabilities, axis=0))

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


