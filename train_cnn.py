import argparse
import logging
import sys
import os
import copy 
from configparser import ConfigParser

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models.vision.cnn import CNN, Trainer  
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

import math 
from models.learning.frl import Feature_RL, frl_env

UTILTIIES = ["Malloy"]
UTIL_LOSSES = ["MSE", "L1"]
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}".format(data)
                                for data in DATASETS]
FEATURE_MAP = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]

def predict_utilities(stimuli, model, train_loader, inv_temp):
    stim_indices = []
    utilities = []
    for stim in stimuli: 
        for feature_index, features  in enumerate(FEATURE_MAP):
            if(all(np.array(stim)-1==features)):
                index=feature_index
        stim_indices.append(index)
    #print("stim indicies: ", stim_indices)
    for _, data in enumerate(train_loader):
        data_subset = []
        for stim_index in stim_indices:
            stim = torch.unsqueeze(data[stim_index], 0) 
            #im = np.transpose((data[stim_index].detach().numpy() * 255).astype(np.uint8), [1, 2, 0]) 
            #im = Image.fromarray(im)
            #im.show()
            _, util = model(stim)
            utilities.append(util.detach().numpy())
    
    utilities = np.array(utilities).clip(min=0) + 1e-6
    values = np.array([math.exp(inv_temp * utilities[stim_index]) for stim_index in range(3)])
    value_softmax = [values[0] / sum(values), values[1] / sum(values), values[2] / sum(values)]
    
    guess = np.random.choice([0,1,2], 1, p=[value_softmax[0], value_softmax[1], value_softmax[2]])[0]
    prediction = stimuli[guess]

    return (prediction, guess, [value_softmax[0], value_softmax[1], value_softmax[2]])

def get_updated_utilities(features):
    updated_utilities = np.array([np.sum((features[feature[0]] , features[feature[1] + 3], features[feature[2] + 6])) for feature in FEATURE_MAP])
    return (torch.from_numpy(updated_utilities)).float() 

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
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-ul', '--util-loss',
                       default=default_config['util_loss'], choices=UTIL_LOSSES,
                       help="Type of Utility loss function to use.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

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
    modelling.add_argument('-u', '--upsilon', type=int,
                            default=default_config['upsilon'],
                            help='Weight for utility loss in training.')
    modelling.add_argument('--feature_labels', type=int,
                            default=default_config['feature_labels'],
                            help='Weight for utility loss in training.')

    modelling.add_argument('--dropout_percent', type=int,
                            default=20,
                            help='Dropout for CNN')
                
    
    args = parser.parse_args(args_to_parse)


    return args

def save_model(model, exp_dir, metadata):
    model.save(exp_dir)
    return 

def load_model(exp_dir, args):
    model = CNN(utility_type=args.utility_type, img_size=args.img_size, latent_dim=2*args.latent_dim,  kwargs=vars(args))
    return model.load(exp_dir, args)

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
    Choices = mat['DimTaskData'][0][0]['Choices']
    Outcomes = mat['DimTaskData'][0][0]['Outcomes']
    Stimuli = mat['DimTaskData'][0][0]['Stimuli']
    RelevantDim = mat['DimTaskData'][0][0]['RelevantDim']
    CorrectFeature = mat['DimTaskData'][0][0]['CorrectFeature']
    ReactionTimes = mat['DimTaskData'][0][0]['ReactionTimes']
    Feature_Thetas_500, Feature_Thetas_300 = Feature_RL.get_thetas()

    # retrain and predict 
    ResponseAccuracy = pd.DataFrame()
    states = np.matrix([
    [1,2,3],
    [1,2,3],
    [1,2,3]
    ])

    actions = np.linspace(0,1,num=1)
    env = frl_env(states=states, actions=actions, rewards=[])

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
    if(not os.path.exists(exp_dir)): os.mkdir(exp_dir)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    
    if(not args.is_eval_only):
        # pretrain model or load pretrained on reconstructing stimuli set
        model = CNN(utility_type=args.utility_type, img_size=args.img_size, latent_dim=2*args.latent_dim,  kwargs=vars(args))
        model = model.to(device)  # make sure trainer and viz on same device
        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger)
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        trainer = Trainer(model, optimizer,
                            device=device,
                            logger=logger,
                            save_dir=exp_dir,
                            is_progress_bar=not args.no_progress_bar)
        utilities = np.ones((27))* 0.5
        utilities = torch.from_numpy(utilities.astype(np.float64)).float()
        # feature label for red, square, hatch [1,0,0,1,0,0,1,0,0]
        feature_labels = []
        for color in [0,1,2]:
            for shape in [3,4,5]:
                for texture in [6,7,8]:
                    feature_label = np.zeros((9))
                    feature_label[color] = 1
                    feature_label[shape] = 1
                    feature_label[texture] = 1
                    feature_labels.append(feature_label)

        feature_labels = np.array(feature_labels)
        trainer(train_loader,
                utilities=utilities, 
                feature_labels=feature_labels,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))
    
    model = load_model(exp_dir, args)
    model.to(device)
    base_model = copy.deepcopy(model)

    for participant_id in range(0,22):
        alpha_300 = Feature_Thetas_300[participant_id][0] 
        beta_300 = Feature_Thetas_300[participant_id][1] 
        delta_300 = Feature_Thetas_300[participant_id][2] 

        alpha_500 = Feature_Thetas_500[participant_id][0] 
        beta_500 = Feature_Thetas_500[participant_id][1] 
        delta_500 = Feature_Thetas_500[participant_id][2] 

        feature_rl_300 = Feature_RL(env, eta = alpha_300, delta = delta_300, beta = beta_300)
        feature_rl_500 = Feature_RL(env, eta = alpha_500, delta = delta_500, beta = beta_500)

        last_prediction = -1
        old_relevant = -1

        for trial_num in range(0,800):
            color_feature_1 = Stimuli[trial_num][0][0][participant_id]
            shape_feature_1 = Stimuli[trial_num][0][1][participant_id]
            texture_feature_1 = Stimuli[trial_num][0][2][participant_id] 
            color_feature_2 = Stimuli[trial_num][1][0][participant_id]
            shape_feature_2 = Stimuli[trial_num][1][1][participant_id] 
            texture_feature_2 = Stimuli[trial_num][1][2][participant_id] 
            color_feature_3 = Stimuli[trial_num][2][0][participant_id] 
            shape_feature_3 = Stimuli[trial_num][2][1][participant_id] 
            texture_feature_3 = Stimuli[trial_num][2][2][participant_id] 

            relevant = RelevantDim[trial_num][participant_id]
            correct = CorrectFeature[trial_num][participant_id]
            outcome = Outcomes[trial_num][participant_id]

            if(math.isnan(Choices[trial_num][0][participant_id])):
                continue 

            chosen = [
                int(Choices[trial_num][0][participant_id]),
                int(Choices[trial_num][1][participant_id]),
                int(Choices[trial_num][2][participant_id])
            ]
            
            stimulus = [
                [color_feature_1, shape_feature_1, texture_feature_1],
                [color_feature_2, shape_feature_2, texture_feature_2],
                [color_feature_3, shape_feature_3, texture_feature_3]
            ]

            for option_index, option in enumerate(stimulus):
                if (option == chosen): choice_index = option_index
            
            if(relevant != old_relevant):
                feature_rl_300.reset()
                feature_rl_500.reset()
                trial_game_index = 0

            prediction_outcome = True
            if(trial_num >= 500):
                (prediction, guess, percentages) = feature_rl_300.predict(stimulus)
                # ADD to dataframe 
                feature_rl_300.forget(chosen)
                feature_rl_300.train_state_reward(chosen, outcome)
                feature_values = feature_rl_300.q_table

            else: 
                (prediction, guess, percentages) = feature_rl_500.predict(stimulus)
                feature_rl_500.forget(chosen)
                feature_rl_500.train_state_reward(chosen, outcome)
                feature_values = feature_rl_500.q_table
            
            frl_chosen_percentage = percentages[choice_index]
            
            train_loader = get_dataloaders(args.dataset, batch_size=args.batch_size)
            inv_temp = beta_300 if trial_num >= 500 else beta_500
            (prediction, guess, bvae_percentages) = predict_utilities(stimulus, model, train_loader, inv_temp = inv_temp)
            bvae_chosen_percentage = bvae_percentages[choice_index]

            if(trial_game_index < 25):
                ResponseAccuracy = ResponseAccuracy.append({    "EpisodeTrial": trial_game_index, 
                                                                "ParticipantId":  participant_id, 
                                                                "Accuracy":frl_chosen_percentage, 
                                                                "Model Type":"FRL",
                                                                "Correct":correct}, ignore_index=True)
                
                ResponseAccuracy = ResponseAccuracy.append({    "EpisodeTrial": trial_game_index, 
                                                                "ParticipantId":  participant_id, 
                                                                "Accuracy":bvae_chosen_percentage, 
                                                                "Model Type":"BVAE",
                                                                "Epochs":args.model_epochs,
                                                                "Correct":correct}, ignore_index=True)

            trial_game_index += 1
            old_relevant = relevant

            updated_utilities = get_updated_utilities(feature_values.flatten())
            #print("updated_utilities: ", updated_utilities)
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            trainer = Trainer(model, optimizer,
                                    device=device,
                                    logger=None,
                                    save_dir=exp_dir,
                                    is_progress_bar=False)
            
            epochs = args.model_epochs if trial_game_index > 1 else 100
            trainer(train_loader,
                utilities=updated_utilities, 
                feature_labels=feature_labels,
                epochs=epochs, 
                checkpoint_every=10000)

    """print(ResponseAccuracy)
    #ResponseAccuracy.plot(x='EpisodeTrial', y='Accuracy')
    ax = sns.lineplot(data=ResponseAccuracy, x="EpisodeTrial", y="Accuracy", hue="Model Type")
    ax.set_title('FRL vs. CNN Predictive Accuracy (100 Epochs)')
    ax.set_ylabel('Predictive Accuracy')
    ax.set_xlabel('Episode Trial')
    plt.show()"""
    #print(np.mean(all_probabilities, axis=0))

    ResponseAccuracy.to_pickle(exp_dir + "./Predictions.pkl") 

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


