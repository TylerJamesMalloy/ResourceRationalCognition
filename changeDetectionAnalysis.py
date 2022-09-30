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

all_marble_colors = pd.read_csv("./data/marbles/source/colors.csv")
all_marble_colors = all_marble_colors['colors']



folder = './data/marbles/decisions/data2/'
all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]

good_participants = ['1088359975_20220708.csv', '1384981370_20220710.csv', '1748395787_20220709.csv', '1832380163_20220710.csv', '1996454642_20220710.csv', '2285081095_20220709.csv', '3072823518_20220709.csv', '3209482804_20220710.csv', '3280341755_20220709.csv', '3437307782_20220709.csv', '3684971091_20220710.csv', '4192753508_20220710.csv', '4617021112_20220709.csv', '4984990593_20220710.csv', '5649795488_20220711.csv', '6261906642_20220709.csv', '6967768109_20220708.csv', '7036685623_20220709.csv', '7361812709_20220709.csv', '7714472260_20220710.csv', '7763967651_20220710.csv', '7781888656_20220709.csv', '8056959514_20220709.csv', '8114269562_20220709.csv', '8214654421_20220710.csv', '8242903913_20220710.csv', '8466633972_20220709.csv', '8473787759_20220709.csv', '8854732576_20220710.csv', '8893453676_20220710.csv', '8988448256_20220710.csv', '9201972787_20220709.csv', '9375774875_20220710.csv', '9553285857_20220709.csv', '9852782779_20220709.csv']
#good_participants = all_participant_data


def accuracy(participant_data):
    data = pd.read_csv(join(folder, participant_data))  

    print(participant_data)

    color_rewards = [2,3,4]

    dataFrame = pd.DataFrame()


    data = data.tail(200)
    game = 0
    utility_example_ind = 0
    
    for i, ind in enumerate(data.index):
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

        changed = data['changed'][ind]
        change_index = int(data['change_index'][ind])

        key_press = data['key_press'][ind]
        
        if(key_press != 'k' and key_press != 'j'):
            continue

        if(not changed): continue 

        if(key_press == 'j'): # predict same 
            detected = 0
        else:
            detected = 1

        if(change_index == 0):
            if(np.sum(stimuli_marble_values[stim_1] == stimuli_marble_values[new_stim]) >= 6 and np.sum(stimuli_marble_values[stim_1] == stimuli_marble_values[new_stim]) < 9):
                utility_change = np.mean(stimuli_marble_values[stim_1]) - np.mean(stimuli_marble_values[new_stim])
                utility_change = np.round(np.abs(utility_change), 2)

                d = {"Utility Change": utility_change, "Change Detection Accuracy":detected}
                dataFrame = dataFrame.append(d, ignore_index=True)
        elif(change_index == 1):
            if(np.sum(stimuli_marble_values[stim_2] == stimuli_marble_values[new_stim]) >= 6 and np.sum(stimuli_marble_values[stim_2] == stimuli_marble_values[new_stim]) < 9):
                utility_change = np.mean(stimuli_marble_values[stim_2]) - np.mean(stimuli_marble_values[new_stim])
                utility_change = np.round(np.abs(utility_change), 2)

                d = {"Utility Change": utility_change, "Change Detection Accuracy":detected}

                dataFrame = dataFrame.append(d, ignore_index=True)
        else:
            continue
        

    return dataFrame  

def main():
    all_data = pd.DataFrame()
    #for participant_data in all_participant_data:
    for participant_data in good_participants:
        data_accuracy = accuracy(participant_data)
        all_data = all_data.append(data_accuracy, ignore_index=True)


    all_data.to_pickle("./participantChangeDetection.pkl")

    print(all_data)
    ax = sns.barplot(x="Utility Change", y="Change Detection Accuracy", data=all_data)
    plt.show()

if __name__ == '__main__':
    main()
