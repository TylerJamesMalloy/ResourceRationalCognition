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

import math

from os import listdir
from os.path import isfile, join

import pandas as pd 
import numpy as np 
import ast 
import matplotlib.pyplot as plt
import scipy.optimize as optimize

import seaborn as sns 

import statsmodels.api as sm

from scipy.stats import entropy

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

all_marble_colors = pd.read_csv("./data/marbles/source/colors.csv")
all_marble_colors = all_marble_colors['colors']

stimuli_mean_utilities = []
stimuli_deviations = []
stimuli_marble_values = []

all_means = []
all_std = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))
    stimuli_marble_values.append(marble_values)
    all_means.append(np.mean(marble_values))
    all_std.append(np.std(marble_values))

print(stimuli_marble_values[20])
print(stimuli_marble_values[7])
print(all_means[20])
print(all_means[7])

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

sample_20_small = []
sample_7_small = []

sample_20_big = []
sample_7_big = []



for _ in range(1000):
    sample_mean_20 = []
    sample_mean_7 = []
    for _ in range(10):
        sample_mean_20.append(np.random.choice(stimuli_marble_values[20]))
        sample_mean_7.append(np.random.choice(stimuli_marble_values[7]))
    
    sample_20_small.append(np.mean(sample_mean_20))
    #sample_20_small.append(np.mean(sample_mean_20) > np.mean(sample_mean_7))
    sample_7_small.append(np.mean(sample_mean_7))
    #sample_7_small.append(np.mean(sample_mean_20) < np.mean(sample_mean_7))

    sample_mean_20 = []
    sample_mean_7 = []

    for _ in range(100):
        sample_mean_20.append(np.random.choice(stimuli_marble_values[20]))
        sample_mean_7.append(np.random.choice(stimuli_marble_values[7]))
    
    sample_20_big.append(np.mean(sample_mean_20))
    #sample_20_big.append(np.mean(sample_mean_20) > np.mean(sample_mean_7 ))
    sample_7_big.append(np.mean(sample_mean_7))
    #sample_7_big.append(np.mean(sample_mean_20) < np.mean(sample_mean_7 ))

print(entropy(sample_20_small))
print(entropy(sample_7_small))

print(entropy(sample_20_big))
print(entropy(sample_7_big))

ax[0,0].hist(sample_20_small)
ax[1,0].hist(sample_7_small)

ax[0,1].hist(sample_20_big)
ax[1,1].hist(sample_7_big)
plt.show() 