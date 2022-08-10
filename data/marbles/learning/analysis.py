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
import ast 
import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 

from os import listdir
from os.path import isfile, join

all_marble_colors = pd.read_csv("../source/colors.csv")
all_marble_colors = all_marble_colors['colors']

folder = '../learning/data'
all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]


def main():
    all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]

    for participant_data in all_participant_data:
    #for participant_data in ["1251003574_20220803.csv", "3507807121_20220803.csv", "6089478690_20220803.csv", "7687409581_20220803.csv", "9501547299_20220803.csv"]:
        data = pd.read_csv(join(folder, participant_data))  
        stimuli_set = int(data['marble_set'][0])

        all_color_rewards = [
            [40,30,25],
            [40,25,30],
            [30,40,25],
            [35,40,25],
            [30,25,35]
        ]

        dataFrame = pd.DataFrame()
        data = data.tail(200)

        game_ind = 0
        for i, ind in enumerate(data.index):
            if i % 20 == 0: 
                game_ind += 1
                trial_ind = 0
            
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

            stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
            stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

            stim_1_util = stimuli_marble_values

            trial_ind += 1

            if(data['type'][ind] == 1.0):
                

                key_press = data['key_press'][ind]
                reward = int(data['reward'][ind])

                if(key_press != 'd' and key_press != 'f'):
                    continue
                
                if(key_press == 'd'):
                    correct = 1 if stim_1_true_util > stim_2_true_util else 0
                elif(key_press == 'f'):
                    correct = 1 if stim_2_true_util > stim_1_true_util else 0
                else:
                    assert(False)
                
                dataFrame = dataFrame.append({"Accuracy":correct, "Trial": trial_ind, "Game": game_ind}, ignore_index=True)
                
                
            
            if(data['type'][ind] == 0.0):
                key_press = data['key_press'][ind]
                if(key_press != 'k' and key_press != 'j'):
                    continue

                changed = data['changed'][ind]
                
                if(key_press == 'j'): # predict same 
                    correct = 1 if not changed else  0

                elif(key_press == 'k'):
                    correct = 1 if changed else  0
                
                dataFrame = dataFrame.append({"Accuracy":correct, "Trial": trial_ind, "Game": game_ind}, ignore_index=True)

    
    print(dataFrame)
    sns.lineplot(data=dataFrame, x="Trial", y="Accuracy")
    plt.show()
    assert(False)
    #print(predictive_accuracy)
    if(len(predictive_accuracy) == 0): return 0
    return -1 * np.mean(predictive_accuracy)


if __name__ == '__main__':
    main()
