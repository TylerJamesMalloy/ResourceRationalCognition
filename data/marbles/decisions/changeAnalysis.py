from cmath import nan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd 
import numpy as np 
import seaborn as sns
import scipy.stats as stat
import matplotlib.pyplot as plt 
import glob
import ast 
import math 
import scipy 

import matplotlib.pyplot as plt

data_paths = glob.glob("both/*")
all_marble_colors = pd.read_csv("../source/colors.csv")
all_marble_colors = all_marble_colors['colors']

response_accuracy = []
bad_participants = []
good_participants = []

for participant_id, data_path in enumerate(data_paths):
    #print(data_path)
    data = pd.read_csv(data_path)
    paricipant_correct_responses = 0
    paricipant_incorrect_responses = 0
    cumulative_regret = 0

    experiment_data = data.tail(180).to_numpy()
    participant_points = 0

    change_detection = []

    
    for response in experiment_data:
        if np.array([value != value for value in response]).any(): 
            cumulative_regret += 1
            continue 

        reward = response[2]
        if(reward == reward): participant_points += reward
        
        stim_1 = int(response[6])
        stim_2 = int(response[7])
        key_press = response[1]
        correct = None 

        if(key_press == "j" or key_press == 'k'): 
            changed = response[10] if len(response) == 12 else response[11]
            changed_index = response[1]

            correct = 1 if ((changed == 0.0 and key_press == "j") or (changed == 1.0 and key_press == "k")) else 0
            if not correct:
                cumulative_regret += 3

        if correct != None:
            if(correct == 1):
                paricipant_correct_responses += 1
            else:
                paricipant_incorrect_responses += 1

    #if((paricipant_correct_responses + paricipant_incorrect_responses) == 0 or paricipant_correct_responses / (paricipant_correct_responses + paricipant_incorrect_responses) < 0.61):
    #response_accuracy.append(paricipant_correct_responses) 
    #response_accuracy.append(participant_points) 
    if((paricipant_correct_responses + paricipant_incorrect_responses) > 0):
        response_accuracy.append(paricipant_correct_responses / (paricipant_correct_responses + paricipant_incorrect_responses)) 
        #response_accuracy.append(cumulative_regret)
        if(paricipant_correct_responses / (paricipant_correct_responses + paricipant_incorrect_responses) <= 0.68):
            bad_participants.append(data_path)
        else:
            good_participants.append(data_path)
    else:
        response_accuracy.append(0)
        #response_accuracy.append(999)


print(good_participants)
print(len(good_participants))
response_accuracy = np.array(response_accuracy)
#response_accuracy = response_accuracy[(response_accuracy < 999)]
#response_accuracy = response_accuracy[(response_accuracy < 200)]

#print("number of participants with low cumulative regret: ", len(response_accuracy))

response_accuracy = np.array(response_accuracy)

decent_response_accuracy = response_accuracy[(response_accuracy > 0.68)]
print("number of participants with more than 68 percent change detection: ", len(decent_response_accuracy), " out of ", len(response_accuracy), " total participants")

fig, ax = plt.subplots(figsize=(8,3))
_, bins, _ = plt.hist(decent_response_accuracy, 12, histtype = 'bar', density=True)


mu, sigma = scipy.stats.norm.fit(decent_response_accuracy)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
#print(mu, sigma, best_fit_line)
plt.plot(bins, best_fit_line, axes=ax)

#plt.xticks([0, 30, 68, 120, 150])
#plt.xticks([0.64, 0.75, 0.9])
plt.title("Percent Strictly Correct Utility Choices")
plt.show()

