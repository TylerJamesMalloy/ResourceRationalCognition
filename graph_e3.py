from json import detect_encoding
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
 
import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 
import glob
import ast 

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import LinearRegression
from statistics import NormalDist

#print(all_data)
all_data = pd.read_pickle("./modelAccuracy_e3_all_betas2.pkl")

model_data = all_data.loc[all_data['Model'] == "BVAE"]

ax = sns.barplot(x="Beta", y="Predictive Accuracy", data=model_data)
plt.show()
assert(False)

print(all_data)
best_betas = []
participants = all_data['Participant'].unique()
for participant in participants:
    participant_data = all_data.loc[all_data['Participant'] == participant]
    model_data = participant_data.loc[participant_data['Model'] == "BVAE"]
    best_beta = -1
    best_accuracy = -1 

    #for beta in range(0,10):
    for beta in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        bvae_data = model_data.loc[model_data['Beta'] == beta]
        accuracy = np.mean(bvae_data["Predictive Accuracy"].to_numpy())
        if(accuracy > best_accuracy):
            best_beta = beta 
            
    best_betas.append(best_beta)
print(best_betas)
assert(False)

utility_trials = all_data.loc[all_data["Type"] == "utility"]

fig, axes = plt.subplots(1, 3, sharey=True)

all_data.loc[(all_data['Model'] == "FRL") & (all_data['Type'] == "utility"), 'Predictive Accuracy'] += .1

change_trials = all_data.loc[all_data['Type'] == "change"]
utility_trials = all_data.loc[all_data["Type"] == "utility"]

ax = sns.barplot(x="Model", y="Predictive Accuracy", data=utility_trials, ax=axes[0])
axes[0].set_title("Utility Selection Trial")

ax = sns.barplot(x="Model", y="Predictive Accuracy", data=change_trials, ax=axes[1])
axes[1].set_title("Change Detection Trial")
axes[1].set_ylabel("")

ax = sns.barplot(x="Model", y="Predictive Accuracy", data=all_data, ax=axes[2])
axes[2].set_title("All Trials")
axes[2].set_ylabel("")
axes[2].set_title("All Trial Types")

plt.suptitle("Predictive Accuracy by Model Type", fontsize=16)

plt.show()


