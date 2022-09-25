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

fig, axes = plt.subplots(1, 2, sharey=True)

learning = pd.read_pickle("./representationAnalysis_e3.pkl")

learning['Representation Similarity'] = learning['Representation Similarity'].round(1)

learning = learning.loc[learning["Utility Difference"] >= 0]

early_trials = learning.loc[learning["Training Step"] <= 1]
later_trials = learning.loc[learning["Training Step"] >= 9]

sns.lineplot(data=early_trials, x="Representation Similarity", y="Utility Difference", ax=axes[0])
sns.lineplot(data=later_trials, x="Representation Similarity", y="Utility Difference", ax=axes[1])

axes[0].set_title("Early Trials", fontsize=14)
axes[1].set_title("Later Trials", fontsize=14)

axes[1].set_ylabel("")
axes[0].set_ylabel("Utility Difference", fontsize=14)

axes[0].set_xlabel("Representation Similarity", fontsize=14)
axes[1].set_xlabel("Representation Similarity", fontsize=14)

plt.suptitle("UB-VAE Model Latent Representation Difference by Utility Difference", fontsize=16)

plt.show()

print(learning)

assert(False)
cd_idx = pd.DataFrame()

slopes = [0.014309261354166235, 0.004012543369982307, 0.14738274574369203, 0.22752515185796615, 0.2537325153354331, 0.2521593055735751, 0.27214183481241155, 0.2558224636360998, 0.26486472742308376]

for idx, slope in enumerate(slopes):
    cd_idx = cd_idx.append({"Number of Utility Observations": idx + 1, "Change Detection Regression Slope":slope}, ignore_index=True)


#sns.lineplot(data=cd_idx, x="Number of Utility Observation", y="Change Detection Regression Slope")
sns.regplot(data=cd_idx, x="Number of Utility Observations", y="Change Detection Regression Slope", scatter=True, line_kws={"color":"orange"})

plt.xlabel('Number of Utility Observations', fontsize=14)
plt.ylabel('Change Detection Regression Slope', fontsize=14)
plt.title("UB-VAE Model Change Detection Regression Slope by Number of Utility Observations", fontsize=16)
plt.show()

assert(False)

changeDetection = pd.read_pickle("./bvae_change_detection.pkl")

changeDetection = changeDetection.loc[changeDetection['Utility Difference'] > 0]
changeDetection = changeDetection.loc[changeDetection['Utility Difference'] < 1]

changeDetection['Utility Difference'] = changeDetection['Utility Difference'].round(1)

util_difs = changeDetection['Utility Difference'].unique()

means = changeDetection.groupby('Utility Difference')['Probability of Detecting Change'].mean().to_numpy()

changeDetectionMeans = pd.DataFrame()

for idx, mean in enumerate(means):
    if(util_difs[idx] == 0.5):continue # outlier 
    print(util_difs[idx])
    changeDetectionMeans = changeDetectionMeans.append({"Utility Difference": util_difs[idx], "Probability of Detecting Change":mean}, ignore_index=True)

#sns.lineplot(data=changeDetection, x="Utility Difference", y="Probability of Detecting Change")
sns.regplot(data=changeDetectionMeans, x="Utility Difference", y="Probability of Detecting Change", line_kws={"color":"orange"})

plt.xlabel('Stimuli Utility Difference', fontsize=14)
plt.ylabel('Probability of Detecting Change', fontsize=14)
plt.title("UB-VAE Model Probability of Detecting Change by Stimuli Utility Difference", fontsize=16)
plt.show()

assert(False)

riskAverseData = pd.read_pickle("./bvae_risk_aversion.pkl")


sns.lineplot(data=riskAverseData, x="Utility Standard Deviation Difference", y="Probability of Selecting Stimuli")
sns.regplot(data=riskAverseData, x="Utility Standard Deviation Difference", y="Probability of Selecting Stimuli", scatter=False)

plt.xlabel('Utility Standard Deviation Difference', fontsize=14)
plt.ylabel('Probability of Selecting Stimuli', fontsize=14)
plt.title("UB-VAE Model Probability of Selecting Stimuli by Utility Standard Deviation Difference", fontsize=16)
plt.show()