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

fig, axes = plt.subplots(1, 3, sharey=True)

order_list = ['CPT', 'MEU', 'BVAE', 'CNN', 'Sparse']

utilitySelectionData = pd.read_pickle("./modelAccuracy_e1.pkl")
ax = sns.barplot(x="Model", y="Predictive Accuracy", data=utilitySelectionData, ax=axes[0], order=order_list)
axes[0].set_title("Utility Selection Trial")

changeDetectionData = pd.read_pickle("./modelAccuracy_e2.pkl")
ax = sns.barplot(x="Model", y="Predictive Accuracy", data=changeDetectionData, ax=axes[1], order=order_list)
axes[1].set_title("Change Detection Trial")
axes[1].set_ylabel("")

df_merged = utilitySelectionData.append(changeDetectionData, ignore_index=True)

#allTrialData = pd.read_pickle("./modelAccuracy_e2.pkl")
ax = sns.barplot(x="Model", y="Predictive Accuracy", data=df_merged, ax=axes[2], order=order_list)
axes[2].set_title("All Trials")
axes[2].set_ylabel("")
axes[2].set_title("All Trial Types")

plt.suptitle("Predictive Accuracy by Model Type", fontsize=16)

plt.show()

