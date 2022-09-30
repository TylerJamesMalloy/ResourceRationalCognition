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

order_list = ['BVAE', 'CNN', 'Sparse', 'CPT', 'MEU']


utilitySelectionData = pd.read_pickle("./modelAccuracy_e1.pkl")
ax = sns.barplot(x="Model", y="Predictive Accuracy", data=utilitySelectionData, ax=axes[0], order=order_list)
axes[0].set_title("Utility Selection Trial")

order_list = ['BVAE', 'CNN', 'Sparse', 'Utility', 'Visual']

changeDetectionData = pd.read_pickle("./modelAccuracy_e2.pkl")

mask = changeDetectionData["Model"] == "MEU"
column_name = 'Model'
changeDetectionData.loc[mask, column_name] = "Utility"

mask = changeDetectionData["Model"] == "CPT"
column_name = 'Model'
changeDetectionData.loc[mask, column_name] = "Visual"

#changeDetectionData_v2 = pd.read_pickle("./modelAccuracy_e2_v2.pkl")
ax = sns.barplot(x="Model", y="Predictive Accuracy", data=changeDetectionData, ax=axes[1], order=order_list)
axes[1].set_title("Change Detection Trial")
axes[1].set_ylabel("")


order_list = order_list = ['BVAE', 'CNN', 'Sparse']

#cpt_fixed_data = changeDetectionData_v2.loc[changeDetectionData_v2['Model'] == "CPT"]

df_merged = utilitySelectionData.append(changeDetectionData, ignore_index=True)

df_merged = df_merged.loc[(df_merged["Model"] != "CPT") & (df_merged["Model"] != "MEU")]

#allTrialData = pd.read_pickle("./modelAccuracy_e2.pkl")
ax = sns.barplot(x="Model", y="Predictive Accuracy", data=df_merged, ax=axes[2], order=order_list)
axes[2].set_title("All Trials")
axes[2].set_ylabel("")
axes[2].set_title("All Trial Types")

plt.suptitle("Predictive Accuracy by Model Type", fontsize=16)

plt.show()

