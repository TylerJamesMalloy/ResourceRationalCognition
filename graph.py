import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 

data = pd.DataFrame()
bvae = pd.read_pickle("./results/bvae/n3_b4/Predictions.pkl")
vae = pd.read_pickle("./results/vae/Predictions.pkl")


data = data.append(bvae, ignore_index=True)
data = data.append(vae, ignore_index=True)


ax = sns.lineplot(data=data, x="EpisodeTrial", y="Accuracy", hue="Beta")
ax.set_title('FRL vs. BVAE vs VAE Predictive Accuracy (1 Epochs)', fontsize=20)
ax.set_ylabel('Predictive Accuracy', fontsize=16)
ax.set_xlabel('Episode Trial', fontsize=16)
plt.show()