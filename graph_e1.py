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


representationData = pd.read_pickle("./e1_Representations.pkl")
overlapData = pd.DataFrame()
#print(representationData)
def kl_normal_loss(pm, plv, qm, qlv):
    qv = np.exp(qlv)
    pv = np.exp(plv)
    pm = np.asarray(pm)
    qm = np.asarray(qm)
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm))) 



for ind in representationData.index:
    latent_means = representationData['Latent Mean'][ind]
    latent_vars = representationData['Latent Var'][ind]
    util_std    = representationData['Utility STD'][ind]
    util_mean   = representationData['Mean Utility'][ind]

    for comp in representationData.index:
        if ind == comp: continue 

        comp_latent_means = representationData['Latent Mean'][comp]
        comp_latent_vars = representationData['Latent Var'][comp]
        comp_util_std    = representationData['Utility STD'][comp]
        comp_util_mean   = representationData['Mean Utility'][comp]

        kl_loss = kl_normal_loss(comp_latent_means, comp_latent_vars, latent_means, latent_vars)
        #utility_loss = ((util_mean - comp_util_mean) ** 2) / 2
        #utility_loss = np.abs(util_mean - comp_util_mean)
        if util_std != 0 and comp_util_std != 0: 
            utility_overlap = NormalDist(mu=util_mean, sigma=util_std).overlap(NormalDist(mu=comp_util_mean, sigma=comp_util_std))
        else:
            utility_overlap = 0
            #utility_overlap = np.abs(util_mean - comp_util_mean) / 2

        #utility_overlap = round(utility_overlap, 1)
        #utility_overlap = round(utility_overlap*4)/4
        utility_overlap = round(utility_overlap*20)/20
        overlap = 0 
        for latent_mean, latent_logvar, comp_latent_mean, comp_latent_var, in zip(latent_means, latent_vars, comp_latent_means, comp_latent_vars):
            overlap += NormalDist(mu=latent_mean, sigma= np.exp(latent_logvar * 0.5)).overlap(NormalDist(mu=comp_latent_mean, sigma= np.exp(comp_latent_var * 0.5)))

        d = {"Stimuli Representation Overlap": overlap,  "Stimuli Utility Distribution Overlap": utility_overlap}
        overlapData = overlapData.append(d, ignore_index=True)


overlapData = overlapData[overlapData["Stimuli Utility Distribution Overlap"] > 0.3]
ax = sns.lineplot(x="Stimuli Utility Distribution Overlap", y="Stimuli Representation Overlap", data=overlapData)

x = overlapData['Stimuli Utility Distribution Overlap'].unique()
y = np.array([])
e = np.array([])

print("number of unique values: ", len(x))

for val in x:
    data = overlapData.loc[overlapData['Stimuli Utility Distribution Overlap'] == val]
    y = np.append(y, data['Stimuli Representation Overlap'].mean())
    e = np.append(e, data['Stimuli Representation Overlap'].var())

#x = overlapData['Stimuli Utility Difference'].to_numpy()
#y = overlapData['Stimuli Representation Overlap'].to_numpy()
model = LinearRegression()
x = x.reshape(-1, 1)
reg = model.fit(x, y)

print("reg score: ", reg.score(x, y))

y_pred = model.predict(x)
plt.plot(x, y_pred, color="red", linewidth=2)
#ax.set(xlim=(-0.5, 0.5))
plt.title("Stimuli Representation Overlap by Utility Distribution Overlap", fontsize=18)
plt.xlabel('Stimuli Utility Distribution Overlap', fontsize=14)
plt.ylabel('Representation Overlap', fontsize=14)
plt.show()


#all_data = pd.read_csv("./modelAccuracy.csv")
#ax = sns.barplot(x="Model", y="Predictive Accuracy", data=all_data)
#ax.set(ylim=(0.5, 0.7))
#plt.title("Model Predictive Accuracy")
#plt.show()

assert(False)
all_marble_colors = pd.read_csv("./data/marbles/source/colors.csv")
all_marble_colors = all_marble_colors['colors']

"""
stimuli_mean_utilities = []
stimuli_deviations = []
stimuli_marble_values = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))
    stimuli_marble_values.append(marble_values)
"""

# analyzing performance to show risk-averse behaviour 

"""
#e1_data_paths = glob.glob("./data/marbles/decisions/both/*")
#good_participants = ['1206472879_20220607.csv', '1481432285_20220607.csv', '1571956186_20220623.csv', '1655417347_20220607.csv', '169075273_20220607.csv', '2307452822_20220602.csv', '2429104214_20220607.csv', '2669911017_20220607.csv', '282458648_20220622.csv', '2969042685_20220622.csv', '3010097611_20220622.csv', '302208162_20220607.csv', '3050445599_20220701.csv', '3072452500_20220623.csv', '3310926888_20220622.csv', '3437000926_20220623.csv', '3525719084_20220607.csv', '3545302544_20220623.csv', '3709436102_20220622.csv', '3774486973_20220702.csv', '3777558888_20220602.csv', '4302545825_20220607.csv', '4309499885_20220624.csv', '4424042522_20220623.csv', '4604314752_20220625.csv', '4717805082_20220622.csv', '4784817211_20220623.csv', '4833293935_20220607.csv', '5138618294_20220607.csv', '5144493038_20220602.csv', '5534437613_20220701.csv', '5878990705_20220607.csv', '6130248011_20220622.csv', '6174180168_20220602.csv', '6176365135_20220602.csv', '6247410167_20220607.csv', '682320948_20220701.csv', '685851185_20220701.csv', '6969137467_20220622.csv', '7056217438_20220622.csv', '748797646_20220518.csv', '7489651562_20220701.csv', '7708514735_20220701.csv', '7729591288_20220607.csv', '7811512263_20220623.csv', '8198410857_20220622.csv', '8254485902_20220623.csv', '8488980532_20220602.csv', '8762245299_20220622.csv', '8880742555_20220623.csv', '8894686670_20220622.csv', '908986223_20220622.csv', '9162481065_20220607.csv', '9177013872_20220518.csv', '9195187466_20220607.csv', '9262283903_20220623.csv', '934906418_20220623.csv', '9412783563_20220607.csv', '9796178986_20220623.csv', '9824877929_20220518.csv']
good_participants = ['1206472879_20220607.csv', '13896051_20220623.csv', '1481432285_20220607.csv', '1571956186_20220623.csv', '1655417347_20220607.csv', '1670585779_20220607.csv', '169075273_20220607.csv', '1917494030_20220701.csv', '2024508277_20220622.csv', '2307452822_20220602.csv', '2429104214_20220607.csv', '2467485070_20220701.csv', '2616009388_20220607.csv', '2669911017_20220607.csv', '282458648_20220622.csv', '2969042685_20220622.csv', '3010097611_20220622.csv', '3016456232_20220622.csv', '302208162_20220607.csv', '3050445599_20220701.csv', '3072452500_20220623.csv', '3231145639_20220622.csv', '3310926888_20220622.csv', '3437000926_20220623.csv', '3453896152_20220518.csv', '3525719084_20220607.csv', '3545302544_20220623.csv', '3627110067_20220623.csv', '3709436102_20220622.csv', '3774486973_20220702.csv', '3777558888_20220602.csv', '3868544605_20220622.csv', '424296399_20220622.csv', '4302545825_20220607.csv', '4309499885_20220624.csv', '4424042522_20220623.csv', '4522484535_20220602.csv', '4604314752_20220625.csv', '4717805082_20220622.csv', '4737559307_20220623.csv', '4758284626_20220607.csv', '4784817211_20220623.csv', '4786413128_20220623.csv', '4833293935_20220607.csv', '5138618294_20220607.csv', '5144493038_20220602.csv', '5347559166_20220701.csv', '5534437613_20220701.csv', '5552993317_20220602.csv', '5878990705_20220607.csv', '5979681843_20220623.csv', '6130248011_20220622.csv', '6174180168_20220602.csv', '6176365135_20220602.csv', '6247410167_20220607.csv', '6737332423_20220607.csv', '6745644970_20220623.csv', '682320948_20220701.csv', '685851185_20220701.csv', '6948208686_20220602.csv', '6969137467_20220622.csv', '7043291063_20220622.csv', '7056217438_20220622.csv', '7075207841_20220622.csv', '7243344067_20220701.csv', '7351329913_20220701.csv', '748797646_20220518.csv', '7489651562_20220701.csv', '7560795788_20220624.csv', '7708514735_20220701.csv', '7729591288_20220607.csv', '7811512263_20220623.csv', '7839131207_20220623.csv', '7916373955_20220622.csv', '8198410857_20220622.csv', '8254485902_20220623.csv', '8488980532_20220602.csv', '851978686_20220701.csv', '8762245299_20220622.csv', '8880742555_20220623.csv', '8894686670_20220622.csv', '9023291296_20220602.csv', '908333355_20220623.csv', '908986223_20220622.csv', '9162481065_20220607.csv', '9177013872_20220518.csv', '9195187466_20220607.csv', '9262283903_20220623.csv', '934906418_20220623.csv', '9410194125_20220623.csv', '9412783563_20220607.csv', '9796178986_20220623.csv', '9824877929_20220518.csv']

e1_all_data = pd.DataFrame()

for e1_id, e1_data_path in enumerate(good_participants):
    e1_data_path = "./data/marbles/decisions/both/" + e1_data_path
    data = pd.read_csv(e1_data_path)
    data = data.tail(200)
    data["Participant ID"] = e1_id
    e1_all_data = e1_all_data.append(data, ignore_index=True)

riskAverse = pd.DataFrame()
num_equal = 0
not_equal = 0
chose_lower_risk = 0 
chose_higher_risk = 0 
for ind in e1_all_data.index:
    if(e1_all_data['type'][ind] == 1.0):
        key_press = e1_all_data['key_press'][ind]
        if(key_press != 'd' and key_press != 'f'):
            continue 
        
        stim_1_true_util = stimuli_mean_utilities[int(e1_all_data['stim_1'][ind])]
        stim_2_true_util = stimuli_mean_utilities[int(e1_all_data['stim_2'][ind])]

        stim_1_std = stimuli_deviations[int(e1_all_data['stim_1'][ind])]
        stim_2_std = stimuli_deviations[int(e1_all_data['stim_2'][ind])]

        if(stim_1_true_util == stim_2_true_util):
            if((key_press == 'd' and stim_1_std <= stim_2_std) or (key_press == 'f' and stim_2_std <= stim_1_std)):
                chose_lower_risk += 1
            else:
                chose_higher_risk += 1
        
        chose = 0 if(key_press == 'd') else 1
        d = {"Deviation Difference": round(stim_1_std - stim_2_std, 2), "Chose":chose, "Participant":e1_all_data['Participant ID'][ind]}
        riskAverse = riskAverse.append(d, ignore_index=True)
        chose = 1 if(key_press == 'd') else 0
        d = {"Deviation Difference": round(stim_2_std - stim_1_std, 2), "Chose":chose, "Participant":e1_all_data['Participant ID'][ind]}
        riskAverse = riskAverse.append(d, ignore_index=True)

print("num chose_lower_risk stimuli: ", chose_lower_risk)
print("not chose_higher_risk stimuli: ", chose_higher_risk)

print(riskAverse)

riskAverse  = riskAverse[riskAverse['Deviation Difference'].between(-0.5, 0.5)]

x = riskAverse['Deviation Difference'].to_numpy()
y = riskAverse['Chose'].to_numpy()
model = LinearRegression()
x = x.reshape(-1, 1)
model.fit(x, y)

y_pred = model.predict(x)

ax = sns.lineplot(x="Deviation Difference", y="Chose", data=riskAverse)
plt.plot(x, y_pred, color="red", linewidth=2)
#ax.set(xlim=(-0.5, 0.5))
plt.title("Probability of Selecting Stimuli by Utility Deviation Difference", fontsize=18)
plt.xlabel('Utility Deviation Difference', fontsize=14)
plt.ylabel('Probability of Selecting Stimuli', fontsize=14)
plt.show()
"""

"""
e2_all_data = pd.DataFrame()
good_participants = ['1088359975_20220708.csv', '1384981370_20220710.csv', '1748395787_20220709.csv', '1832380163_20220710.csv', '1996454642_20220710.csv', '2285081095_20220709.csv', '3072823518_20220709.csv', '3209482804_20220710.csv', '3280341755_20220709.csv', '3437307782_20220709.csv', '3684971091_20220710.csv', '4192753508_20220710.csv', '4617021112_20220709.csv', '4984990593_20220710.csv', '5649795488_20220711.csv', '6261906642_20220709.csv', '6967768109_20220708.csv', '7036685623_20220709.csv', '7361812709_20220709.csv', '7714472260_20220710.csv', '7763967651_20220710.csv', '7781888656_20220709.csv', '8056959514_20220709.csv', '8114269562_20220709.csv', '8214654421_20220710.csv', '8242903913_20220710.csv', '8466633972_20220709.csv', '8473787759_20220709.csv', '8854732576_20220710.csv', '8893453676_20220710.csv', '8988448256_20220710.csv', '9201972787_20220709.csv', '9375774875_20220710.csv', '9553285857_20220709.csv', '9852782779_20220709.csv']
# analyzing performance to show bias in change detection based on utility 
for e2_id, e2_data_path in enumerate(good_participants):
    e2_data_path = "./data/marbles/decisions/both/" + e2_data_path
    data = pd.read_csv(e2_data_path)
    data = data.tail(200)
    data["Participant ID"] = e2_id
    e2_all_data = e2_all_data.append(data, ignore_index=True)

changeUtility = pd.DataFrame()
num_equal = 0
not_equal = 0
chose_lower_risk = 0 
chose_higher_risk = 0 
for ind in e2_all_data.index:
    if(e2_all_data['type'][ind] == 0.0):
        key_press = e2_all_data['key_press'][ind]
        changed = e2_all_data['changed'][ind]

        if(key_press != 'j' and key_press != 'k'):
            continue 

        if not changed: continue 
        
        stim_1_true_util = stimuli_mean_utilities[int(e2_all_data['stim_1'][ind])]
        stim_2_true_util = stimuli_mean_utilities[int(e2_all_data['stim_2'][ind])]
        new_stim_true_util = stimuli_mean_utilities[int(e2_all_data['new_stim'][ind])]

        changed = e2_all_data['changed'][ind]
        change_index = e2_all_data['change_index'][ind]

        first_stim_true_util = stim_1_true_util if change_index == 0 else stim_2_true_util

        utility_error = np.abs(first_stim_true_util - new_stim_true_util) 
        utility_error = round(utility_error, 3)

        if(key_press == 'k'):
            detect_change = 1
        else:
            detect_change = 0

        d = {"Utility Difference": utility_error, "Detect Change": detect_change }
        changeUtility = changeUtility.append(d, ignore_index=True)

x = changeUtility['Utility Difference'].unique()
y = np.array([])
e = np.array([])

print("number of unique values: ", len(x))

for val in x:
    data = changeUtility.loc[changeUtility['Utility Difference'] == val]
    y = np.append(y, data['Detect Change'].mean())
    e = np.append(e, data['Detect Change'].var())


#x = [0.7809187279151943, 0.8361344537815126, 0.8391959798994975, 0.8994252873563219, 0.8875502008032129, 0.9, 0.9366197183098591, 0.8932038834951457, 0.9459459459459459, 0.9166666666666666, 1.0]
#y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#m_x = changeUtility['Utility Difference'].to_numpy()
#m_y = changeUtility['Detect Change'].to_numpy()
model = LinearRegression()
x = x.reshape(-1, 1)
reg = model.fit(x, y)

print("reg score: ", reg.score(x, y))

y_pred = model.predict(x)

print(changeUtility)
#ax = sns.lineplot(x="Utility Difference", y="Detect Change", data=changeUtility)
#ax.set(xlim=(-0.5, 0.5))
plt.errorbar(x, y, yerr=e, fmt='o')
#plt.scatter(x,y)
plt.title("Probability of Detecting Change by Utility Difference", fontsize=18)
plt.plot(x, y_pred, color="red", linewidth=2)
plt.xlabel('Utility Difference', fontsize=14)
plt.ylabel('Probability of Detecting Change', fontsize=14)
plt.show()"""


        
# analyze performance to show learning utlity in experiment 3 

"""e3_all_data = pd.DataFrame()
changeUtility = pd.DataFrame()
good_participants = ['1049540426_20220809.csv', '1158286381_20220812.csv', '1224801952_20220817.csv', '125257521_20220809.csv', '1361833784_20220817.csv', '1472936997_20220812.csv', '1480381967_20220815.csv', '1683383581_20220809.csv', '1789205864_20220815.csv', '1804789041_20220812.csv', '2160484470_20220809.csv', '2452276079_20220817.csv', '248473192_20220817.csv', '2485825004_20220812.csv', '2551137904_20220812.csv', '2572158007_20220817.csv', '285880003_20220815.csv', '2891310284_20220817.csv', '2975869345_20220812.csv', '3549067443_20220815.csv', '3633378519_20220812.csv', '3758324478_20220816.csv', '376286187_20220809.csv', '3865498490_20220815.csv', '390755978_20220816.csv', '4081939425_20220816.csv', '4154479176_20220815.csv', '440344663_20220813.csv', '4513281267_20220817.csv', '4573741990_20220815.csv', '4647530528_20220815.csv', '4715121391_20220817.csv', '4758191482_20220815.csv', '4773591768_20220817.csv', '4799514765_20220812.csv', '4934510627_20220816.csv', '501044799_20220816.csv', '5027782038_20220816.csv', '5176739543_20220812.csv', '5265534006_20220816.csv', '5559758084_20220815.csv', '5892509075_20220816.csv', '5906483058_20220815.csv', '6074039749_20220812.csv', '6314725237_20220816.csv', '6506762788_20220815.csv', '6652616958_20220816.csv', '6764555397_20220812.csv', '6945026478_20220805.csv', '7198253621_20220816.csv', '728983901_20220817.csv', '7509475451_20220815.csv', '7711000091_20220816.csv', '7840412677_20220817.csv', '7972392719_20220812.csv', '8070693962_20220817.csv', '8483879839_20220815.csv', '8557939177_20220816.csv', '8759020784_20220812.csv', '9057819681_20220817.csv', '9152362149_20220817.csv', '9273892904_20220817.csv', '9312196920_20220816.csv', '9329966902_20220817.csv', '9348576762_20220815.csv', '9547662512_20220816.csv', '9748880425_20220815.csv']

folder = './data/marbles/learning/data/'
good_participants = [f for f in listdir(folder) if isfile(join(folder, f))]

for e3_id, e3_data_path in enumerate(good_participants):
    e3_data_path = "./data/marbles/learning/data/" + e3_data_path
    data = pd.read_csv(e3_data_path)
    data = data.tail(100)
    data["Participant ID"] = e3_id
    e3_all_data = e3_all_data.append(data, ignore_index=True)

stim_index = 0

for ind in e3_all_data.index:
    all_color_rewards = [
        [45,30,25],
        [45,35,25],
        [50,35,30],
        [50,35,25],
        [50,40,25]
    ]

    color_reward_index = int(e3_all_data['color_reward_index'][ind])
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

    if(e3_all_data['type'][ind] == 0.0):
        key_press = e3_all_data['key_press'][ind]
        changed = e3_all_data['changed'][ind]

        if(key_press != 'j' and key_press != 'k'):
            continue 

        if not changed: continue 
        
        stim_1_true_util = stimuli_mean_utilities[int(e3_all_data['stim_1'][ind])] / 10
        stim_2_true_util = stimuli_mean_utilities[int(e3_all_data['stim_2'][ind])] / 10
        new_stim_true_util = stimuli_mean_utilities[int(e3_all_data['new_stim'][ind])] / 10

        changed = e3_all_data['changed'][ind]
        change_index = e3_all_data['change_index'][ind]

        first_stim_true_util = stim_1_true_util if change_index == 0 else stim_2_true_util

        utility_error = np.abs(first_stim_true_util - new_stim_true_util) 
        #utility_error = round(utility_error, 1)

        if(key_press == 'k'):
            detect_change = 1
        else:
            detect_change = 0

        d = {"Utility Difference": utility_error, "Detect Change": detect_change, "Index": stim_index }
        changeUtility = changeUtility.append(d, ignore_index=True)
    else:
        stim_index += 1

        if(stim_index % 10 == 0):
            stim_index = 0

    
Indicies = changeUtility['Index'].unique()
#Indicies = np.array([1,2,3,4,6,7,8,9])
Slopes = np.array([])
for index in Indicies:
    data = changeUtility.loc[changeUtility['Index'] == index]
    utility_differences = data['Utility Difference'].unique()
    y = np.array([])
    for utility_difference in utility_differences:
        diff_data = data.loc[data['Utility Difference'] == utility_difference]
        y = np.append(y, diff_data['Detect Change'].mean())
    
    model = LinearRegression()
    utility_differences = np.array(utility_differences)
    utility_differences = utility_differences.reshape(-1, 1)
    reg = model.fit(utility_differences, y)

    ratio = reg.coef_[0]
    Slopes = np.append(Slopes, ratio)

    y_pred = model.predict(utility_differences)

print(Slopes)

#x = [0.7809187279151943, 0.8361344537815126, 0.8391959798994975, 0.8994252873563219, 0.8875502008032129, 0.9, 0.9366197183098591, 0.8932038834951457, 0.9459459459459459, 0.9166666666666666, 1.0]
#y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#m_x = changeUtility['Utility Difference'].to_numpy()
#m_y = changeUtility['Detect Change'].to_numpy()

model = LinearRegression()
Indicies = Indicies.reshape(-1, 1)
reg = model.fit(Indicies, Slopes)
print("reg score: ", reg.score(Indicies, Slopes))

y_pred = model.predict(Indicies)

#print(changeUtility)
#ax = sns.lineplot(x="Utility Difference", y="Detect Change", data=changeUtility)
#ax.set(xlim=(-0.5, 0.5))
#plt.errorbar(Indicies, y, yerr=e, fmt='o')
plt.scatter(Indicies,Slopes)
plt.title("Utility-Detection Ratio by Stimuli Index", fontsize=18)
plt.plot(Indicies, y_pred, color="red", linewidth=2)
plt.xlabel('Stimuli Index', fontsize=14)
plt.ylabel('Utility-Detection Ratio', fontsize=14)
plt.show()"""