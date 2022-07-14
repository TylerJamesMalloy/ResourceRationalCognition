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

import math, random 
from models.learning.frl import Feature_RL, frl_env

from os import listdir
from os.path import isfile, join

import ast 

from sklearn.metrics import log_loss

folder = './data/marbles/decisions/data'

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

all_marble_colors = pd.read_csv("./data/marbles/source/colors.csv")
all_marble_colors = all_marble_colors['colors']

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-name', type=str, default='',
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-ut', '--utility-type',
                       default=default_config['utility'], choices=UTILTIIES,
                       help='Type of utility prediction model to use.')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-ul', '--util-loss',
                       default=default_config['util_loss'], choices=UTIL_LOSSES,
                       help="Type of Utility loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    utility = parser.add_argument_group('BetaH specific parameters')
    utility.add_argument('-u', '--upsilon', type=float,
                       default=default_config['upsilon'],
                       help="Weight of the utility loss parameter.")
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Eval options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')
    
    modelling = parser.add_argument_group('L&DM modelling specific options')
    modelling.add_argument('--model-epochs', type=int,
                            default=default_config['model_epochs'],
                            help='Number of epochs to train utility prediction model.')
    modelling.add_argument('--trial-update', type=str,
                            default=default_config['trial_update'],
                            help='Source for util predictions.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

inv_temp_parameters =  [-0.2575112867908996 ,
                        -2.2316558956665764 ,
                        -5.186854231116182 ,
                        16.576962771293054 ,
                        -1.1385845063598818 ,
                        -0.9708091678381253 ,
                        1.5671358002718052 ,
                        1.1694895858816199 ,
                        0.25005729120830034 ,
                        4.863321299174678 ,
                        77.06880187178643 ,
                        6.005973940078301 ,
                        -1.0243222676842956 ,
                        3.430516661003277 ,
                        -0.8145839558815853 ,
                        0.6908955550163012 ,
                        -0.29327469054392813 ,
                        -0.09942917992611403 ,
                        1.0665258624593519 ,
                        2.0137451836533873 ,
                        -0.21702678551100696 ,
                        0.38964344235419796 ,
                        1.2509992461514603 ,
                        0.36051773284416994 ,
                        -0.9344195915020732 ,
                        12.394742704309065 ,
                        0.2403743648066365 ,
                        1.6943113551051472 ,
                        -0.44619696366180134 ,
                        -1.8270366888393013 ,
                        -0.08614032545515503 ,
                        1.126914425825868 ,
                        -0.0001404904421354179 ,
                        -1.5839329621529532 ,
                        0.8411213385123634 ,
                        1.246175573463317 ,
                        1.4610917576066156 ,
                        -0.33417248950156486 ,
                        75.14497662798708 ,
                        -0.20254957554789293 ,
                        6.32023498038291 ,
                        -0.8113226039819986 ,
                        1.2728878209539323 ,
                        13.433366837433597 ,
                        2.6308846856634287 ,
                        0.8116295648568991 ,
                        -0.41790244857941283 ,
                        3.932503134264196 ,
                        -0.19759490042920347 ,
                        6.253853149735417 ,
                        1.9755056871310381 ,
                        -0.9682972816359624 ,
                        0.6446522750454636 ,
                        2.855165091370026 ,
                        -0.5644718327448696 ,
                        3.0711001989730335 ,
                        -4.6473810144316126 ,
                        -1.6184754105152273 ,
                        0.17823417639402547 ,
                        0.8522623736946785 ,
                        -0.3758940205967951 ,
                        -1.9193385053167125 ,
                        -1.0175445179458777 ,
                        0.26814990390833504 ,
                        -0.3621376184830901 ,
                        1.2050606110934732 ,
                        0.17486271199340692 ,
                        1.8565338602925767 ,
                        -1.6902630340806268 ,
                        2.736803668149758 ,
                        0.27761511183370813 ,
                        -0.8847999076802705 ,
                        73.6665917702792 ,
                        -0.48902307058143274 ,
                        0.4760220789309887 ,
                        0.8423302662505625 ,
                        -0.41320878371110537 ,
                        0.5818122591987652 ,
                        9.067271589620328 ,
                        -0.144058210978376 ,
                        0.7475392001385524 ,
                        0.29887520994021977 ,
                        0.44044413827097384 ,
                        0.14294643233309406 ,
                        12.670148966854338 ,
                        -0.42069356510322303 ,
                        0.13016837277807267 ,
                        0.22935461705688798 ,
                        3.4326372127554796 ,
                        1.3790356837843694 ,]


risk_avers_parameters = [[0.9399480046188267, 0.02318066505794461], [1.1340789837217646, 0.008589519327917631], [1.098409835879281, 0.028991960870179752], [0.6454982000388635, 0.1779177023947266], [1.1477664199420459, -0.006000430734655209], [1.0801113646455764, -0.0131667034470643], [1.2217398489505558, 0.018898151990513885], [1.0807037820922973, -0.01874552702602712], [1.1002546065471732, 0.029369841513125293], [1.0420637861322992, 0.03745078910436088], [0.957847568346029, 0.021201852877554814], [1.1663713030999163, 0.07938391351082481], [1.1728569004691076, -0.03121323603504677], [1.1203616209433724, 0.03164579335400004], [1.2425435774566647, 0.0006423677541024615], [1.2789886543183215, 0.012345238984213136], [0.9080668413241316, 0.09451913940512033], [1.2824468672276803, 0.015254289813540297], [1.166594935263976, 0.005539764663007412], [1.0955881733070132, 0.05934017813630768], [1.1790157330506106, -0.02850705578495849], [1.1889421310438082, -0.011736558903802236], [0.985979495684729, 0.03668712327840537], [1.0476985448915668, -0.028540870780456577], [1.2697191050845116, 0.02208786275603905], [0.8888594762735648, 0.14698888605634366], [1.261689900374983, -0.008479742657989411], [1.1249461687422304, -0.0010209104830559037], [1.1942454812420333, -0.0022609570063297406], [1.2070938495599253, 0.003342496792484523], [1.126585291882074, 0.05995335884922496], [0.9940111907548609, 0.0035360883444547006], [1.1404024755310804, -0.0061512490732070005], [1.0463035893965609, -0.024683309487359563], [1.0655223237761895, 0.04517644743467278], [1.1499689328684886, -0.007013320838842655], [1.2962054072042375, 0.010253104140700988], [1.2183832290790195, -0.014029577310886429], [1.1273899654105608, 0.037334266063839536], [0.9910230722806391, -0.00020769675737293547], [1.3680423274735378, -0.003965514336817272], [1.1346861790404594, -0.026128004097889113], [1.1869256821968313, -0.001195497579557607], [1.0870489114182178, 0.038791464152752156], [1.2032132513200942, -0.01987788676817746], [1.406807844715617, 0.003809148112399144], [1.2342210362509207, 0.003612062340855081], [1.0873927373094932, 0.03331889106168939], [1.1203442709701767, 0.003745652900797449], [0.7930040863419598, 0.12952722948833734], [1.2516495751955097, 0.06302633192602625], [1.1524062855451616, 0.010510798122063], [1.1747890171694195, -0.030918205033862937], [1.0776419835767403, 0.029957484498909635], [1.1549847283266648, 0.016573370407106137], [1.1545714375353389, 0.025046673117890033], [1.1651585475025905, -0.033995055318493034], [1.211807138820804, -0.004515886714531882], [1.0595375744681033, -0.0030556591374441625], [1.1283647378742327, 0.03634674909157138], [1.0860060959826283, 0.02004179105273078], [1.1513957648454083, -0.03469838021347773], [1.1137331911564066, -0.03131801441683289], [1.091014266272218, -0.00695737581358773], [1.2683454057805992, -0.0393964442555102], [1.1243383207421263, -0.016561644378741344], [1.0528156634167818, 0.023214147004346434], [1.1881780048293145, -0.0026541438269383003], [0.9781364608484975, -0.015798666000560558], [1.005037416708196, -0.0363479845405309], [1.2915917294869266, 0.015638419837192247], [1.1638936306434662, -0.016879419506242554], [1.261667856367497, 0.028650879077561253], [0.9470816551848582, 0.02368175594459898], [1.102588669119483, -0.01646148270094508], [1.1283561387859236, -0.0186609887112677], [1.2000410031378725, 0.0007757267566123427], [1.0916549390732526, -0.006652216385118186], [1.0680064325837837, 0.02407717153640837], [1.2100247289760404, 0.06298675045151204], [1.0690996405700186, 0.053690120497248986], [1.0428238520515722, 0.02700928456266011], [1.1244747094693464, 0.007218399573029609], [1.231219008133787, 0.01522547855488256], [0.6778353449901994, 0.14581985369051098], [1.1707859725846455, -0.02334817457639639], [1.12507099466307, -0.008489439504817588], [1.0267810850636334, -0.03689800834513646], [1.3695039991465847, 0.054397120339813254], [1.1582517374689003, 0.008501363754372217]]

participant_betas = [   9.999964803157473  ,
                        1.0  ,
                        1.0  ,
                        2.146012958042111  ,
                        1.0  ,
                        1.0  ,
                        17.59668978193483  ,
                        1.0  ,
                        1.0  ,
                        14.694104696461123  ,
                        1.0  ,
                        1.5736292172689639  ,
                        1.0859867736639592  ,
                        1.0  ,
                        4.637138362895814  ,
                        42.24084642246553  ,
                        7.092098582847013  ,
                        14.38619957832277  ,
                        1.0  ,
                        2.0809270187566606  ,
                        4.712479688010844  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        1.0495445521203837  ,
                        13.490183194926093  ,
                        2.158480920919687  ,
                        1.0  ,
                        1.0  ,
                        13.52441117108075  ,
                        1.2417550730590285  ,
                        1.318418322007665  ,
                        1.0  ,
                        1.0  ,
                        35.52925224097744  ,
                        1.0  ,
                        1.7205410890332349  ,
                        1.0  ,
                        2.007358336016597  ,
                        1.0  ,
                        12.076933128573973  ,
                        4.764648385299535  ,
                        1.0  ,
                        1.1027739073609935  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        1.2583529284148522  ,
                        11.183712697990721  ,
                        1.0  ,
                        1.401900244212472  ,
                        1.0  ,
                        41.57032647926863  ,
                        1.0  ,
                        6.497087868187239  ,
                        13.45184642345969  ,
                        10.800068796421138  ,
                        1.0  ,
                        1.0  ,
                        1.3049788135877023  ,
                        11.636985806584097  ,
                        2.7460702219620297  ,
                        5.889140331327847  ,
                        1.0  ,
                        1.0  ,
                        6.477520534710044  ,
                        1.0  ,
                        6.532013888619628  ,
                        2.185416823914068  ,
                        1.939248602627728  ,
                        1.0  ,
                        35.73101893288693  ,
                        1.0  ,
                        2.8728916630146264  ,
                        1.0  ,
                        3.3969960040381313  ,
                        1.0  ,
                        1.0  ,
                        40.267790218459695  ,
                        1.0  ,
                        8.679607989502918  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        18.406204931919888  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        33.35430698629798  ,
                        1.177903337822979  ,
                        1.0  ,
                        2.141981463988113  ,
                        1.0  ,
                        2.5583678120937554  ,
                        7.316990757334835  ,
                        37.057970737343865  ,
                        1.0  ,
                        1.0  ,
                        41.084588037307725  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        13.765987985838047  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        6.596115982353638  ,
                        1.0  ,
                        6.214670875045485  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        1.0  ,
                        40.62284195728592  ,
                        1.0  ,
                        1.0  ,
                        10.19469450481117  ,
                        10.598942754207464  ,
                        1.0  ,
                        2.920202202079896  ,
                        1.6587400700449888  ,]

stimuli_mean_utilities = []
stimuli_deviations = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))


def log_loss_score(predicted, actual, eps=1e-14):
        """
        :param predicted:   The predicted probabilities as floats between 0-1
        :param actual:      The binary labels. Either 0 or 1.
        :param eps:         Log(0) is equal to infinity, so we need to offset our predicted values slightly by eps from 0 or 1
        :return:            The logarithmic loss between between the predicted probability assigned to the possible outcomes for item i, and the actual outcome.
        """
        predicted = np.clip(predicted, eps, 1-eps)
        loss = -1 * np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1-predicted))

        return loss

def cpt_mse(inv_temp, participant_data, risk_aversion):
    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)
    all_meu_mse = []

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            if(key_press != 'd' and key_press != 'f'):
                continue 

            stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
            stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

            stim_1_deviation = stimuli_deviations[int(data['stim_1'][ind])]
            stim_2_deviation = stimuli_deviations[int(data['stim_2'][ind])]

            #ra_stim_1 = stim_1_true_util + (1/risk_aversion[0] * (stim_1_deviation - stim_2_deviation) - risk_aversion[1])
            #ra_stim_2 = stim_2_true_util + (1/risk_aversion[0] * (stim_2_deviation - stim_1_deviation) - risk_aversion[1])
            ra_stim_1 = stim_1_true_util
            ra_stim_2 = stim_2_true_util
            # Update diff to be equal to ((stim_1_true_util - stim_2_true_util) / risk_aversion[0]) + risk_aversion[1]
            ra_stim_1 = stim_1_true_util + 0.5*(((stim_1_true_util - stim_2_true_util) / risk_aversion[0]) + risk_aversion[1])
            ra_stim_2 = stim_1_true_util - 0.5*(((stim_1_true_util - stim_2_true_util) / risk_aversion[0]) + risk_aversion[1])

            #true_utils = np.array([stim_1_true_util, stim_2_true_util])
            ra_utils = np.array([ra_stim_1, ra_stim_2])
            meu_softmax = np.exp(ra_utils / inv_temp) / np.sum(np.exp(ra_utils / inv_temp), axis=0)

            true_utils = np.array([stim_1_true_util, stim_2_true_util])

            original_softmax = np.exp(true_utils / inv_temp) / np.sum(np.exp(true_utils / inv_temp), axis=0)
            
            if(key_press == 'd'):
                y = np.array([1,0])
            else:
                y = np.array([0,1])

            meu_softmax = np.array(meu_softmax)
            meu_mse = (np.square(meu_softmax - y)).mean() 
            meu_mse = log_loss(y, meu_softmax)
            #meu_mse = log_loss(y, [0.5,0.5])
            all_meu_mse.append(meu_mse)

    print(all_meu_mse)
    print(np.sort(all_meu_mse))
    print(np.mean(np.sort(all_meu_mse)))

    _, bins, _ = plt.hist(all_meu_mse, 50, density=1, alpha=0.5)
    plt.show()

    assert(False)
    meu_mmse = np.mean(all_meu_mse)
    return(meu_mmse)

bad_participants = ['1206472879_20220607.csv', '1481432285_20220607.csv', '1624730318_20220623.csv', '1655417347_20220607.csv', '169075273_20220607.csv', '1951250462_20220623.csv', '2824017091_20220701.csv', '282458648_20220622.csv', '2843103640_20220602.csv', '302208162_20220607.csv', '3050445599_20220701.csv', '3072452500_20220623.csv', '3130983426_20220602.csv', '3202767511_20220602.csv', '3310926888_20220622.csv', '3525719084_20220607.csv', '3597899139_20220622.csv', '3774486973_20220702.csv', '4148888000_20220622.csv', '4302545825_20220607.csv', '4309499885_20220624.csv', '4384161035_20220701.csv', '4424042522_20220623.csv', '4512223036_20220602.csv', '4717805082_20220622.csv', '4737559307_20220623.csv', '4833293935_20220607.csv', '5138618294_20220607.csv', '5144493038_20220602.csv', '5250442910_20220623.csv', '5552993317_20220602.csv', '5878990705_20220607.csv', '5957558472_20220607.csv', '6174180168_20220602.csv', '6176365135_20220602.csv', '6247410167_20220607.csv', '640250500_20220701.csv', '6969137467_20220622.csv', '7056217438_20220622.csv', '7351329913_20220701.csv', '748797646_20220518.csv', '7686756703_20220701.csv', '7708514735_20220701.csv', '7729591288_20220607.csv', '8198410857_20220622.csv', '8254485902_20220623.csv', '8305923582_20220623.csv', '8805493182_20220622.csv', '8880742555_20220623.csv', '8894686670_20220622.csv', '9162481065_20220607.csv', '9177013872_20220518.csv', '9412783563_20220607.csv', '9462710583_20220701.csv', '9796178986_20220623.csv', '9824877929_20220518.csv', '9854752339_20220623.csv', '9876575671_20220623.csv']

def bvae_mse(optim_args, participant_data):
    beta = optim_args[0]
    inv_temp = optim_args[1]

    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = None

    data = pd.read_csv(join(folder, participant_data))  
    args.img_size = get_img_size(args.dataset)

    stimuli_set = int(data['marble_set'][0])

    #model = load_model(exp_dir + "/set" + str(stimuli_set))
    model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
    model.to(device)
    # set beta and  upsilon
    #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=logger,
                                    set=stimuli_set)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #print("setting beta: ", beta)
    setattr(args, 'betaH_B', beta)
    #setattr(args, 'upsilon', 1e6)
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    
    trainer = Trainer(model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir + "/set" + str(stimuli_set),
                        is_progress_bar=not args.no_progress_bar)
    
    utilities = np.array(stimuli_mean_utilities)
    utilities = torch.from_numpy(utilities.astype(np.float64)).float()
    trainer(train_loader,
            utilities=utilities, 
            epochs=200,
            checkpoint_every=1000000)

    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli 
    
    pred_y = []
    true_y = []
    log_losses = []

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            if(key_press != 'd' and key_press != 'f'):
                continue 
            
            stim_1 = stimuli[int(data['stim_1'][ind])].unsqueeze(0)
            stim_1_recon , _, _, stim_1_pred_util = model(stim_1)

            stim_2 = stimuli[int(data['stim_2'][ind])].unsqueeze(0)
            stim_2_recon , _, _, stim_2_pred_util = model(stim_2)
        
            pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
            bvae_softmax = np.exp(pred_utils / inv_temp) / np.sum(np.exp(pred_utils / inv_temp), axis=0)

            stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
            stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

            #print("predicted stim util: ", stim_1_pred_util.item(), " true util is: ", stim_1_true_util)

            if(key_press == 'd'):
                y = np.array([1,0])
            else:
                y = np.array([0,1])

            #log_losses.append(log_loss(y, bvae_softmax))
            pred_y.append(bvae_softmax)
            true_y.append(y)

    return(log_loss(true_y, pred_y))

def color_pred(inv_temp, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)
    all_meu_mse = []

    pred_y = []
    true_y = []

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            if(key_press != 'd' and key_press != 'f'):
                continue 
                
            stim_1 = int(data['stim_1'][ind])
            stim_2 = int(data['stim_2'][ind])

            stim_1_colors = all_marble_colors[stim_1]
            stim_2_colors = all_marble_colors[stim_2]

            stim_1_colors = np.array(ast.literal_eval(stim_1_colors))
            stim_2_colors = np.array(ast.literal_eval(stim_2_colors))

            stim_1_values = np.select([stim_1_colors == 2, stim_1_colors == 1, stim_1_colors == 0], [2, 3, 4], stim_1_colors)
            stim_2_values = np.select([stim_2_colors == 2, stim_2_colors == 1, stim_2_colors == 0], [2, 3, 4], stim_2_colors)

            stim_1_count_high_value = np.sum(stim_1_values == 4)
            stim_2_count_high_value = np.sum(stim_2_values == 4)

            colors = [stim_1_count_high_value, stim_2_count_high_value]

            color_softmax = np.exp(colors / inv_temp) / np.sum(np.exp(colors / inv_temp), axis=0)

            if(key_press == 'd'):
                y = np.array([1,0])
            else:
                y = np.array([0,1])
            
            pred_y.append(color_softmax)
            true_y.append(y)

    meu_mse = log_loss(true_y, pred_y)
    return(meu_mse)

def get_mse(inv_temp, participant_data):
    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)
    all_meu_mse = []
    pred_y = []
    true_y = []
    predictive_accuracy = []

    correct = 0
    incorrect = 0

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            if(key_press != 'd' and key_press != 'f'):
                continue 

            stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
            stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

            true_utils = np.array([stim_1_true_util, stim_2_true_util])
            meu_softmax = np.exp(true_utils * inv_temp) / np.sum(np.exp(true_utils * inv_temp), axis=0)

            if(key_press == 'd'):
                y = np.array([1,0])
                predictive_accuracy.append(meu_softmax[1])
            else:
                y = np.array([0,1])
                predictive_accuracy.append(meu_softmax[0])
            
            if((key_press == 'd' and stim_1_true_util >= stim_2_true_util) or (key_press == 'f' and stim_1_true_util <= stim_2_true_util) ):
                correct += 1
            else:
                incorrect += 1

            meu_softmax = np.array(meu_softmax)
            meu_mse = log_loss(y, meu_softmax)

            

            pred_y.append(meu_softmax)
            true_y.append(y)

            all_meu_mse.append(meu_mse)
    
    # change to maximize predictive accuracy 

    if(len(all_meu_mse) == 0):
        print(participant_data)
    meu_mse = log_loss(true_y, pred_y)
    
    return(np.mean(predictive_accuracy))

def get_cpt(vars, participant_data):
    inv_temp = vars[0]
    cpt_balance = vars[1]

    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)
    all_meu_mse = []
    pred_y = []
    true_y = []

    correct = 0
    incorrect = 0

    for ind in data.index:
        if(data['type'][ind] == 1.0):
            key_press = data['key_press'][ind]
            if(key_press != 'd' and key_press != 'f'):
                continue 

            stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
            stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

            stim_1_true_std = stimuli_deviations[int(data['stim_1'][ind])]
            stim_2_true_std = stimuli_deviations[int(data['stim_2'][ind])]

            true_utils = np.array([stim_1_true_util - (cpt_balance * stim_1_true_std), stim_2_true_util - (cpt_balance * stim_2_true_std)])
            meu_softmax = np.exp(true_utils / inv_temp) / np.sum(np.exp(true_utils / inv_temp), axis=0)

            if(key_press == 'd'):
                y = np.array([1,0])
            else:
                y = np.array([0,1])
            
            if((key_press == 'd' and stim_1_true_util >= stim_2_true_util) or (key_press == 'f' and stim_1_true_util <= stim_2_true_util) ):
                correct += 1
            else:
                incorrect += 1

            meu_softmax = np.array(meu_softmax)
            #meu_softmax = [0.5, 0.5]
            meu_mse = log_loss(y, meu_softmax)

            pred_y.append(meu_softmax)
            true_y.append(y)

            all_meu_mse.append(meu_mse)

    if(len(all_meu_mse) == 0):
        print(participant_data)
    meu_mse = log_loss(true_y, pred_y)
    return(meu_mse)

def performacne(participant_data): 
    data = pd.read_csv(join(folder, participant_data))  
    data = data.tail(200)
    all_meu_mse = []

    correct = 0
    incorrect = 0

    for ind in data.index:
        if(data['type'][ind] == 0.0):
            key_press = data['key_press'][ind]
            if(key_press != 'j' and key_press != 'k'):
                continue 

            changed = int(data['changed'][ind])
            
            if((key_press == 'j' and changed) or (key_press == 'k' and not changed) ):
                correct += 1
            else:
                incorrect += 1
    
    if((correct + incorrect) == 0): return 0 
    return(correct / (correct + incorrect))

import scipy 

def main(args):
    all_participant_data = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_participant_data = [participant_data for participant_data in all_participant_data if participant_data not in bad_participants]
    loss = []
    for participant_id, participant_data in enumerate(all_participant_data):
        #res = scipy.optimize.minimize(color_pred, (10), args=(participant_data), bounds=((-1000, 1000),))
        res = scipy.optimize.minimize(get_mse, (10), args=(participant_data), bounds=((1e-6, 100),))
        #res = scipy.optimize.minimize(get_cpt, (10, 0.5), args=(participant_data), bounds=((-100, 100),(-100, 100)))
        #res = scipy.optimize.minimize(cpt_mse, (10), args=(participant_data, risk_avers_parameters[participant_id]), bounds=((-100, 100),))
        #res = scipy.optimize.minimize(bvae_mse, (10, 10), args=(participant_data), bounds=((0, 1000),(-100, 100)))
        print(res['x'], ",")
        print(res['fun'])
        
        if(res['fun'] > 0.7):
            loss.append(res['fun'])

    print(np.mean(loss))

    loss = np.array(loss)
    _, bins, _ = plt.hist(loss, 50, density=1, alpha=0.5)
    plt.show()

    return 

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


# python .\inv_temp_fit.py marbles/decisions/bvae_ld9_b100 --is-eval-only