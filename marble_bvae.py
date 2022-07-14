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

from sklearn.metrics import log_loss

import ast 

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

#bad_participants = ['1571956186_20220623.csv', '1624730318_20220623.csv', '1951250462_20220623.csv', '2024508277_20220622.csv', '2307452822_20220602.csv', '2616009388_20220607.csv', '282609429_20220602.csv', '2843103640_20220602.csv', '3130983426_20220602.csv', '3453325951_20220623.csv', '3545302544_20220623.csv', '3709436102_20220622.csv', '4148888000_20220622.csv', '4512223036_20220602.csv', '4737559307_20220623.csv', '4772546274_20220624.csv', '4786413128_20220623.csv', '524533978_20220602.csv', '5250442910_20220623.csv', '5525192576_20220623.csv', '5957558472_20220607.csv', '7247975301_20220623.csv', '8805493182_20220622.csv', '9927393440_20220602.csv']
bad_participants = ['1206472879_20220607.csv', '1481432285_20220607.csv', '1624730318_20220623.csv', '169075273_20220607.csv', '1951250462_20220623.csv', '282458648_20220622.csv', '2843103640_20220602.csv', '302208162_20220607.csv', '3130983426_20220602.csv', '3310926888_20220622.csv', '3525719084_20220607.csv', '3597899139_20220622.csv', '4148888000_20220622.csv', '4302545825_20220607.csv', '4309499885_20220624.csv', '4512223036_20220602.csv', '4717805082_20220622.csv', '4737559307_20220623.csv', '5138618294_20220607.csv', '5144493038_20220602.csv', '5250442910_20220623.csv', '5878990705_20220607.csv', '6174180168_20220602.csv', '6247410167_20220607.csv', '6969137467_20220622.csv', '7056217438_20220622.csv', '748797646_20220518.csv', '7729591288_20220607.csv', '8198410857_20220622.csv', '8254485902_20220623.csv', '8805493182_20220622.csv', '8894686670_20220622.csv', '9177013872_20220518.csv', '9796178986_20220623.csv', '9824877929_20220518.csv']

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
cpt_temp_params = [-5.521443644190706 ,
-17.522394061793804 ,
-7.158394456547081 ,
-6.348751541871043 ,
-5.426031085276394 ,
-100.0 ,
-10.680459348788023 ,
-23.245752553512737 ,
-5.653688961352582 ,
-24.81979536370904 ,
-10.92628105069835 ,
-5.804771450009166 ,
-6.871580527116071 ,
-6.440061778009965 ,
-14.428107671286053 ,
-6.011282807638846 ,
-6.1412767642336465 ,
-23.02386108890665 ,
-15.696493400314402 ,
-6.748016275467104 ,
-5.368445634833118 ,
-5.653998561857848 ,
-100.0 ,
-6.260384602034671 ,
-5.438388727962361 ,
-6.451492593673201 ,
-8.568106403894362 ,
-10.503131109106068 ,
-5.768116420490132 ,
-6.498009667140516 ,
-6.08214800740088 ,
-5.417031399064845 ,
-5.719929830690989 ,
-25.446743405275775 ,
-11.468184539350094 ,
-5.915116287972434 ,
-9.48582001635787 ,
-23.444241584510166 ,
-100.0 ,
-6.27514201971756 ,
-60.94197095490304 ,
-5.480394460899383 ,
-55.82872647245884 ,
-6.714592833883996 ,
-5.981473753677929 ,
-57.686540654972006 ,
-11.184241939781508 ,
-7.271878240434618 ,
-5.5715285110637245 ,
-53.21502502568324 ,
-6.975049203467461 ,
-5.412817331972419 ,
-7.121562923314136 ,
-5.95704147194027 ,
-6.307450610435845 ,
-5.9264415348833985 ,
-5.337544763264442 ,
-5.585358632337046 ,
-22.787618083311123 ,
-5.585237558798233 ,
-22.58375877076117 ,
-23.591396393995915 ,
-10.191275886627428 ,
-5.793483152440801 ,
-6.874695088298173 ,
-24.759271693756098 ,
-6.602164057661626 ,
-9.07167193289959 ,
-24.614249222843938 ,
-10.17187184230748 ,
-21.239659294786392 ,
-6.045959445966346 ,
-25.057021049826815 ,
-24.89666115996092 ,
-21.876871480593298 ,
-8.111004576160845 ,
-21.407769322098794 ,
-5.804355930273968 ,
-42.785853263408605 ,
-17.289194066968648 ,
-6.0014434114265445 ,
-7.534575251489922 ,
-23.19838067323918 ,
-5.466290093652317 ,
-6.06211406162673 ,
-5.555500515152903 ,
-5.9337648214232495 ,
-5.399402000471079 ,
-6.783211903639745 ,
-12.454698641087143 ,]

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
    general.add_argument('name', type=str,
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

all_marble_colors = pd.read_csv("./data/marbles/source/colors.csv")
all_marble_colors = all_marble_colors['colors']

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

def softmax(utilities, inv_temp=10):
    print(utilities)
    soft_percentages = []
    for utility in utilities: 
        soft_percentages.append(np.exp(utility * inv_temp) / np.sum(np.exp(utilities * inv_temp), axis=0))
    
    return soft_percentages


def main(args):
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    all_participant_data = [f for f in listdir('./data/marbles/decisions/data/') if isfile(join('./data/marbles/decisions/data/', f))]
    all_participant_data = [participant_data for participant_data in all_participant_data if participant_data not in bad_participants]
    args.img_size = get_img_size(args.dataset)

    models = []
    if(not args.is_eval_only):
        for stimuli_set in [0,1,2,3,4,5]:

            if(not os.path.exists(exp_dir + "/set" + str(stimuli_set))):
                os.makedirs(exp_dir + "/set" + str(stimuli_set))

            model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
            model = model.to(device)  # make sure trainer and viz on same device
            #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
            train_loader = get_dataloaders(args.dataset,
                                            batch_size=args.batch_size,
                                            logger=logger,
                                            set=stimuli_set)
            
            logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            loss_f = get_loss_f(args.loss,
                                n_data=len(train_loader.dataset),
                                device=device,
                                **vars(args))
            trainer = Trainer(model, optimizer, loss_f,
                                device=device,
                                logger=logger,
                                save_dir=exp_dir + "/set" + str(stimuli_set),
                                is_progress_bar=not args.no_progress_bar)
            
            #utilities = np.array(stimuli_mean_utilities)
            utilities = np.ones_like(stimuli_mean_utilities) * np.mean(stimuli_mean_utilities) #2.5733
            utilities = torch.from_numpy(utilities.astype(np.float64)).float()
            trainer(train_loader,
                    utilities=utilities, 
                    epochs=args.epochs,
                    checkpoint_every=args.checkpoint_every,)
            
            models.append(model)

            # SAVE MODEL AND EXPERIMENT INFORMATION
            save_model(trainer.model, exp_dir + "/set" + str(stimuli_set), metadata=vars(args))
    
    models = []
    for stimuli_set in [0,1,2,3,4,5]:
        model = load_model(exp_dir + "/set" + str(stimuli_set))
        model.to(device)

        #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger,
                                        set=stimuli_set)
        
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                            device=device,
                            logger=logger,
                            save_dir=exp_dir + "/set" + str(stimuli_set),
                            is_progress_bar=not args.no_progress_bar)
        
        # fit utilities to participant? 
        # LOO based? 
        utilities = np.array(stimuli_mean_utilities)
        utilities = torch.from_numpy(utilities.astype(np.float64)).float()
        """trainer(train_loader,
                utilities=utilities, 
                epochs=1,
                checkpoint_every=args.checkpoint_every,)"""

        models.append(model)
    
    participant_risk_aversions = risk_avers_parameters
    predictive_accuracy = pd.DataFrame()
    
    for participant_idx, participant_data in enumerate(all_participant_data):
        bvae_participant_accuracies = []
        meu_participant_accuracies = []
        cpt_participant_accuracies = []
        
        inv_temp = inv_temp_parameters[participant_idx]

        #if(participant_idx == 0): continue
        data = pd.read_csv(join('./data/marbles/decisions/data/', participant_data))  
        marble_set = int(data['marble_set'][0])

        participant_model = models[marble_set]

        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger,
                                        set=str(marble_set))

        stimuli = None 
        for _, stimuli in enumerate(train_loader):
            stimuli = stimuli 

        all_bvae_nll = []
        all_meu_nll = []
        all_cpt_nll = []

        #print(data.head)
        for ind in data.index:
            if(data['type'][ind] == 1.0):
                stim_1 = stimuli[int(data['stim_1'][ind])].unsqueeze(0)
                stim_1_recon , _, _, stim_1_pred_util = participant_model(stim_1)

                stim_2 = stimuli[int(data['stim_2'][ind])].unsqueeze(0)
                stim_2_recon , _, _, stim_2_pred_util = participant_model(stim_2)

                #print(stim_1_util.item(), stim_2_util.item())
                key_press = data['key_press'][ind]
                if(key_press != 'd' and key_press != 'f'):
                    break 

                pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
                bvae_softmax = np.exp(pred_utils * inv_temp) / np.sum(np.exp(pred_utils * inv_temp), axis=0)

                stim_1_true_util = stimuli_mean_utilities[int(data['stim_1'][ind])]
                stim_2_true_util = stimuli_mean_utilities[int(data['stim_2'][ind])]

                true_utils = np.array([stim_1_true_util, stim_2_true_util])
                meu_softmax = np.exp(true_utils * inv_temp) / np.sum(np.exp(true_utils * inv_temp), axis=0)
                
                risk_aversion = participant_risk_aversions[participant_idx]
                stim_1_deviation = stimuli_deviations[int(data['stim_1'][ind])]
                stim_2_deviation = stimuli_deviations[int(data['stim_2'][ind])]

                ra_stim_1 = stim_1_true_util
                ra_stim_2 = stim_2_true_util

                if(key_press == 'd'):
                    sd_diff = (stim_1_deviation - stim_2_deviation)
                    ra_stim_1 = (sd_diff * risk_aversion[0]) + risk_aversion[1]
                else:
                    sd_diff = (stim_2_deviation - stim_1_deviation)
                    ra_stim_2 = (sd_diff * risk_aversion[0]) + risk_aversion[1]

                cpt_inv_temp = cpt_temp_params[participant_idx]
                cpt_utils = np.array([ra_stim_1, ra_stim_2])
                cpt_softmax = np.exp(cpt_utils * cpt_inv_temp) / np.sum(np.exp(cpt_utils * cpt_inv_temp), axis=0)

                if(key_press == 'd'):
                    y = np.array([1,0])
                else:
                    y = np.array([0,1])

                bvae_softmax = np.array(bvae_softmax)
                meu_softmax = np.array(meu_softmax)
                cpt_softmax = np.array(cpt_softmax)
                import scipy 
                #bvae_mse = (np.square(bvae_softmax - y)).mean() #** -2
                #meu_mse = (np.square(meu_softmax - y)).mean()  #** -2
                #cpt_mse = (np.square(cpt_softmax - y)).mean() #** -2
                bvae_nll = log_loss(y,bvae_softmax )
                meu_nll = log_loss(y,meu_softmax)
                cpt_nll = log_loss(y,cpt_softmax)

                all_bvae_nll.append(bvae_nll)
                all_meu_nll.append(meu_nll)
                all_cpt_nll.append(cpt_nll)

        bvae_nlls = np.mean(all_bvae_nll)
        meu_nlls = np.mean(all_meu_nll)
        cpt_nlls = np.mean(all_cpt_nll)

        #assert(False)
        predictive_accuracy = predictive_accuracy.append({"Model Type":"B-BVAE", "Negative Log Loss":bvae_nlls, "Inverse Temperature":inv_temp, "Participant":participant_idx}, ignore_index=True)
        predictive_accuracy = predictive_accuracy.append({"Model Type":"Soft-Max", "Negative Log Loss":meu_nlls, "Inverse Temperature":inv_temp, "Participant":participant_idx}, ignore_index=True)
        predictive_accuracy = predictive_accuracy.append({"Model Type":"Risk Averse", "Negative Log Loss":cpt_nlls, "Inverse Temperature":inv_temp, "Participant":participant_idx}, ignore_index=True)
        
        bvae_participant_accuracies.append(bvae_nlls)
        meu_participant_accuracies.append(meu_nlls)
        cpt_participant_accuracies.append(cpt_nlls)

        bvae_participant_accuracies = np.array(bvae_participant_accuracies)
        #print(betas[np.argmin(bvae_participant_accuracies)], ",")

    fig, axis = plt.subplots(1, 1, sharex=True, sharey=True)
    #print(predictive_accuracy)
    #p = sns.lineplot(data=predictive_accuracy, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[1])
    #p = sns.lineplot(data=predictive_accuracy, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axis)
    p = sns.barplot(data=predictive_accuracy, x="Model Type", y="Negative Log Loss")
    p.set_title("All Trials Negative Log Loss")

    """fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #print(predictive_accuracy)
    correct_trials = predictive_accuracy.loc[predictive_accuracy["Correct"] == True]
    incorrect_trials = predictive_accuracy.loc[predictive_accuracy["Correct"] == False]

    #p = sns.lineplot(data=predictive_accuracy, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[1])
    p = sns.lineplot(data=correct_trials, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[0])
    p.set_title("Correct Trials Predictive Accuracy")
    p = sns.lineplot(data=incorrect_trials, x="Inverse Temperature", y="Predictive Accuracy", hue="Model Type", ax=axes[1])
    p.set_title("Incorrect Trials Predictive Accuracy")"""

    plt.show()

    print(predictive_accuracy)


    return 

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


