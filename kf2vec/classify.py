# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Fixed bug associated with cdist on gpu
# Fixed bug in torch.index_select for subsetting lambdas

# Uses feature extracted from 10K dataset
# Kmer counts computed for >80K large contigs and <=80K chimeric contigs separately


import time
import itertools
import logging
import re
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import sklearn
from sklearn.metrics import accuracy_score



import sys
import math
import copy
from . import models
from . import datasets
from . import losses
from . import parameter_inits
from . import utils
from .utils import *
from . import weight_inits
from .weight_inits import *





# Hyper-parameters
features_scaler = 1e4



def classify_func(features_folder, feature_input_file_list, model_file, seed, classification_result):

    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)


    since = time.time()

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(classification_result, 'classification.log'), 'w+'), logging.StreamHandler()]

    #logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s: %(message)s', handlers=handlers)

    logging.basicConfig(level=level, format=format, handlers=handlers)
    # logging.info('Hey, this is working!')


    #######################################################################
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #######################################################################
    logging.info('\n==> Input arguments...\n')


    logging.info('Feature directory: {}'.format(features_folder))
    logging.info('Model: {}'.format(model_file))
    logging.info('Seed: {}'.format(seed))


    #######################################################################
    # Prepare dataset
    """
    logging.info('\n==> Preparing Data...\n')

    feature_input = feature_input.iloc[:,:]*features_scaler
    input_size = np.shape(feature_input)[1]
    logging.info("Dimensions of feature matrix rows: {}, cols: {}".format(np.shape(feature_input)[0], np.shape(feature_input)[1]))
    """

    # #######################################################################
    # Get names
    # backbone_names = feature_input.index.tolist()

    #######################################################################
    # Model
    logging.info('\n==> Building model...\n')


    ##### Load model #####
    state = torch.load(os.path.join(model_file, "classifier_model.ckpt"))


    input_size = state["model_input_size"]
    hidden_size_fc1 = state["model_hidden_size_fc1"]
    class_count = state["model_class_count"]

    # Need to find a way to save model name as well
    model = models.NeuralNetClassifierOnly(input_size, hidden_size_fc1, class_count)

    model.load_state_dict(state['state_dict'])
    # model.to(device)
    model.to("cpu")


    logging.info('Number of Classes: {}'.format(class_count))

    #######################################################################
    # Prepare input data
    logging.info('\n==> Preparing Input Data...\n')

    classes_fname = "classes.out"
    if os.path.isfile(os.path.join(classification_result, classes_fname)):
        os.remove(os.path.join(classification_result, classes_fname))


    classes_header = ["genome", "top_class", "top_p"] + [str(x) for x in list(range(class_count))]
    df_classes_header = pd.DataFrame(columns=classes_header)
    df_classes_header.to_csv(os.path.join(classification_result, classes_fname), index=False, sep='\t', mode = "a")

    """
    #######################################################################
    logging.info('\n==> Model parameters----------')
    # for parameter in model.parameters():
    #     logging.info(parameter.shape)

    for name, param in model.named_parameters():
        logging.info("{} : {}".format(name, param.shape))

    # list(model.parameters())[0].grad

    # Total number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info("Total parameters: {}".format(pytorch_total_params))

    # Total number of trainable parameters
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable parameters: {}".format(pytorch_trainable_params))
    """

    #######################################################################
    # Training model
    logging.info('\n==> Compute model output...\n')

    # Compute model output
    model.eval()

    with torch.no_grad():

        for feature_input_file in feature_input_file_list:

            feature_input = pd.read_csv(feature_input_file, index_col=None, header=None, sep=',')
            feature_input.set_index(0, inplace=True)
            feature_input = feature_input.iloc[:, :] * features_scaler


            # Apply mask. Deducing k value
            if input_size == 7812:  # k = 7
                my_alphabet_kmers = pd.read_csv("test_kmers_7_sorted", sep=" ", header=None, names=["kmer"])
                logging.info('Masking: {}, k={}'.format("True", 7))

                my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))
                feature_input = feature_input.iloc[:, [z for z in range(0, len(my_mask)) if my_mask[z] == True]]

            elif input_size == 1884:  # k = 6
                my_alphabet_kmers = pd.read_csv("test_kmers_6_sorted", sep=" ", header=None, names=["kmer"])
                logging.info('Masking: {}, k={}'.format("True", 6))

                my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))
                feature_input = feature_input.iloc[:, [z for z in range(0, len(my_mask)) if my_mask[z] == True]]

            elif input_size == 12:  # k = 3
                my_alphabet_kmers = pd.read_csv("vocab_generator_k3C_fin.fa", sep=" ", header=None, names=["kmer"])
                logging.info('Masking: {}, k={}'.format("True", 3))

                my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))
                feature_input = feature_input.iloc[:, [z for z in range(0, len(my_mask)) if my_mask[z] == True]]

            elif input_size == 88:  # k = 4
                my_alphabet_kmers = pd.read_csv("vocab_generator_k4C_fin.fa", sep=" ", header=None, names=["kmer"])
                logging.info('Masking: {}, k={}'.format("True", 4))

                my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))
                feature_input = feature_input.iloc[:, [z for z in range(0, len(my_mask)) if my_mask[z] == True]]

            elif input_size == 420:  # k = 5
                my_alphabet_kmers = pd.read_csv("vocab_generator_k5C_fin.fa", sep=" ", header=None, names=["kmer"])
                logging.info('Masking: {}, k={}'.format("True", 5))

                my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))
                feature_input = feature_input.iloc[:, [z for z in range(0, len(my_mask)) if my_mask[z] == True]]

            elif input_size == 32116:  # k = 8
                my_alphabet_kmers = pd.read_csv("vocab_generator_k8C_fin.fa", sep=" ", header=None, names=["kmer"])
                logging.info('Masking: {}, k={}'.format("True", 8))

                my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))
                feature_input = feature_input.iloc[:, [z for z in range(0, len(my_mask)) if my_mask[z] == True]]

            elif input_size == 129540:  # k = 9
                my_alphabet_kmers = pd.read_csv("vocab_generator_k9C_fin.fa", sep=" ", header=None, names=["kmer"])
                logging.info('Masking: {}, k={}'.format("True", 9))

                my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))
                feature_input = feature_input.iloc[:, [z for z in range(0, len(my_mask)) if my_mask[z] == True]]



            model_class = model(torch.from_numpy(feature_input.values).float())


            ps = torch.exp(model_class)
            top_p, top_class = ps.topk(1, dim=1)

            # Get names
            backbone_names = feature_input.index.tolist()
            #print(backbone_names)


            #######################################################################
            # Compute distance matrix for single query

            # Detach gradient and convert to numpy
            df_classes = pd.DataFrame(np.hstack((top_class.detach().numpy(), top_p.detach().numpy(), ps.detach().numpy())))


            # Attach species names
            df_classes.columns = ["top_class", "top_p"] + [str(x) for x in list(range(class_count))]
            df_classes.insert(loc=0, column='genome', value=backbone_names)


            #######################################################################
            # Write to file (append to  classes.out file)
            df_classes.to_csv(os.path.join(classification_result, classes_fname), index=False, sep='\t', mode = "a", header = False)

            #######################################################################


    logging.info('\n==> Classification Completed!\n')

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))







