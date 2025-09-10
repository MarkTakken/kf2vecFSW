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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import sklearn
from sklearn.metrics import accuracy_score



import sys
import os
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
import subprocess
from io import StringIO
import csv





# Hyper-parameters
features_scaler = 1e4



def classify_func(features_folder, feature_input_file_list, model_file, seed, classification_result, block_size):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    since = time.time()

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(classification_result, 'classification.log'), 'w+'),
                logging.StreamHandler()]
    # logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s: %(message)s', handlers=handlers)
    logging.basicConfig(level=level, format=format, handlers=handlers)

    #######################################################################
    logging.info('\n==> Input arguments...\n')

    logging.info('Feature directory: {}'.format(features_folder))
    logging.info('Model: {}'.format(model_file))
    logging.info('Seed: {}'.format(seed))
    #######################################################################
    # Model
    logging.info('\n==> Building model...\n')

    # Load model
    state = torch.load(os.path.join(model_file, "classifier_model.ckpt"))
    input_size = state["model_input_size"]
    class_count = state["model_class_count"]

    model = models.NeuralNetClassifierOnly(input_size, state["model_hidden_size_fc1"], class_count)
    model.load_state_dict(state['state_dict'])
    model.to("cpu")
    model.eval()

    # Prepare output file
    classes_fname = os.path.join(classification_result, "classes.out")
    if os.path.exists(classes_fname):
        os.remove(classes_fname)
    header = ["genome", "top_class", "top_p"] + [str(x) for x in range(class_count)]
    with open(classes_fname, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(header)


    # Process files in blocks
    for z in range(0, len(feature_input_file_list), block_size):
        chunk_files = feature_input_file_list[z:z + block_size]

        # Read all files in the chunk
        lines = []
        for f in chunk_files:
            with open(f) as ff:
                lines.extend(ff)
        df = pd.read_csv(StringIO("".join(lines)), header=None)
        df.set_index(0, inplace=True)

        X = df.values * features_scaler
        X_tensor = torch.from_numpy(X).float()

        # Forward pass
        ps = torch.exp(model(X_tensor))
        top_p, top_class = ps.topk(1, dim=1)

        # Write results directly
        genomes_np = df.index.values[:, None]
        results = results = np.hstack([genomes_np, top_class.detach().numpy(),top_p.detach().numpy(), ps.detach().numpy()])
        with open(classes_fname, "a", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(results)

    time_elapsed = time.time() - since
    logging.info('\n==> Classification Completed!\n')
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))




