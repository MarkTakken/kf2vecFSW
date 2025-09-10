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


def query_func(features_folder, features_csv_file_list, model_file, classes, seed, output_folder, remap, block_size):

    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    since = time.time()

    # logging setup
    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(output_folder, 'query_run.log'), 'w+'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format, handlers=handlers)

    #######################################################################
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #######################################################################
    logging.info('\n==> Input arguments...\n')
    logging.info('Query directory: {}'.format(features_folder))
    logging.info('Model directory: {}'.format(model_file))
    logging.info('Class information: {}'.format(classes))
    logging.info('Seed: {}'.format(seed))

    #######################################################################
    # Read classification information
    logging.info('\n==> Querying...\n')

    classes_fname = "classes.out"
    classification_df_all = pd.read_csv(os.path.join(classes, classes_fname), sep="\t", header=0)

    features_csv_basename = [os.path.basename(q).split(".kf")[0] for q in features_csv_file_list]
    classification_df = classification_df_all.loc[classification_df_all.genome.isin(features_csv_basename)]

    classification_df["top_class"] = classification_df["top_class"].astype(int)
    class_count = classification_df.top_class.unique()

    logging.info('Total subtrees to query: {}'.format(classification_df.top_class.unique().size))

    #######################################################################
    # Optional label remap dict
    remap_dict = None
    if remap:
        try:
            my_map_df = pd.read_csv(remap, sep="\t", header=0)
            remap_dict = pd.Series(my_map_df.new_label.values, index=my_map_df.label).to_dict()
            logging.info('Remap loaded: {} entries'.format(len(remap_dict)))
        except Exception as e:
            logging.warning('Could not read remap file {}: {}'.format(remap, e))
            remap_dict = None

    #######################################################################
    # Process each clade
    for c in class_count:
        current_clade = classification_df.loc[classification_df["top_class"] == c]
        contig_ids = current_clade["genome"].to_list()

        if not contig_ids:
            continue

        logging.info('\n==> Working on subtree {} ({} contigs)...\n'.format(c, len(contig_ids)))

        # Load trained model for this clade
        state = torch.load(os.path.join(model_file, "model_subtree_{}.ckpt".format(c)))
        input_size = state["model_input_size"]
        hidden_size_fc1 = state["model_hidden_size_fc1"]
        embedding_size = state["model_embedding_size"]

        model = models.NeuralNet(input_size, hidden_size_fc1, embedding_size)
        model.load_state_dict(state['state_dict'])
        model.to("cpu")
        model.eval()

        # Load backbone embeddings for distance computation
        df_embeddings = pd.read_csv(
            os.path.join(model_file, 'embeddings_subtree_{}.csv'.format(c)),
            sep="\t", header=None, index_col=0
        )
        embeddings_tensor = torch.from_numpy(df_embeddings.values).float()
        backbone_names = df_embeddings.index.tolist()

        # Open output files once per clade (fast streaming writes)
        dist_path = os.path.join(output_folder, f"apples_input_di_mtrx_subtree_{c}.csv")
        emb_path = os.path.join(output_folder, f"embedding_subtree_{c}.emb")

        with open(dist_path, "w", newline="") as f_dist, open(emb_path, "w", newline="") as f_emb:
            dist_writer = csv.writer(f_dist, delimiter="\t")
            emb_writer = csv.writer(f_emb, delimiter="\t")

            # Distance matrix header: first cell empty, then backbone species labels
            dist_writer.writerow([""] + backbone_names)

            with torch.no_grad():
                for z in range(0, len(contig_ids), block_size):
                    chunk_files = contig_ids[z:z + block_size]

                    # Bulk-read features via `cat` for speed
                    cat_filelist = [os.path.join(features_folder, f + ".kf") for f in chunk_files]
                    cat_result = subprocess.run(["cat"] + cat_filelist, capture_output=True, text=True)
                    feature_input = pd.read_csv(StringIO(cat_result.stdout), index_col=None, header=None, sep=',')

                    # Set index to query label, scale features
                    feature_input.set_index(0, inplace=True)
                    feature_input = feature_input.iloc[:, :] * features_scaler

                    # Query labels (apply remap if provided)
                    query_labels = feature_input.index.tolist()
                    if remap_dict is not None:
                        query_labels = [remap_dict.get(q, q) for q in query_labels]

                    # Forward pass
                    outputs = model(torch.from_numpy(feature_input.values).float())

                    # Pairwise distances to backbone embeddings
                    pairwise_outputs = torch.cdist(outputs, embeddings_tensor, p=2,
                                                   compute_mode='donot_use_mm_for_euclid_dist')
                    pairwise_outputs = torch.square(pairwise_outputs)
                    pairwise_outputs = torch.where(
                        pairwise_outputs < 1.0e-6,
                        torch.tensor(0, dtype=pairwise_outputs.dtype),
                        pairwise_outputs
                    )

                    # Convert tensors to python lists once per chunk for fast writing
                    dist_block = pairwise_outputs.detach().cpu().numpy().tolist()
                    emb_block = outputs.detach().cpu().numpy().tolist()

                    # Stream rows to files
                    # Distance matrix rows: label + distances
                    for lbl, row_vals in zip(query_labels, dist_block):
                        dist_writer.writerow([lbl] + row_vals)

                    # Embedding rows: label + embedding vector; NO header
                    for lbl, emb_vals in zip(query_labels, emb_block):
                        emb_writer.writerow([lbl] + emb_vals)

        logging.info('Wrote distance matrix: {}'.format(dist_path))
        logging.info('Wrote embeddings: {}'.format(emb_path))

        logging.info('\n==> Computation is completed for subtree {}!\n'.format(c))
        hrs, _min, sec = hms(time.time() - since)
        logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))

    logging.info('\n==> Computation Completed!\n')
    hrs, _min, sec = hms(time.time() - since)
    logging.info('Total time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))
