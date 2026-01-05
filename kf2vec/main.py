# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Fixed bug associated with cdist on gpu
# Fixed bug in torch.index_select for subsetting lambdas

# Uses feature extracted from 10K dataset
# Kmer counts computed for >80K large contigs and <=80K chimeric contigs separately

import warnings

# Suppress the specific torchvision image extension warning
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension",
    category=UserWarning
)

import time
import itertools
import logging
import re
import argparse
import fnmatch
import treeswift
from treeswift import read_tree_newick
import warnings
from importlib import resources



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
import shutil
import glob
import subprocess
from subprocess import call, check_output, STDOUT
import multiprocessing as mp
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
from . import train_classifier_model
from .train_classifier_model import *
from . import classify
from .classify import *
from . import train_model_set
from .train_model_set import *
from . import query
from .query import *
#from query_last import *
#from query_consec import *

from . import  train_model_set_chunks
from .train_model_set_chunks import *
from . import train_classifier_model_chunks
from .train_classifier_model_chunks import *

default_k_len = 7
min_k_len = 2
max_k_len = 31
default_subtree_sz = 850
default_multiplier = 100

hidden_size_fc1 = 2048
embedding_size = 1024
batch_size = 16

default_cl_epochs = 2000
default_di_epochs = 8000

learning_rate = 0.00001     # 1e-5
learning_rate_min = 3e-6    # 3e-6
learning_rate_decay = 2000

seed = 28
default_block_sz = 4000

chunk_sz = 10000 # Minimum chunk size
chunk_cnt_thr = 5 # Minimum number of chunks to preserve genome in a dataset.


__version__ = 'kf2vec 0.1.3'


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def get_kmers(args):
    """
    Processes FASTA files using Jellyfish, encodes k-mers numerically,
    and saves normalized frequency matrices as .npy files.
    """
    # Base mapping: 0=A, 1=T, 2=C, 3=G
    base_map = {b'A': 0, b'T': 1, b'C': 2, b'G': 3}

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    fasta_files = glob.glob(os.path.join(args.input_dir, "*.fna"))

    for fna_path in fasta_files:
        base_name = os.path.basename(fna_path).replace(".fna", "")
        jf_file = f"{base_name}.jf"
        
        print(f"--- Processing {base_name} ---")

        try:
            # 1. Run Jellyfish Count
            # -C for canonical, -s 100M for hash size, -t 10 for threads
            subprocess.run([
                "jellyfish", "count", "-m", str(args.k), "-s", "100M", 
                "-t", "10", "-C", fna_path, "-o", jf_file
            ], check=True)

            # 2. Run Jellyfish Dump
            # -c includes counts, -t is tab-delimited
            dump_process = subprocess.Popen(
                ["jellyfish", "dump", "-c", "-t", jf_file],
                stdout=subprocess.PIPE, text=True
            )

            kmer_data = []
            counts = []

            # 3. Parse Dump and Encode
            for line in dump_process.stdout:
                seq, count = line.strip().split()
                # Check for standard DNA bases only
                if all(base in "ATCG" for base in seq):
                    # Convert string to list of integers based on your mapping
                    encoded_seq = [base_map[b.encode()] for b in seq]
                    kmer_data.append(encoded_seq)
                    counts.append(int(count))

            if not kmer_data:
                print(f"Warning: No valid ATCG k-mers found in {base_name}")
                continue

            # 4. Create the Nx(k+1) Matrix
            kmer_matrix = np.array(kmer_data, dtype=np.float32)
            counts_array = np.array(counts, dtype=np.float32)
            
            # Normalize frequencies
            normalized_freqs = counts_array / np.sum(counts_array)
            
            # Combine: Append frequencies as the (k+1)-th column
            final_matrix = np.column_stack((kmer_matrix, normalized_freqs))

            # 5. Save as .npy
            output_path = os.path.join(args.output_dir, f"{base_name}_k{args.k}.npy")
            np.save(output_path, final_matrix)
            print(f"Saved: {output_path} (Shape: {final_matrix.shape})")

        except subprocess.CalledProcessError as e:
            print(f"Error running Jellyfish on {fna_path}: {e}")
        finally:
            # 6. Clean up intermediate Jellyfish files
            if os.path.exists(jf_file):
                os.remove(jf_file)

def divide_tree(args):

    # Read tree file
    try:
        tree = treeswift.read_tree_newick(args.tree)
    except:
        print("No such file '{}'".format(args.tree), file=sys.stderr)
        exit(0)



    # Split path and filename
    head_tail = os.path.split(args.tree)
    tree_name = os.path.splitext(os.path.basename(args.tree))[0]


    # Set branch lengths to 1
    for node in tree.traverse_postorder():
        if node.label  != None:
            node.edge_length = 1.0
    tree_tmp = os.path.join(head_tail[0], "{}.{}".format(tree_name, "tree_tmp"))


    # Save tree output
    tree.write_tree_newick(tree_tmp)
    #d=tree.diameter()


    # Run TreeCluster
    subtree_tmp = os.path.join(head_tail[0] , "{}.{}".format(tree_name, "subtrees_tmp"))

    call(["TreeCluster.py", "-i", tree_tmp, "-o", subtree_tmp, "-m", "sum_branch", "-t", str(2 * args.size)],
         stderr=open(os.devnull, 'w'))


    # Reformat TreeCluster output
    current_subclades = pd.read_csv(subtree_tmp, sep= '\t', header = 0)

    # Check for -1 subtrees
    labels = current_subclades.loc[current_subclades["ClusterNumber"] ==-1]
    problematic_labels = labels["SequenceName"].to_list()

    if len(problematic_labels) > 0:
        warnings.warn('{} samples are assigned to subtrees -1 and will be excluded.\n'
                      'Please check rooting of your phylogeny or increase subtree size.'.format(len(problematic_labels)))
    else:
        print("There are no -1 subtrees. Keep going...")


    current_subclades = current_subclades.rename({"SequenceName" : "genome", "ClusterNumber" : "clade"}, axis = 1)
    current_subclades["clade"] = current_subclades["clade"]-1
    current_subclades = current_subclades.loc[ current_subclades["clade"] !=-2]


    # Save to file
    subtrees = os.path.join(head_tail[0], "{}.{}".format(tree_name, "subtrees"))
    current_subclades.to_csv(subtrees, index = False, sep = " ", header = True)


    # Clean up
    os.remove(tree_tmp)
    os.remove(subtree_tmp)


def get_frequencies(args):

    print('\n==> Starting k-mer counting for {}\n'.format(args.input_dir))

    # Check if input directory exist
    if os.path.exists(args.input_dir):
        pass
    else:
        print("No such directory '{}'".format(args.input_dir), file=sys.stderr)
        exit(0)

    # Check if output directory exist
    if os.path.exists(args.output_dir):
        pass
    else:
        print("No such directory '{}'".format(args.output_dir), file=sys.stderr)
        exit(0)


    # Making a list of sample names
    #print('\n==> Making a list of sample names...\n')

    formats = ['.fq', '.fastq', '.fa', '.fna', '.fasta']
    files_names = [f for f in os.listdir(args.input_dir)
                   if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]
    samples_names = [f.rsplit('.f', 1)[0] for f in files_names]


    vocab_path = resources.files("kf2vec")/"data"

    # Read kmer alphabet
    if args.k==7:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "test_kmers_7_sorted"), sep = " ", header = None, names = ["kmer"])
    elif args.k==3:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "vocab_generator_k3C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==4:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "vocab_generator_k4C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==5:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "vocab_generator_k5C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==6:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "test_kmers_6_sorted"), sep = " ", header = None, names = ["kmer"])
    elif args.k==8:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "vocab_generator_k8C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==9:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "vocab_generator_k9C_fin.fa"), sep = " ", header = None, names = ["kmer"])
    elif args.k==10:
        my_alphabet_kmers = pd.read_csv(os.path.join(vocab_path, "vocab_generator_k10C_fin.fa"), sep = " ", header = None, names = ["kmer"])



    # Compute kmer counts per file
    for i in range (0, len(files_names)):


        #print('\n==> Start processing. Sample: {}'.format(files_names[i]))
        #print(files_names[i])

        # Run jellyfish
        f1 = os.path.join(args.output_dir, "{}.{}".format(samples_names[i],"jf"))
        subprocess.run(["jellyfish", "count", "-m", str(args.k), "-s", "100M", "-t", str(args.p),
               "-C", os.path.join( args.input_dir, files_names[i]) ,"-o", f1],
             stderr=open(os.devnull, 'w'))

        # Match filename to the pattern of jellyfish adds suffix to the f1 output
        pattern = os.path.join(f1 + "*")
        f1 = glob.glob(pattern)[0]

        f2 = os.path.join(args.output_dir, "{}.{}".format(samples_names[i], "dump"))
        with open(f2, "w") as outfile:
            subprocess.run(["jellyfish", "dump", "-c", f1], stdout=outfile)

        # Read into dataframe
        #print('>>> Reading counts. Sample: {}'.format(files_names[i]))
        my_current_kmers = pd.read_csv(f2, sep = " ", header = None, names = ["kmer", "counts"])


        # Merge dataframe with alphabet kmers
        my_merged_counts = pd.merge(my_alphabet_kmers, my_current_kmers, how = 'left', left_on="kmer", right_on="kmer")
        my_merged_counts = my_merged_counts[['counts']].fillna(0)


        # Add pseudocounts if flag is on
        if args.pseudocount:
            print('>>> Adding pseudocounts. Sample: {}'.format(files_names[i]))
            my_merged_counts["counts"] = my_merged_counts["counts"] + 0.5

        else:
            pass

        # if raw counts are not requested frequencies are normalized to 1
        if not args.raw_cnt:
            print('>>> Normalizing. Sample: {}'.format(files_names[i]))
            my_merged_counts["counts"] = my_merged_counts["counts"] / my_merged_counts["counts"].sum()

        my_merged_counts["counts"] = my_merged_counts["counts"].astype(str)
        my_merged_list = my_merged_counts["counts"].to_list()


        # Output into file
        #print('>>> Preparing output. Sample: {}'.format(files_names[i]))
        f3 = os.path.join(args.output_dir, "{}.{}".format(samples_names[i], "kf"))
        my_output = ",".join(my_merged_list)

        with open(f3, "w") as f:

            f.write("{},".format (str(samples_names[i])))
            f.write(my_output)
            f.write("\n")


        # Clean up
        try:
            os.remove(f1)
        except:
            print('\n==> .jf is not present. Sample: {}'.format(files_names[i]))


        try:
            os.remove(f2)
        except:
            print('\n==> .dump is not present. Sample: {}'.format(files_names[i]))


    print('\n==> Done processing {}'.format(args.input_dir))


def train_classifier(args):

    # Concatenate kmer frequencies into single dataframe
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    """
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep= ',')
        li.append(df)


    frame = pd.concat(li, axis=0, ignore_index=True)
    """

    #frame.set_index(0, inplace=True)

    # Concatenate inputs into single dataframe
    #frame = construct_input_dataframe(li)

    train_classifier_model_func(args.input_dir, all_files, args.subtrees, args.e, args.hidden_sz, args.batch_sz, args.lr, args.lr_min, args.lr_decay, args.seed, args.mask, args.o)


def classify(args):

    # Read kmer frequencies filenames into a list
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))

    # # Delete previous log file if exist
    # try:
    #     os.remove(os.path.join(args.output_classes, 'classification.log'))
    # except OSError:
    #     pass

    classify_func(args.input_dir, all_files, args.model, args.seed, args.o, args.block)


def scale_tree(args):

    # Read tree file
    try:
        tree = treeswift.read_tree_newick(args.tree)
    except:
        print("No such file '{}'".format(args.tree), file=sys.stderr)
        exit(0)


    # Split path and filename
    head_tail = os.path.split(args.tree)
    filename, file_extension = os.path.splitext(os.path.basename(args.tree))


    # Scale branches by 100
    print("Original diameter: {}".format(tree.diameter()))
    tree.scale_edges(args.factor)
    print("Scaled diameter: {}".format(tree.diameter()))

    new_tree_name = "{}_r{}{}".format(filename, args.factor, file_extension)
    # Output scaled tree
    tree.write_tree_newick(os.path.join(head_tail[0], new_tree_name))



def get_distances(args):

    # Read tree file
    try:
        tree = treeswift.read_tree_newick(args.tree)
    except:
        print("No such file '{}'".format(args.tree), file=sys.stderr)
        exit(0)


    # Split path and filename
    head_tail = os.path.split(args.tree)
    tree_name = os.path.splitext(os.path.basename(args.tree))[0]

    # Scale branches by 100
    #tree.scale_edges(100)

    # Compute distance matrix for a full tree and convert into dataframe
    if args.mode == "full_only" or args.mode == "hybrid":

        only_leaves = tree.num_nodes(internal=False)  # exclude internal nodes

        # Warning for trees with more than 12K species
        if only_leaves > 12000:
            warnings.warn('Phylogeny contains {} samples which is above recommended threshold of 12000 species.\n'
                          'Computation of distance matrix might take long time.'.format(only_leaves))
        else:
            pass

        M = tree.distance_matrix(leaf_labels=True)
        df = pd.DataFrame.from_dict(M, orient='index').fillna(0)
        df.to_csv(os.path.join(head_tail[0], '{}_full.di_mtrx'.format(tree_name)), index=True, sep='\t')



    # Read clades information if provided by the user
    if args.mode == "hybrid" or args.mode == "subtrees_only":

        if args.subtrees is None:
            print("No such file '{}'. Please provide /.subtrees file or change mode to full_only".format(args.subtrees), file=sys.stderr)
            exit(0)

        else:
            clade_input = pd.read_csv(args.subtrees, sep=' ', header=0, index_col=0)
            clade_selection = list(set(clade_input["clade"].to_list()))


            # Compute distance matrices for subtrees
            for c in clade_selection:

                # Get labels of the subtree
                labels_to_keep = set(clade_input.loc[clade_input["clade"] == c].index.to_list())

                # NOTE: Here I am not checking for single species clades since
                # they should have been eliminated during clading step

                # Generate subtree
                tree2 = tree.extract_tree_with(labels_to_keep)

                # Compute distance matrix for a subtree and convert into dataframe
                M = tree2.distance_matrix(leaf_labels=True)
                df = pd.DataFrame.from_dict(M, orient='index').fillna(0)
                df.to_csv(os.path.join(head_tail[0], '{}_subtree_{}.di_mtrx'.format(tree_name, c)), index=True, sep='\t')



def train_model_set(args):


    # Concatenate kmer frequencies into single dataframe
    print("Running train_model_set")
    if not(args.no_fsw):
        all_files = glob.glob(os.path.join(args.input_dir, "*.npy"))
    else:
        all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))


    # li = []
    #
    # for filename in all_files:
    #     df = pd.read_csv(filename, index_col=None, header=None, sep= ',')
    #     li.append(df)
    #
    #
    # frame = pd.concat(li, axis=0, ignore_index=True)
    # frame.set_index(0, inplace=True)
    #
    # # Concatenate inputs into single dataframe
    # # frame = construct_input_dataframe(li)

    train_model_set_func(args.input_dir, all_files, args.subtrees, args.true_dist, args.e, args.hidden_sz, args.embed_sz, args.batch_sz, args.lr, args.lr_min, args.lr_decay, args.clade, args.seed, args.o, args.test_set, args.save_interval,
                         use_fsw=not(args.no_fsw), base_dim=args.base_dim, fswout_dim=args.fswout_dim)



def query(args):

    # Read kmer frequencies filenames into a list
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))
    """
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep=',')
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.set_index(0, inplace=True)

    # Concatenate inputs into single dataframe
    # frame = construct_input_dataframe(li)
    """

    query_func(args.input_dir, all_files, args.model, args.classes, args.seed, args.o, args.remap, args.block)

   # # Query last model if exist
   #  try:
   #      query_func_last(args.input_dir, all_files, args.model, args.classes, args.seed, args.o)
   #  except:
   #      pass
   #
   #  # Query consec model if exist
   #  try:
   #      query_func_consec(args.input_dir, all_files, args.model, args.classes, args.seed, args.o)
   #  except:
   #      pass



def build_library(args):


    print("\n==> Computing k-mer frequences\n")
    get_frequencies(args)


    print("\n==> Splitting phylogeny into subtrees\n")
    divide_tree(args)


    print("\n==> Computing distance matrices\n")

    # Split path and filename
    head_tail = os.path.split(args.tree)
    tree_name = os.path.splitext(os.path.basename(args.tree))[0]

    args.subtrees = os.path.join(head_tail[0], "{}.{}".format(tree_name, "subtrees"))
    get_distances(args)


    print("\n==> Training classifier model\n")
    args.input_dir = args.output_dir
    # args.subtrees = args.subtrees # defined above
    args.e = args.cl_epochs
    args.hidden_sz = args.cl_hidden_sz
    args.batch_sz = args.cl_batch_sz
    args.lr = args.cl_lr
    args.lr_min = args.cl_lr_min
    args.lr_decay = args.cl_lr_decay
    args.seed = args.cl_seed
    args.o = args.output_dir

    train_classifier(args)


    print("\n==> Training distance models\n")
    #args.input_dir = args.output_dir # defined above
    args.true_dist = head_tail[0]
    # args.subtrees = args.subtrees # defined above
    args.e = args.di_epochs
    args.hidden_sz = args.di_hidden_sz
    args.embed_sz = args.di_embed_sz
    args.batch_sz = args.di_batch_sz
    args.lr = args.di_lr
    args.lr_min = args.di_lr_min
    args.lr_decay = args.di_lr_decay
    args.seed = args.di_seed
    #args.o = args.output_dir # defined above

    train_model_set(args)


    print('\n==> Building library step is completed!\n')



def process_query_data(args):

    print("\n==> Computing k-mer frequences\n")
    get_frequencies(args)


    print("\n==> Classifying query samples\n")
    args.input_dir = args.output_dir
    args.model = args.classifier_model
    args.seed = args.cl_seed
    args.o = args.output_dir

    classify(args)


    print("\n==> Computing model distances\n")
    # args.input_dir = args.output_dir # defined above
    args.model = args.distance_model
    args.classes = args.output_dir
    args.seed = args.di_seed
    # args.o = args.output_dir # defined above

    query(args)


    print('\n==> Query processing step is completed!\n')


def get_chunks(args):

    since = time.time()


    # Check if input directory exist
    if os.path.exists(args.input_dir):
        pass
    else:
        print("No such directory '{}'".format(args.input_dir), file=sys.stderr)
        exit(0)

    # Check if output directory exist
    if os.path.exists(args.output_dir):
        pass
    else:
        print("No such directory '{}'".format(args.output_dir), file=sys.stderr)
        exit(0)


    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(args.output_dir, 'get_chunks_{}.log'.format(os.path.basename(os.path.normpath(args.input_dir)))), 'w+'),
                logging.StreamHandler()]

    # logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s: %(message)s', handlers=handlers)

    logging.basicConfig(level=level, format=format, handlers=handlers)
    # logging.info('Hey, this is working!')

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('\n==> Making a list of sample names. Time: {:02d}:{:02d}:{:02d}\n'.format(hrs, _min, sec))


    # Making a list of sample names
    formats = ['.fq', '.fastq', '.fa', '.fna', '.fasta']
    files_names = [f for f in os.listdir(args.input_dir)
                   if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]
    samples_names = [f.rsplit('.f', 1)[0] for f in files_names]


    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('\n==> Start processing samples. Time: {:02d}:{:02d}:{:02d}\n'.format(hrs, _min, sec))

    # Process single sample from beginning to the end
    for i in range(0, len(files_names)):

        logging.info('\n==> Start processing. Sample: {}'.format(files_names[i]))

        # Create tmp directory in output folder
        ndr1 = "{}_tmp".format(samples_names[i])
        if not os.path.exists(os.path.join(args.output_dir, ndr1)):
            os.makedirs(os.path.join(args.output_dir, ndr1))

        # Create contigs directory in output folder
        ndr2 = "{}_contigs".format(samples_names[i])
        if not os.path.exists(os.path.join(args.output_dir, ndr2)):
            os.makedirs(os.path.join(args.output_dir, ndr2))

        # Create chunks directory in output folder
        ndr3 = "{}_chunks".format(samples_names[i])
        if not os.path.exists(os.path.join(args.output_dir, ndr3)):
            os.makedirs(os.path.join(args.output_dir, ndr3))

        # Create kf directory in output folder
        ndr4 = "{}_kf".format(samples_names[i])
        if not os.path.exists(os.path.join(args.output_dir, ndr4)):
            os.makedirs(os.path.join(args.output_dir, ndr4))


        # Convert multiline fasta into single line fasta (linearizing files)
        # Better practice would be to write or use programs that can handle wrapped fasta
        logging.info('>>> Formatting to single line. Sample: {}'.format(files_names[i]))

        f_single_line = os.path.join(*[args.output_dir, ndr1, "{}.{}".format(samples_names[i], "sline")])
        with open(f_single_line, "w") as outfile:
            subprocess.run(["seqtk", "seq", "-l", "0", os.path.join(args.input_dir, files_names[i])], stdout=outfile)



        # Replace multiple occurrences of N or n with single N, exclude N chars in sequence name:
        logging.info('>>> Replacing stretches of N. Sample: {}'.format(files_names[i]))

        f0 = os.path.join(*[args.output_dir, ndr1, "{}.{}".format(samples_names[i], "short")])
        find_txt_command = ["awk", '{!/^(>)/ && gsub(/[N|n]+/,"N")}1', f_single_line]
        with open(f0, 'w') as my_out_file:
            subprocess.run(find_txt_command, stdout=my_out_file, shell=False)



        # Filter out contigs below threshold length
        logging.info('>>> Filtering contigs below threshold {}. Sample: {}'.format(str(chunk_sz), files_names[i]))

        f1 = os.path.join(*[args.output_dir, ndr1, "{}.{}".format(samples_names[i], "filt")])
        # cmd = "reformat.sh -Xmx2g in=" + f0 + " out=" + f1 + " minlength=" + str(chunk_sz) + " overwrite=true"
        # subprocess.call(cmd, shell=True)

        subprocess.run(["seqkit", "seq", "-m", str(chunk_sz), f0, "-o", f1, "-g", "-v"], stderr=open(os.devnull, 'w'))

        # with open(f1, 'w') as my_out_file:
        #     subprocess.run(["seqkit", "seq", "-m", str(chunk_sz), f0, "-g", "-v"], stdout=my_out_file, shell=False)



        # Check if .filt file is empty (no contigs > threshold). If empty skip to the next sample.
        with open(f1) as file_obj:
            # read first character
            first_char = file_obj.read(1)
            if not first_char:

                time_elapsed = time.time() - since
                hrs, _min, sec = hms(time_elapsed)
                logging.info( '\n==> Excluded {}. No contigs above threshold length. Time: {:02d}:{:02d}:{:02d}\n'.format(files_names[i],hrs, _min, sec))

                # Clean up tmp folders
                """
                shutil.rmtree(os.path.join(*[args.output_dir, ndr1]))
                shutil.rmtree(os.path.join(*[args.output_dir, ndr2]))
                shutil.rmtree(os.path.join(*[args.output_dir, ndr3]))
                shutil.rmtree(os.path.join(*[args.output_dir, ndr4]))
                """
                # Skip to the next sample
                continue



        # Split sample into contigs
        logging.info('>>> Splitting into contigs. Sample: {}'.format(files_names[i]))
        subprocess.run(["seqkit", "split", "--by-id", f1, "--out-dir", os.path.join(*[args.output_dir, ndr2])],
             stderr=open(os.devnull, 'w'))


        # Making a list of sample names for contig files
        logging.info('>>> Getting contig ids. Sample: {}'.format(files_names[i]))

        formats = ['.filt']
        files_names2 = [f for f in os.listdir(os.path.join(*[args.output_dir, ndr2]))
                       if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]
        samples_names2 = [f.rsplit('.filt', 1)[0] for f in files_names2]



        # Compute statistics per contig
        logging.info('>>> Computing contig statistics. Sample: {}'.format(files_names[i]))

        ordered_chunks_fnames = []
        for j in range(0, len(files_names2)):

            # print(files_names2[j])
            # print(samples_names2[j])

            # Run seqtk to compute contig length
            comp_stdout = check_output(["seqtk", "comp", os.path.join(*[args.output_dir, ndr2, files_names2[j]])],  stderr=STDOUT, universal_newlines=True)
            reads_stat = comp_stdout.split('\n')
            total_length = int(reads_stat[0].split("\t")[1])

            # Compute overlap and split contigs into chunks
            total_chunks =  math.ceil(total_length/chunk_sz)
            if total_chunks != 1:
                ovrlap = int(math.ceil((total_chunks*chunk_sz - total_length) / (total_chunks - 1)))
            else:
                ovrlap = 0
            step = chunk_sz - ovrlap
            # print(total_length)
            # print(total_chunks)
            # print(ovrlap)
            # print(step)
            f2 = os.path.join(os.path.join(*[args.output_dir, ndr2, "{}.{}".format(samples_names2[j], "fna")]))
            subprocess.run(["seqkit", "sliding", os.path.join(*[args.output_dir, ndr2, files_names2[j]]), "--step", str(step), "--window", str(chunk_sz), "--out-file", f2 ],
                 stderr=open(os.devnull, 'w'))

            # Save order of sliding chunks
            comp_stdout2 = check_output(["seqkit", "seq", "-n", f2],
                                       stderr=STDOUT, universal_newlines=True)
            reads_stat2 = comp_stdout2.split('\n') # Need to verify if works with simple chunk name
            ordered_chunks_fnames.extend(reads_stat2)

            # Drop empty chunk names
            ordered_chunks_fnames = [q for q in ordered_chunks_fnames if q!=""]

            # Split contig into chunks
            subprocess.run(["seqkit", "split", "--by-id", f2, "--out-dir", os.path.join(*[args.output_dir, ndr3])],
                           stderr=open(os.devnull, 'w'))

            # Split contig into chunks (using bbtools)
            #cmd2 = "demuxbyname.sh -Xmx2g in=" + f2 + " out=" + os.path.join(*[args.output_dir, ndr3, ]) + "/" + samples_names2[j] + ".part_%.fna" + " header"
            #subprocess.call(cmd2, shell=True)

        # Drop sample with number of chunks below threshold
        if len(ordered_chunks_fnames) < chunk_cnt_thr:
            time_elapsed = time.time() - since
            hrs, _min, sec = hms(time_elapsed)
            logging.info(
                '\n==> Excluded {}. {} chunks is too low. {} is required. Time: {:02d}:{:02d}:{:02d}\n'.format(
                    files_names[i], len(ordered_chunks_fnames), chunk_cnt_thr, hrs, _min, sec))

            # Clean up tmp folders

            shutil.rmtree(os.path.join(*[args.output_dir, ndr1]))
            shutil.rmtree(os.path.join(*[args.output_dir, ndr2]))
            shutil.rmtree(os.path.join(*[args.output_dir, ndr3]))
            shutil.rmtree(os.path.join(*[args.output_dir, ndr4]))

            # Skip to the next sample
            continue

            #logging.info('\n==> Done processing. Sample: {}'.format(files_names[i]))

        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)
        logging.info('\n==> Done chunk processing for {}. Time: {:02d}:{:02d}:{:02d}\n'.format(files_names[i], hrs, _min, sec))


        # Compute kmer frequences for a given sample
        orig_input_dir = args.input_dir # Save original values
        orig_output_dir = args.output_dir # Save original values

        args.input_dir = os.path.join(*[args.output_dir, ndr3])
        args.output_dir = os.path.join(*[args.output_dir, ndr4])

        args.k = args.k
        args.p = args.p
        args.pseudocount = args.pseudocount
        args.raw_cnt = True

        get_frequencies(args)

        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)
        logging.info('\n==> Done computing k-mer frequences for {}. Time: {:02d}:{:02d}:{:02d}\n'.format(files_names[i], hrs, _min, sec))


        # Assign original input and directories
        args.input_dir = orig_input_dir
        args.output_dir = orig_output_dir


        # Summarize kmer frequences into single file
        # Use with seqkit split
        my_order_list = ["{}.part_{}.part_{}.kf".format(samples_names[i], w.split("_sliding")[0], w) for w in ordered_chunks_fnames]
        my_order_list = [t.replace("sliding:", "sliding__") for t in  my_order_list]

        # Use with demuxbyname cmd
        #my_order_list = ["{}.part_{}.part_{}.kf".format(samples_names[i], w.split("_sliding")[0], w) for w in
        #                 ordered_chunks_fnames]
        #my_order_list = ["{}.kf".format(w) for w in ordered_chunks_fnames]



        f3 = os.path.join(os.path.join(*[args.output_dir, "{}.{}".format(samples_names[i], "kf")]))
        # with open(f3, 'w') as outfile:
        #     for fname in my_order_list:
        #         with open(os.path.join(*[ args.output_dir, ndr4, fname])) as infile:
        #             for line in infile:
        #                 outfile.write(line)

        with open(f3, 'wb') as wfd:
            for fname in my_order_list:
                with open(os.path.join(*[ args.output_dir, ndr4, fname]), 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)



        # Clean up tmp folders

        shutil.rmtree(os.path.join(*[args.output_dir, ndr1]))
        shutil.rmtree(os.path.join(*[args.output_dir, ndr2]))
        shutil.rmtree(os.path.join(*[args.output_dir, ndr3]))
        shutil.rmtree(os.path.join(*[args.output_dir, ndr4]))


    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('\n==> Done getting chunks. Time: {:02d}:{:02d}:{:02d}\n'.format(hrs, _min, sec))




def train_model_set_chunks(args):



    # Concatenate kmer frequencies into single dataframe
    print("Running train_model_set_chunks")
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))


    # li = []
    #
    # for filename in all_files:
    #     df = pd.read_csv(filename, index_col=None, header=None, sep= ',')
    #     li.append(df)
    #
    #
    # frame = pd.concat(li, axis=0, ignore_index=True)
    # frame.set_index(0, inplace=True)
    #
    # # Concatenate inputs into single dataframe
    # # frame = construct_input_dataframe(li)
    #
    #
    train_model_set_chunks_func(args.input_dir, args.input_dir_fullgenomes, all_files, args.subtrees, args.true_dist, args.e, args.hidden_sz, args.embed_sz, args.batch_sz, args.lr, args.lr_min, args.lr_decay, args.clade, args.seed, args.cap, args.o)


def train_classifier_chunks(args):

    # Concatenate kmer frequencies into single dataframe
    print("Running train_classifier_chunks")
    all_files = glob.glob(os.path.join(args.input_dir, "*.kf"))


    train_classifier_model_chunks_func(args.input_dir, args.input_dir_fullgenomes, all_files, args.subtrees, args.e, args.hidden_sz, args.batch_sz, args.lr, args.lr_min, args.lr_decay, args.seed, args.mask, args.cap, args.o)



def main():
    # Input arguments parser
    parser = argparse.ArgumentParser(description='K-mer frequency to distance\n{}'.format(__version__),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', help='print the current version', version='{}'.format(__version__))
    # parser.add_argument('--debug', action='store_true', help='Print the traceback when an exception is raised')
    subparsers = parser.add_subparsers(title='commands',
                                       description='get_kmers                Extract k-mers and their frequencies from FASTA files\n'
                                                   'get_frequencies          Extract k-mer frequency from a reference genome-skims or assemblies\n'
                                                   'divide_tree              Divides input phylogeny into subtrees\n'
                                                   'scale_tree               Multiplies all edges in the tree by multiplier\n'
                                                   'get_distances            Compute distance matrices\n'
                                                   'train_classifier         Train classifier model based on backbone subtrees\n'
                                                   'classify                 Classifies query samples using previously trained classifier model\n'
                                                   # 'train_model     Performs correction of subsampled distance matrices obtained for reference\n'
                                                   'train_model_set          Trains all models for subtrees consecutively\n'
                                                   'query                    Query subtree models\n'
                                                   # 'genome-skims or assemblies'
                                                   # 'build_library            Wrapper command to preprocess backbone sequences and phylogeny to train classifier and distance models\n'
                                                   # 'process_query_data       Wrapper command to preprocess query sequences, classify and compute distances to backbone species\n'

                                                   'get_chunks               Extract chunks from reference assemblies\n'
                                                   'train_model_set_chunks   Trains all models for subtrees consecutively using chunked input\n'
                                                   'train_classifier_chunks  Train classifier model based on backbone subtrees (genomes split into chunks)\n'
                                       ,
                                       help='Run kf2vec {commands} [-h] for additional help',
                                       dest='{commands}')


    parser_kmer = subparsers.add_parser('get_kmers',
                                        description='Extract kmers and frequencies from FASTA files')
    parser_kmer.add_argument('-input_dir',
                             help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_kmer.add_argument('-output_dir',
                             help='Directory for k-mer frequency outputs (dir for .kf files)')
    parser_kmer.add_argument('-k', type=int, choices=list(range(min_k_len, max_k_len+1)), default=default_k_len, help='K-mer length [{}-{}]. '.format(min_k_len, max_k_len) +
                                                                                         'Default: {}'.format(default_k_len), metavar='K')
    parser_kmer.set_defaults(func=get_kmers)

    # Get_frequencies command subparser

    ### To invoke
    ##### To debug:
    ##### jellyfish count -m 7 -s 100M  ../toy_example/test_fna/G000196015.fna -o ../toy_example/test_kf/G000196015.jf; jellyfish  dump -c ../toy_example/test_kf/G000196015.jf

    ### python main.py get_frequencies -input_dir /Users/nora/PycharmProjects/test_freq -output_dir /Users/nora/PycharmProjects/test_freq
    ### python main.py get_frequencies -input_dir /Users/nora/PycharmProjects/test_freq -pseudocount

    ### python main.py get_frequencies - input_dir ../toy_example/train_tree_fna - output_dir ../toy_example/train_tree_kf
    ### python main.py get_frequencies -input_dir ../toy_example/test_fna -output_dir ../toy_example/test_kf

    #  python -m kf2vec.main get_frequencies -input_dir /Users/nora/PycharmProjects/test_freq -output_dir /Users/nora/PycharmProjects/test_freq
    parser_freq = subparsers.add_parser('get_frequencies',
                                       description='Process a library of reference genome-skims or assemblies')
    parser_freq.add_argument('-input_dir',
                            help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_freq.add_argument('-output_dir',
                             help='Directory for k-mer frequency outputs (dir for .kf files)')
    parser_freq.add_argument('-k', type=int, choices=list(range(min_k_len, max_k_len+1)), default=default_k_len, help='K-mer length [{}-{}]. '.format(min_k_len, max_k_len) +
                                                                                         'Default: {}'.format(default_k_len), metavar='K')
    parser_freq.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_freq.add_argument('-pseudocount', action='store_true',
                           help='Computes k-mer counts with 0.5 pseudocount added to each frequency value')
    parser_freq.add_argument('-raw_cnt', action='store_true',
                             help='Computes raw k-mer counts without normalization')
    parser_freq.set_defaults(func=get_frequencies)


    # Divide_tree command subparser

    ### To invoke
    ### python main.py divide_tree -size 850 -tree /Users/nora/PycharmProjects/astral.rand.lpp.r100.EXTENDED.nwk
    ### python main.py divide_tree -tree ../toy_example/train_tree_newick/train_tree.nwk -size 2

    ### python -m kf2vec.main divide_tree -tree ../toy_example/train_tree_newick/train_tree.nwk -size 2

    parser_div = subparsers.add_parser('divide_tree',
                                       description='Divides input phylogeny into subtrees.')
    parser_div.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)')
    parser_div.add_argument('-size', type=int, default=default_subtree_sz, help='Size of the subtree. ' +
                                                                                         'Default: {}'.format(default_subtree_sz))
    parser_div.set_defaults(func=divide_tree)


    # Scale_tree command subparser

    ### To invoke
    ### python main.py scale_tree -tree /Users/nora/PycharmProjects/test_tree.nwk  -factor 100
    ### python main.py scale_tree -tree ../toy_example/train_tree_newick/train_tree.nwk  -factor 100
    ### python -m kf2vec.main scale_tree -tree ../toy_example/train_tree_newick/train_tree.nwk  -factor 100


    parser_scale = subparsers.add_parser('scale_tree',
                                       description='Scales all edges in the tree by multiplier.')
    parser_scale.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)')
    parser_scale.add_argument('-factor', type=float, default=default_multiplier, help='Multiplier. ' +
                                                                                'Default: {}'.format(
                                                                                    default_multiplier))
    parser_scale.set_defaults(func=scale_tree)


    # Get_distances command subparser

    ### To invoke
    ### python main.py get_distances -tree /Users/nora/PycharmProjects/test_tree.nwk  -subtrees  /Users/nora/PycharmProjects/my_test.subtrees -mode subtrees_only

    ### python main.py get_distances -tree ../toy_example/train_tree_newick/train_tree.nwk  -subtrees  ../toy_example/train_tree_newick/train_tree.subtrees -mode subtrees_only
    ### SINGLE CLADE: python main.py get_distances -tree ../toy_example/train_tree_newick_single_clade/train_tree.nwk  -subtrees  ../toy_example/train_tree_newick_single_clade/train_tree_single_clade.subtrees -mode subtrees_only
    ###
    ### python -m kf2vec.main get_distances -tree ../toy_example/train_tree_newick/train_tree.nwk  -subtrees  ../toy_example/train_tree_newick/train_tree.subtrees
    ### SINGLE CLADE: python -m kf2vec.main get_distances -tree ../toy_example/train_tree_newick_single_clade/train_tree.nwk  -subtrees  ../toy_example/train_tree_newick_single_clade/train_tree_single_clade.subtrees

    parser_distances = subparsers.add_parser('get_distances',
                                             description='Computes distance matrices')
    parser_distances.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)', required=True)
    parser_distances.add_argument('-subtrees',
                                  help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_distances.add_argument('-mode', type=str, metavar='',
                                  #choices={"full_only", "hybrid", "subtrees_only"}, default="hybrid",
                                  default="subtrees_only",
                                  #help='Ways to perform distance computation [full_only, hybrid, subtrees_only]. ' +
                                  help = 'Ways to perform distance computation [subtrees_only]. ' +
                                       #'Default: hybrid')
                                       'Default: subtrees_only')

    parser_distances.set_defaults(func=get_distances)


    # Train_classifier command subparser

    ### To invoke
    ### python main.py train_classifier -input_dir /Users/nora/PycharmProjects/train_tree_kf -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py train_classifier -input_dir ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10 -o ../toy_example/train_tree_models
    ### python main.py train_classifier -input_dir ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10  -hidden_sz 2000 -batch_sz 32 -o ../toy_example/train_tree_models

    ### python -m kf2vec.main train_classifier -input_dir ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10  -o ../toy_example/train_tree_models

    parser_trclas = subparsers.add_parser('train_classifier',
                                        description='Train classifier model based on backbone subtrees')
    parser_trclas.add_argument('-input_dir',
                             help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for backbone)')
    parser_trclas.add_argument('-subtrees', help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    # parser_trclas.add_argument('-e', type=int, metavar='', choices=list(range(1, max_cl_epochs)), default=default_cl_epochs, help='Epochs [1-{}]. '.format(max_cl_epochs-1) +
    #                                                                                     'Default: {}'.format(default_cl_epochs))
    parser_trclas.add_argument('-e', type=int, default=default_cl_epochs, help='Number of epochs. ' +
                                                                               'Default: {}'.format(default_cl_epochs))
    parser_trclas.add_argument('-hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size. ' +
                                                                                       'Default: {}'.format(hidden_size_fc1))
    parser_trclas.add_argument('-batch_sz', type=int, default=batch_size, help='Batch size. ' +
                                                                                        'Default: {}'.format(batch_size))
    parser_trclas.add_argument('-lr', type=float, default=learning_rate, help='Start learning rate. ' +
                                                                                        'Default: {}'.format(learning_rate))
    parser_trclas.add_argument('-lr_min', type=float, default=learning_rate_min, help='Minimum learning rate. ' +
                                                                                        'Default: {}'.format(learning_rate_min))
    parser_trclas.add_argument('-lr_decay', type=float, default=learning_rate_decay, help='Learning rate decay. ' +
                                                                                      'Default: {}'.format(learning_rate_decay))
    parser_trclas.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                            'Default: {}'.format(seed))
    parser_trclas.add_argument('-mask', action='store_true',
                             help=argparse.SUPPRESS)#'Masks low complexity k-mers in input features (reduces input dimension'
    parser_trclas.add_argument('-o',
                               help='Model output path')

    parser_trclas.set_defaults(func=train_classifier)


    # Classify command subparser

    ### To invoke
    ### python main.py classify -input_dir /Users/nora/PycharmProjects/test_tree_kf -model /Users/nora/PycharmProjects/my_toy_input  -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py classify -input_dir ../toy_example/test_kf -model ../toy_example/train_tree_models -o ../toy_example/test_results
    ### python -m kf2vec.main classify -input_dir ../toy_example/test_kf -model ../toy_example/train_tree_models -o ../toy_example/test_results

    parser_classify = subparsers.add_parser('classify',
                                          description='Classifies query inputs using previously trained classifier model')
    parser_classify.add_argument('-input_dir',
                               help='Directory of input k-mer frequencies for queries samples: assemblies or reads (dir of .kf files for queries)')
    parser_classify.add_argument('-model',
                               help='Classification model')
    parser_classify.add_argument('-block', type=int, default=default_block_sz, help='Block size for file processing. ' +
                                                                       'Default: {}'.format(default_block_sz))
    parser_classify.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                     'Default: {}'.format(seed))
    parser_classify.add_argument('-o',
                               help='Output path')

    parser_classify.set_defaults(func=classify)




    # Train_model_set command subparser

    ### To invoke
    ### python main.py train_model_set -input_dir /Users/nora/PycharmProjects/train_tree_kf  -true_dist /Users/nora/PycharmProjects  -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py train_model_set -input_dir ../toy_example/train_tree_kf -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 1 -clade 0 -o ../toy_example/train_tree_models
    ### WITH SPECIFIED TEST SET: python main.py train_model_set -input_dir ../toy_example/train_tree_kf -test_set /Users/nora/PycharmProjects/toy_example/test_set.txt -true_dist ../toy_example/train_tree_newick_single_clade  -subtrees ../toy_example/train_tree_newick_single_clade/train_tree_single_clade.subtrees -e 5 -clade 0 -o ../toy_example/train_tree_models
    ### python main.py train_model_set -input_dir ../toy_example/train_tree_kf -test_set /Users/nora/PycharmProjects/toy_example/test_set.txt -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 1 -clade 0 -o ../toy_example/train_tree_models

    ### python -m kf2vec.main train_model_set -input_dir ../toy_example/train_tree_kf -test_set /Users/nora/PycharmProjects/toy_example/test_set.txt -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 1 -clade 0 -o ../toy_example/train_tree_models
    parser_train_model_set = subparsers.add_parser('train_model_set',
                                            description='Trains individual models for each subtree')
    parser_train_model_set.add_argument('-input_dir',
                               help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for backbone)')
    parser_train_model_set.add_argument('-test_set',
                                        help='File that contains list of filenames (no extension) to be used as a test set (list of .kf files for test subset)')
    parser_train_model_set.add_argument('-true_dist',
                                        help='Directory of distamce matrices for backbone subtrees (dir of *subtree_INDEX.di_mtrx files for backbone)')
    parser_train_model_set.add_argument('-subtrees',
                               help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_train_model_set.add_argument('-e', type=int, default=default_di_epochs, help='Number of epochs. ' +
                                                                                                 'Default: {}'.format(default_di_epochs))
    parser_train_model_set.add_argument('-hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size. ' +
                                                                                                'Default: {}'.format(hidden_size_fc1))
    parser_train_model_set.add_argument('-embed_sz', type=int, default=embedding_size, help='Embedding size. ' +
                                                                                                'Default: {}'.format(embedding_size))
    parser_train_model_set.add_argument('-batch_sz', type=int, default=batch_size, help='Batch size. ' +
                                                                            'Default: {}'.format(batch_size))
    parser_train_model_set.add_argument('-lr', type=float, default=learning_rate, help='Start learning rate. ' +
                                                                        'Default: {}'.format(learning_rate))
    parser_train_model_set.add_argument('-lr_min', type=float, default=learning_rate_min, help='Minimum learning rate. ' +
                                                                         'Default: {}'.format(learning_rate_min))
    parser_train_model_set.add_argument('-lr_decay', type=float, default=learning_rate_decay, help='Learning rate decay. ' +
                                                                                          'Default: {}'.format(learning_rate_decay))
    parser_train_model_set.add_argument('-clade', type=int, nargs='*', help='Clade number to train. ' +
                                                                                   'Default: all')
    parser_train_model_set.add_argument('-save_interval', type=int, help='Save model after specified interval of epochs. ' +
                                                                            'Default: last')
    parser_train_model_set.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                       'Default: {}'.format(seed))
    parser_train_model_set.add_argument('-o',
                               help='Model output path')
    parser_train_model_set.add_argument('-no_fsw', action='store_true', help="Keep original model")
    parser_train_model_set.add_argument('-fswout_dim', type=int, default=512)
    parser_train_model_set.add_argument('-base_dim', type=int, default=4)

    parser_train_model_set.set_defaults(func=train_model_set)



    # Query_model_set command subparser

    ### To invoke
    ### python main.py query -input_dir /Users/nora/PycharmProjects/test_tree_kf  -model /Users/nora/PycharmProjects/my_toy_input  -classes /Users/nora/PycharmProjects/my_toy_input  -o /Users/nora/PycharmProjects/my_toy_input
    ### python main.py query -input_dir ../toy_example/test_kf  -model ../toy_example/train_tree_models -classes ../toy_example/test_results  -o ../toy_example/test_results -remap /Users/nora/Documents/ml_metagenomics/cami_long_reads/my_rename_test.tsv

    ### python -m kf2vec.main query -input_dir /Users/nora/PycharmProjects/toy_example/test_kf  -model /Users/nora/PycharmProjects/my_toy_input  -classes /Users/nora/PycharmProjects/toy_example/test_results  -o /Users/nora/PycharmProjects/my_toy_input -block 3 -remap /Users/nora/Documents/ml_metagenomics/cami_long_reads/my_rename_test.tsv
    ### python -m kf2vec.main query -input_dir /Users/nora/PycharmProjects/test_tree_kf  -model /Users/nora/PycharmProjects/my_toy_input  -classes /Users/nora/PycharmProjects/my_toy_input  -o /Users/nora/PycharmProjects/my_toy_input

    parser_query = subparsers.add_parser('query',
                                                   description='Query models')
    parser_query.add_argument('-input_dir',
                                        help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for queries)')
    parser_query.add_argument('-model',
                                        help='Directory of models and embeddings (dir of model_subtree_INDEX.ckpt and embeddings_subtree_INDEX.csv files for backbone)')
    parser_query.add_argument('-classes',
                                        help='Path to classification file with subtrees information obtained from classify command (classes.out file)')
    parser_query.add_argument('-block', type=int, default=default_block_sz, help='Block size for file processing. ' +
                                                                       'Default: {}'.format(default_block_sz))
    parser_query.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                              'Default: {}'.format(seed))
    parser_query.add_argument('-remap',
                                       help='Remap file with alterntive output names ("label" and "new_label" columns in .tsv format)')
    parser_query.add_argument('-o',
                                        help='Output path')

    parser_query.set_defaults(func=query)



    # Build_library command subparser

    ### To invoke
    ### python main.py build_library -input_dir /Users/nora/PycharmProjects/train_tree_fna -output_dir /Users/nora/PycharmProjects/train_tree_output -size 2 -tree /Users/nora/PycharmProjects/test_tree.nwk -mode subtrees_only -cl_epochs 1 -di_epochs 1

    ### python main.py build_library -input_dir ../toy_example/train_tree_fna -output_dir ../toy_example/combo_models -size 2 -tree ../toy_example/train_tree_newick/train_tree.nwk -mode subtrees_only -cl_epochs 10 -di_epochs 1

    parser_build_library = subparsers.add_parser('build_library',
                                                   description='Wrapper command that combines subcommands: get_frequencies (from backbone sequences), divide_tree, get_distance, train_classifier and train_model_set')

    parser_build_library.add_argument('-input_dir',
                             help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_build_library.add_argument('-output_dir',
                             help='Directory for all outputs (dir for .kf files)')
    parser_build_library.add_argument('-k', type=int, choices=list(range(min_k_len, max_k_len + 1)), default=default_k_len,
                             help='K-mer length [{}-{}]. '.format(min_k_len, max_k_len) +
                                  'Default: {}'.format(default_k_len), metavar='K')
    parser_build_library.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                             help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                  'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_build_library.add_argument('-pseudocount', action='store_true',
                             help='Computes k-mer counts with 0.5 pseudocount added to each frequency value')
    parser_build_library.add_argument('-raw_cnt', action='store_true',
                             help='Computes raw k-mer counts without normalization')

    parser_build_library.add_argument('-tree', help='Input phylogeny (a .newick/.nwk format)')
    parser_build_library.add_argument('-size', type=int, default=default_subtree_sz, help='Size of the subtree. ' +
                                                                 'Default: {}'.format(default_subtree_sz))

    parser_build_library.add_argument('-mode', type=str, metavar='',
                                  choices={"full_only", "hybrid", "subtrees_only"}, default="hybrid",
                                  help='Ways to perform distance computation [full_only, hybrid, subtrees_only]. ' +
                                       'Default: hybrid')

    parser_build_library.add_argument('-cl_epochs', type=int, default=default_cl_epochs, help='Number of epochs to train classifier model. ' +
                                                               'Default: {}'.format(default_cl_epochs))
    parser_build_library.add_argument('-cl_hidden_sz', type=int, default=hidden_size_fc1, help='Classifier hidden size. ' +
                                                                                       'Default: {}'.format(hidden_size_fc1))
    parser_build_library.add_argument('-cl_batch_sz', type=int, default=batch_size, help='Classifier batch size. ' +
                                                                                         'Default: {}'.format(batch_size))
    parser_build_library.add_argument('-cl_lr', type=float, default=learning_rate, help='Classifier start learning rate. ' +
                                                                              'Default: {}'.format(learning_rate))
    parser_build_library.add_argument('-cl_lr_min', type=float, default=learning_rate_min, help='Classifier minimum learning rate. ' +
                                                                              'Default: {}'.format(learning_rate_min))
    parser_build_library.add_argument('-cl_lr_decay', type=float, default=learning_rate_decay, help='Classifier learning rate decay. ' +
                                             'Default: {}'.format(learning_rate_decay))
    parser_build_library.add_argument('-cl_seed', type=int, default=seed, help='Classifier random seed. ' +
                                                                    'Default: {}'.format(seed))


    parser_build_library.add_argument('-di_epochs', type=int, default=default_di_epochs, help='Number of epochs to train distance models. ' +
                                                                        'Default: {}'.format(default_di_epochs))
    parser_build_library.add_argument('-di_hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size for distance models. ' +
                                                                                                 'Default: {}'.format(hidden_size_fc1))
    parser_build_library.add_argument('-di_embed_sz', type=int, default=embedding_size, help='Distance model embedding size. ' +
                                                                                          'Default: {}'.format(embedding_size))
    parser_build_library.add_argument('-di_batch_sz', type=int, default=batch_size, help='Distance model batch size. ' +
                                             'Default: {}'.format(batch_size))
    parser_build_library.add_argument('-di_lr', type=float, default=learning_rate, help='Distance model start learning rate. ' +
                                                                                       'Default: {}'.format(learning_rate))
    parser_build_library.add_argument('-di_lr_min', type=float, default=learning_rate_min, help='Distance model minimum learning rate. ' +
                                             'Default: {}'.format(learning_rate_min))
    parser_build_library.add_argument('-di_lr_decay', type=float, default=learning_rate_decay, help='Distance learning rate decay. ' +
                                           'Default: {}'.format(learning_rate_decay))
    parser_build_library.add_argument('-di_seed', type=int, default=seed, help='Distance model random seed. ' +
                                                                               'Default: {}'.format(seed))


    parser_build_library.set_defaults(func=build_library)



    # Process_query_data command subparser

    ### To invoke
    ### python main.py process_query_data -input_dir /Users/nora/PycharmProjects/test_freq -output_dir /Users/nora/PycharmProjects/test_tree_output  -classifier_model /Users/nora/PycharmProjects/train_tree_output -distance_model /Users/nora/PycharmProjects/train_tree_output

    ### python main.py process_query_data -input_dir ../toy_example/test_fna -output_dir ../toy_example/combo_results   -classifier_model ../toy_example/combo_models -distance_model ../toy_example/combo_models

    parser_process_query_data = subparsers.add_parser('process_query_data',
                                         description='Wrapper command that combines subcommands: get_frequencies (from query samples), classify and query')

    parser_process_query_data.add_argument('-input_dir',
                                      help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_process_query_data.add_argument('-output_dir',
                                      help='Directory for outputs (dir for .kf files)')
    parser_process_query_data.add_argument('-k', type=int, choices=list(range(3, 11)), default=7,
                                      help='K-mer length [3-10]. ' +
                                           'Default: 7', metavar='K')
    parser_process_query_data.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)),
                                      default=mp.cpu_count(),
                                      help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                           'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_process_query_data.add_argument('-pseudocount', action='store_true',
                                      help='Computes k-mer counts with 0.5 pseudocount added to each frequency value')

    parser_process_query_data.add_argument('-classifier_model',
                                 help='Classification model path')
    parser_process_query_data.add_argument('-cl_seed', type=int, default=seed, help='Clssification random seed. ' +
                                                                               'Default: {}'.format(seed))

    parser_process_query_data.add_argument('-distance_model',
                              help='Directory of models and embeddings (dir of model_subtree_INDEX.ckpt and embeddings_subtree_INDEX.csv files for backbone)')
    parser_process_query_data.add_argument('-di_seed', type=int, default=seed, help='Query random seed. ' +
                                                                                    'Default: {}'.format(seed))


    parser_process_query_data.set_defaults(func=process_query_data)



    ### To invoke
    ### python main.py get_chunks -input_dir ../toy_example/train_tree_fna - output_dir ../toy_example/train_tree_kf
    ### python main.py get_chunks -input_dir ../toy_example/test_fna -output_dir ../toy_example/test_kf
    ### python main.py get_chunks -input_dir ../filt_10k -output_dir ../filt_10k_out

    ### python -m kf2vec.main get_chunks -input_dir ../toy_example/train_tree_fna -output_dir ../toy_example/train_tree_chunks

    parser_chunks = subparsers.add_parser('get_chunks',
                                        description='Process a library of reference genome-skims or assemblies')
    parser_chunks.add_argument('-input_dir',
                             help='Directory of input genomes or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_chunks.add_argument('-output_dir',
                             help='Directory for k-mer frequency outputs (dir for .kf files)')
    parser_chunks.add_argument('-k', type=int, choices=list(range(min_k_len, max_k_len + 1)), default=default_k_len,
                             help='K-mer length [{}-{}]. '.format(min_k_len, max_k_len) +
                                  'Default: {}'.format(default_k_len), metavar='K')
    parser_chunks.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                             help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                  'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_chunks.add_argument('-pseudocount', action='store_true',
                             help='Computes k-mer counts with 0.5 pseudocount added to each frequency value')
    # parser_chunks.add_argument('-raw_cnt', action='store_true',
    #                          help='Computes raw k-mer counts without normalization')

    parser_chunks.set_defaults(func=get_chunks)



    # Train_model_set_chunks command subparser

    ### To invoke
    ### python main.py train_model_set_chunks -input_dir /Users/nora/PycharmProjects/train_tree_kf  -true_dist /Users/nora/PycharmProjects  -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input
    ### python main.py train_model_set_chunks -input_dir /Users/nora/PycharmProjects/filt_10k_out -input_dir_fullgenomes /Users/nora/PycharmProjects/train_tree_kf -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 1 -o ../toy_example/train_tree_models -clade 1 0

    ### python -m kf2vec.main train_model_set_chunks -input_dir /Users/nora/PycharmProjects/filt_10k_out -input_dir_fullgenomes /Users/nora/PycharmProjects/train_tree_kf -true_dist ../toy_example/train_tree_newick  -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 1 -o ../toy_example/train_tree_models -clade 1 0

    parser_train_model_set_chunks = subparsers.add_parser('train_model_set_chunks',
                                                   description='Trains individual models for each subtree using chunked genomes as input')
    parser_train_model_set_chunks.add_argument('-input_dir',
                                        help='Directory of input k-mer frequencies for chunked assemblies (dir of .kf files for chunked backbone species)')
    parser_train_model_set_chunks.add_argument('-input_dir_fullgenomes',
                                               help='Directory of input k-mer frequencies for full assemblies (dir of .kf files for full backbone species)')
    parser_train_model_set_chunks.add_argument('-true_dist',
                                        help='Directory of distamce matrices for backbone subtrees (dir of *subtree_INDEX.di_mtrx files for backbone)')
    parser_train_model_set_chunks.add_argument('-subtrees',
                                        help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    parser_train_model_set_chunks.add_argument('-e', type=int, default=default_di_epochs, help='Number of epochs. ' +
                                                                                        'Default: {}'.format(
                                                                                            default_di_epochs))
    parser_train_model_set_chunks.add_argument('-hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size. ' +
                                                                                              'Default: {}'.format(
                                                                                                  hidden_size_fc1))
    parser_train_model_set_chunks.add_argument('-embed_sz', type=int, default=embedding_size, help='Embedding size. ' +
                                                                                            'Default: {}'.format(
                                                                                                embedding_size))
    parser_train_model_set_chunks.add_argument('-batch_sz', type=int, default=batch_size, help='Batch size. ' +
                                                                                        'Default: {}'.format(
                                                                                            batch_size))
    parser_train_model_set_chunks.add_argument('-lr', type=float, default=learning_rate, help='Start learning rate. ' +
                                                                                       'Default: {}'.format(
                                                                                           learning_rate))
    parser_train_model_set_chunks.add_argument('-lr_min', type=float, default=learning_rate_min,
                                        help='Minimum learning rate. ' +
                                             'Default: {}'.format(learning_rate_min))
    parser_train_model_set_chunks.add_argument('-lr_decay', type=float, default=learning_rate_decay,
                                        help='Learning rate decay. ' +
                                             'Default: {}'.format(learning_rate_decay))
    parser_train_model_set_chunks.add_argument('-clade', type=int, nargs='*', help='Clade number to train. ' +
                                                                                     'Default: all')
    parser_train_model_set_chunks.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                              'Default: {}'.format(seed))
    parser_train_model_set_chunks.add_argument('-cap', action='store_true',
                                      help='Reduces memory consuption for input dataset (caps k-mer frequences at maximum of 255)')
    parser_train_model_set_chunks.add_argument('-o',
                                        help='Model output path')

    parser_train_model_set_chunks.set_defaults(func=train_model_set_chunks)



    # Train_classifier_chunks command subparser

    ### To invoke
    ### python main.py train_classifier_chunks -input_dir /Users/nora/PycharmProjects/train_tree_kf -subtrees /Users/nora/PycharmProjects/my_test.subtrees -e 1 -o /Users/nora/PycharmProjects/my_toy_input

    ### python main.py train_classifier_chunks -input_dir ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10 -o ../toy_example/train_tree_models
    ### python main.py train_classifier_chunks -input_dir ../toy_example/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10  -hidden_sz 2000 -batch_sz 32 -o ../toy_example/train_tree_models

    ### Tested this locally
    ### python main.py train_classifier_chunks -input_dir /Users/nora/PycharmProjects/filt_10k_out -input_dir_fullgenomes /Users/nora/PycharmProjects/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10  -o ../toy_example/train_tree_models
    ### with cap: python main.py train_classifier_chunks -input_dir /Users/nora/PycharmProjects/filt_10k_out -input_dir_fullgenomes /Users/nora/PycharmProjects/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10 -cap  -o ../toy_example/train_tree_models

    ### Module
    ### with cap: python -m kf2vec.main train_classifier_chunks -input_dir /Users/nora/PycharmProjects/filt_10k_out -input_dir_fullgenomes /Users/nora/PycharmProjects/train_tree_kf -subtrees ../toy_example/train_tree_newick/train_tree.subtrees -e 10 -cap  -o ../toy_example/train_tree_models

    parser_trclas_chunks = subparsers.add_parser('train_classifier_chunks',
                                          description='Train classifier model based on backbone subtrees (genomes split into chunks)')
    parser_trclas_chunks.add_argument('-input_dir',
                               help='Directory of input k-mer frequencies for assemblies or reads (dir of .kf files for backbone)')
    parser_trclas_chunks.add_argument('-input_dir_fullgenomes',
                                               help='Directory of input k-mer frequencies for full assemblies (dir of .kf files for full backbone species)')
    parser_trclas_chunks.add_argument('-subtrees',
                               help='Classification file with subtrees information obtained from divide_tree command (a .subtrees format)')
    # parser_trclas_chunks.add_argument('-e', type=int, metavar='', choices=list(range(1, max_cl_epochs)), default=default_cl_epochs, help='Epochs [1-{}]. '.format(max_cl_epochs-1) +
    #                                                                                     'Default: {}'.format(default_cl_epochs))
    parser_trclas_chunks.add_argument('-e', type=int, default=default_cl_epochs, help='Number of epochs. ' +
                                                                               'Default: {}'.format(default_cl_epochs))
    parser_trclas_chunks.add_argument('-hidden_sz', type=int, default=hidden_size_fc1, help='Hidden size. ' +
                                                                                     'Default: {}'.format(
                                                                                         hidden_size_fc1))
    parser_trclas_chunks.add_argument('-batch_sz', type=int, default=batch_size, help='Batch size. ' +
                                                                               'Default: {}'.format(batch_size))
    parser_trclas_chunks.add_argument('-lr', type=float, default=learning_rate, help='Start learning rate. ' +
                                                                              'Default: {}'.format(learning_rate))
    parser_trclas_chunks.add_argument('-lr_min', type=float, default=learning_rate_min, help='Minimum learning rate. ' +
                                                                                      'Default: {}'.format(
                                                                                          learning_rate_min))
    parser_trclas_chunks.add_argument('-lr_decay', type=float, default=learning_rate_decay, help='Learning rate decay. ' +
                                                                                          'Default: {}'.format(
                                                                                              learning_rate_decay))
    parser_trclas_chunks.add_argument('-seed', type=int, default=seed, help='Random seed. ' +
                                                                     'Default: {}'.format(seed))
    parser_trclas_chunks.add_argument('-mask', action='store_true',
                             help=argparse.SUPPRESS) #'Masks low complexity k-mers in input features (reduces input dimension)'
    parser_trclas_chunks.add_argument('-cap', action='store_true',
                                      help='Reduces memory consuption for input dataset (caps k-mer frequences at maximum of 255)')
    parser_trclas_chunks.add_argument('-o',
                               help='Model output path')

    parser_trclas_chunks.set_defaults(func=train_classifier_chunks)


    args = parser.parse_args()
    #args.func(args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm oop')
    main()




