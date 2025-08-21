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
import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.parallel import DataParallel
import sklearn
from sklearn.metrics import accuracy_score
#import pyarrow as pa
import multiprocessing as mp



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
from collections import deque









# Hyper-parameters
#input_size = 32896    # Canonical kmer count for k=8
#input_size = 8192      # Canonical kmer count for k=7
#N = 10570             # Number of samples in dataset
#hidden_size_fc1 = 2000
#hidden_size_fc1 = 2048
#hidden_size_fc2 = 2000
#embedding_size = 2 ** math.floor(math.log2(10 * N ** (1 / 2)))
#embedding_size = 1024
start_epoch = 0
#num_epochs = 4000
#batch_size = 16

#learning_rate = 0.00001           # 1e-4
#learning_rate_decay = 2000
learning_rate_base = 0.1
learning_rate_update_freq = 100

features_scaler = 1e4

#train_test_split = 0.95
#weight_decay = 1e-5     # L2 regularization
#resume = False



def train_model_set_chunks_func(features_folder, input_dir_fullgenomes, features_csv, clades_info, true_dist_matrix, num_epochs, hidden_size_fc1, embedding_size, in_batch_sz, in_lr, in_lr_min, in_lr_decay, clades_to_train, seed, cap_data, model_filepath):

    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    num_workers_req = 4 * num_gpus

    #### Dataset parameters ####
    params = {'batch_size': in_batch_sz,
              'shuffle': True,
              'num_workers': num_workers_req}


    since = time.time()

    level = logging.INFO
    format = '%(message)s'


    handlers = [logging.FileHandler(os.path.join(model_filepath, 'train_model_{}_clade_{}.log'.format(time.strftime("%Y%m%d_%H%M%S"), ('_'.join([str(elem) for elem in clades_to_train]) if clades_to_train != None else 'all')) ), 'w+'), logging.StreamHandler()]

    #logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s: %(message)s', handlers=handlers)

    logging.basicConfig(level=level, format=format, handlers=handlers)
    # logging.info('Hey, this is working!')


    #######################################################################
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #######################################################################
    logging.info('\n==> Input arguments...\n')

    logging.info('Feature directory: {}'.format(features_folder))
    logging.info('Clades information: {}'.format(clades_info))
    logging.info('Ground truth directory: {}'.format(true_dist_matrix))



    #######################################################################
    # Input parameters
    logging.info('\n==> Parameters...\n')


    logging.info('GPU Support: {}'.format('Yes' if str(device) != 'cpu' else 'No'))
    logging.info('GPU Count: {}'.format(num_gpus))
    logging.info('Workers Count: {}'.format(num_workers_req))
    logging.info('CPU Count: {}'.format(num_cpus))
    logging.info('Hidden Size fc1: {}'.format(hidden_size_fc1))
    # logging.info('Hidden Size fc2: {}'.format(hidden_size_fc2))
    # logging.info('Hidden Size fc3: {}'.format(hidden_size_fc3))
    logging.info('Embedding Size: {}'.format(embedding_size))
    logging.info('Starting Epoch: {}'.format(start_epoch))
    logging.info('Total Epochs: {}'.format(num_epochs))
    logging.info('Batch Size: {}'.format(in_batch_sz))
    logging.info('Learning Rate: %g', in_lr)
    logging.info('Learning Rate Min: %g', in_lr_min)
    logging.info('Learning Rate Decay: %g', in_lr_decay)
    logging.info('Clades to train: {}'.format(' '.join([str(elem) for elem in clades_to_train]) if clades_to_train != None else 'all'))
    logging.info('Random Seed: {}'.format(seed))
    logging.info('Cap kmer frequencies: {}'.format(cap_data))
    # logging.info('Resuming Training:{}'.format('Yes' if resume else 'No'))


    #######################################################################
    # Read classification information

    logging.info('\n==> Subtree training...\n')

    classification_df = pd.read_csv(clades_info, sep=' ', header=0)  # read from csv
    classification_df["clade"] = classification_df["clade"].astype(int)
    class_count = classification_df.clade.unique()
    #class_count = np.array([1]) # Can be used if training for a single clade is needed
    #class_count = np.array([1, 5, 6]) # Train for a single clade 1
    #class_count = np.array([5])  # Train for a single clade 1
    if clades_to_train != None:
        class_count = np.array(clades_to_train)



    # Compute total number of classes
    logging.info('Number of Classes: {}'.format(class_count.size))


    current_class_ids = {}


    # Making a dataframe of sample names
    feat_basename = [os.path.basename(i) for i in features_csv]
    feat_samples_names = [f.rsplit('.kf', 1)[0] for f in feat_basename]
    df_feat_samples_names = pd.DataFrame(feat_samples_names)
    df_feat_samples_names.set_index(0, inplace=True)


    # Extract pathway and file extension
    path_name = os.path.split(features_csv[0])[0]
    ext_name = os.path.splitext(os.path.basename(features_csv[0]))[1] # Redundant


    for c in class_count:

        current_clade = classification_df.loc[classification_df["clade"]==c]
        current_class_ids[c] = current_clade["genome"].to_list()


        #######################################################################
        logging.info('\n==> Working on subtree {}...\n'.format(c))



        #######################################################################
        # Prepare dataset
        logging.info('\n==> Preparing Data...\n')


        # Subset feature input for a given clade
        feature_input = df_feat_samples_names.loc[df_feat_samples_names.index.isin(current_class_ids[c])]


        # Get input dimensions
        tmp_feature_input = pd.read_csv(features_csv[0], index_col=0, header=None, sep= ',')
        input_size = np.shape(tmp_feature_input)[1]


        #feature_input = feature_input.iloc[:,:]*features_scaler

        logging.info("Dimensions of feature matrix rows: {}, cols: {}".format(np.shape(feature_input)[0], input_size))

        # #######################################################################
        # Get names
        backbone_names = feature_input.index.tolist()

        #######################################################################

        # Create mapping dictionaries
        #feature_input = read_feat_mtrx(dataset_features)  # read from dataframe (1000 columns, normalized)
        label_idx_dict, idx_label_dict = get_label_idx_maps(feature_input)    # convert first column into dict


        ##### NEED TO IMPLEMENT ERROR IF DATA IS NOT FOUND #####

        # Load dictionary with true distances
        dir_list = os.listdir(true_dist_matrix)
        dir_list = [i for i in dir_list if  "_subtree_{}.di_mtrx".format(c) in i]


        pdf = pd.read_csv(os.path.join(true_dist_matrix, dir_list[0]), sep='\t', header=0, index_col=0) # read DEPP distance matrix
        pdf_sorted = sort_df(pdf, label_idx_dict)     # sort in the same order as input features


        logging.info("Dimensions of true distance matrix rows: {}, cols: {}".format(np.shape(pdf_sorted)[0], np.shape(pdf_sorted)[1]))
        #logging.info(pdf_sorted)

        #######################################################################
        # Prepare train/test dataset split
        partition = {}
        partition['train'] = backbone_names
        partition['test'] = []


        # Read samples into list
        feature_input_list = [os.path.join(*[path_name, i + ext_name]) for i in backbone_names]

        # Add datasets into the list
        #clade_features_df = [None] * len(backbone_names)

        feature_input_list_mp = [[f, input_size] for f in feature_input_list]


        if cap_data:
            with mp.Pool() as pool:
                clade_features_df = pool.map(my_read_csv_chunk_capped, feature_input_list_mp)
        else:
            with mp.Pool() as pool:
                clade_features_df = pool.map(my_read_csv_chunk, feature_input_list_mp)



        # for d in range (0, len(clade_features_df)):
        #     clade_features_df[d] = pd.read_csv(feature_input_list[d], sep=',', header=None, low_memory=True, dtype = column_types, engine = "pyarrow").iloc[:, 1:].to_numpy()

        # usecols=range(1, input_size+1),
        #dtype={**{0:str},**{i:np.uint16 for i in range(1,input_size+1)}},

        # Custom dataset
        training_set = datasets.Dataset_chunks_2rows(clade_features_df, partition['train'], label_idx_dict, features_scaler)
        test_set = datasets.Dataset_chunks_2rows(clade_features_df, partition['test'], label_idx_dict, features_scaler)
        #val_set = datasets.Dataset(dataset_fname, partition['val'])


        train_size = training_set.__len__()
        test_size = test_set.__len__()
        # val_size = testset.__len__()


        # Data loader
        train_loader = torch.utils.data.DataLoader(training_set, **params)

        if test_size != 0:
            test_loader = torch.utils.data.DataLoader(test_set, **params)
        #val_loader = torch.utils.data.DataLoader(val_set, **params)


        #check whether data are read correctly
        # for i, (data, labels) in enumerate(train_loader):
        #     logging.info(data)
        #     logging.info(labels)


        logging.info('Number of Train Samples: {}'.format(train_size))
        # logging.info('Number of Test Samples: {}'.format(test_size))
        # logging.info('Number of Validation Samples:{}'.format(val_size))


        #######################################################################
        # Model
        logging.info('\n==> Building model...\n')

        #model = models.NeuralNet(input_size, hidden_size_fc1, embedding_size).to(device)
        model = models.NeuralNet(input_size, hidden_size_fc1, embedding_size)
        model_name = "NeuralNet"
        model = DataParallel(model).to(device)



        # Custom weight initialization
        #model.apply(weight_init)
        #model.to(device)


        logging.info('\n==> Model parameters----------')
        # for parameter in model.parameters():
        #     logging.info(parameter.shape)

        # for name, param in model.named_parameters():
        #     logging.info("{} : {}".format(name, param.shape))

        # list(model.parameters())[0].grad

        # Total number of parameters
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        logging.info("Total parameters: {}".format(pytorch_total_params))

        # Total number of trainable parameters
        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("Trainable parameters: {}".format(pytorch_trainable_params))



        #######################################################################
        # Loss and optimizer
        criterion_a = losses.Loss_chunks()
        criterion_a.to(device)


        # Construct optimizer object
        optimizer = torch.optim.Adam(model.parameters(), lr=in_lr)


        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)
        logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))


        #######################################################################
        # Training model
        logging.info('\n==> Training model...\n')

        total_step = len(train_loader)

        early_stop_thresh = 5
        lowest_loss = math.inf
        best_epoch = -1


        #######################################################################
        # Declare lowest loss and best epoch for consecuitive epochs

        consec_lowest_loss = math.inf
        consec_best_epoch = -1

        # Compute count of stopping epochs

        stop_epochs = int(math.ceil(len(backbone_names) / in_batch_sz * 2))
        lq = deque([float('nan')] * stop_epochs, maxlen=stop_epochs)
        logging.info("Stopping epochs: {}".format(stop_epochs))

        #######################################################################


        for epoch in range(num_epochs):


            #######################################################################
            # Train the model
            model.train()

            train_loss = 0.0
            items_count = 0
            num_batches = len(train_loader)


            for i, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1, input_size).to(device)
                #logging.info('images: {}'.format(images))

                # Repeat labels twice so that true matrix rows are pulled twice
                labels = torch.repeat_interleave(labels, 2, dim=0)
                # logging.info('labels: {}'.format(labels))
                # logging.info('names: {}'.format([idx_label_dict[i] for i in labels.tolist()]))


                real_dist = pairwise_true_dist(labels, pdf_sorted).to(device) # get true distances
                #logging.info('real_dist: {}'.format(real_dist))


                # Alternative way to duplicate tensor
                # real_dist =  torch.repeat_interleave(real_dist,  2, dim=1)
                # real_dist = torch.repeat_interleave(real_dist, 2, dim=0)

                # Forward pass
                outputs = model(images.float())
                #logging.info('outputs: {}'.format(outputs))

                train_dist = pairwise_train_dist(outputs)
                #logging.info('train_dist: {}'.format(train_dist))


                loss_a  = criterion_a(train_dist, real_dist)
                if epoch > 5 and loss_a.item() > 0.2:
                    logging.info('Epoch [{}/{}], Step [{}/{}], Outlier: {:.20f} batch size: {}'
                                 .format(epoch + 1, num_epochs, i + 1, total_step, loss_a.item(), images.shape[0]))

                batch_elmts = images.shape[0]/2
                train_loss += loss_a.item() * batch_elmts # running loss
                items_count += batch_elmts
                #print(train_loss)


                loss = loss_a

                if torch.isnan(loss):
                    logging.info('Loss: {}'.format(loss))


                # Backward and optimize
                optimizer.zero_grad()

                # if torch.isnan(loss):
                #     logging.info('Loss grad before: {}'.format(loss.grad))


                loss.backward()

                # if torch.isnan(loss):
                #     logging.info('Loss grad after: {}'.format(loss.grad))

                optimizer.step()


            train_loss /= items_count

            # Save best model if train loss lower than lowest loss
            if  train_loss < lowest_loss:
                lowest_loss = train_loss
                best_epoch = epoch

                # Save best model
                # Access the underlying model
                actual_model = model.module
                actual_model.to('cpu')
                save_trained_model(model_name, input_size, hidden_size_fc1, embedding_size,
                                   actual_model.state_dict(), model_filepath, model_filename = "model_subtree_{}.ckpt".format(c))
                actual_model.to(device)

            # elif epoch - best_epoch > early_stop_thresh:
            #     print("Early stopped training at epoch %d" % epoch)
            #     break  # terminate the training loop



            # Add train_loss to loss queue from the left and compute mean
            lq.appendleft(train_loss)
            lq_mean = np.nanmean(lq)

            # Save consecuitive model
            if  lq_mean < consec_lowest_loss:
                consec_lowest_loss = lq_mean
                consec_best_epoch = epoch

                # Save the model
                actual_model = model.module
                actual_model.to('cpu')
                save_trained_model(model_name, input_size, hidden_size_fc1, embedding_size,
                                   actual_model.state_dict(), model_filepath, model_filename="consec_model_subtree_{}.ckpt".format(c))
                actual_model.to(device)


            time_elapsed = time.time() - since
            hrs, _min, sec = hms(time_elapsed)

            if (i+1) % 1 == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Train loss: {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, train_loss, hrs, _min, sec))


            ###########################################################################
            # Test the model

            if test_size != 0:

                model.eval()

                test_loss = 0.0
                items_count = 0
                num_batches = len(test_loader)

                with torch.no_grad():

                    for i, (images, labels) in enumerate(test_loader):

                        images = images.reshape(-1, input_size).to(device)



                        # Repeat labels twice so that true matrix rows are pulled twice
                        labels = torch.repeat_interleave(labels, 2, dim=0)

                        real_dist = pairwise_true_dist(labels, pdf_sorted).to(device)  # get true distances

                        outputs = model(images.float())
                        #train_dist = pairwise_train_dist(outputs)
                        train_dist = pairwise_train_dist(outputs)


                        loss_a = criterion_a(train_dist, real_dist)
                        batch_elmts = images.shape[0] / 2
                        test_loss += loss_a.item() *  batch_elmts # running loss
                        items_count +=  batch_elmts
                        loss = loss_a


                test_loss /= items_count

                time_elapsed = time.time() - since
                hrs, _min, sec = hms(time_elapsed)

                if (i+1) % 1 == 0:
                    logging.info('Epoch [{}/{}], Step [{}/{}], Test loss: {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                          .format(epoch + 1, num_epochs, i + 1, num_batches, test_loss, hrs, _min, sec))


            # Output current learning rate
            curr_lr = optimizer.param_groups[0]['lr']
            logging.info('Epoch {}\t \
                  LR:{:.20f}'.format(epoch + 1, curr_lr))

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = curr_lr+0.1



            # Update learning rate
            if (epoch) % learning_rate_update_freq == 0:
                lr = in_lr_min + in_lr * (learning_rate_base ** (epoch / in_lr_decay))


                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


        #######################################################################
        # Output
        logging.info('Best Epoch [{}/{}], Lowest loss: {:.20f}'
                     .format(best_epoch + 1, num_epochs, lowest_loss))
        # Output
        logging.info('Best consecutive Epoch [{}/{}], Lowest loss: {:.20f}'
                     .format(consec_best_epoch + 1, num_epochs, consec_lowest_loss))

        #######################################################################
        # Save last model
        # Access the underlying model
        actual_model = model.module
        actual_model.cpu()
        save_trained_model(model_name, input_size, hidden_size_fc1, embedding_size,
                           actual_model.state_dict(), model_filepath, model_filename="last_model_subtree_{}.ckpt".format(c))


        #######################################################################
        # Prepare data to compute embeddings for backbone species

        # Read samples into list
        embed_lst_full_pathname = [os.path.join(*[input_dir_fullgenomes, i + ".kf"]) for i in backbone_names]

        with mp.Pool() as pool:
            embed_lst_tmp = pool.map(my_read_csv, embed_lst_full_pathname)

        # Concatenate list of dataframes and turn into numpy array
        embed_lst_df = pd.concat(embed_lst_tmp)
        embed_lst_df = embed_lst_df.iloc[:, :] * features_scaler
        embed_lst = embed_lst_df.to_numpy()


        #######################################################################
        ##### Load best model #####
        # NEED TO CHECK IF FILE EXISTS
        # Not sure if I need to redefine to get rid of DataParallel
        model = models.NeuralNet(input_size, hidden_size_fc1, embedding_size)

        state = torch.load(os.path.join(model_filepath, "model_subtree_{}.ckpt".format(c)))
        model.load_state_dict(state['state_dict'])

        model.to("cpu")

        #### Output embeddings and distortions ####
        model.eval()

        with torch.no_grad():
            outputs = model(torch.from_numpy(embed_lst).float())

        df_outputs, df_embeddings = compute_embeddings(outputs, backbone_names)

        logging.info("Dimensions of distortion matrix rows:{} cols:{}".format(len(df_outputs), len(df_outputs.columns)))
        df_outputs.to_csv(os.path.join(model_filepath, 'distortions_subtree_{}.csv'.format(c)), index=False, sep='\t')

        logging.info("Dimensions of embedding output rows:{} cols:{}".format(len(df_embeddings), len(df_embeddings.columns)))
        df_embeddings.to_csv(os.path.join(model_filepath, 'embeddings_subtree_{}.csv'.format(c)), index=False, sep='\t', header = False)

        """
        #######################################################################
        ##### Load last model #####
        # NEED TO CHECK IF FILE EXISTS
        state = torch.load(os.path.join(model_filepath, "last_model_subtree_{}.ckpt".format(c)))

        model.load_state_dict(state['state_dict'])
        model.to("cpu")

        #### Output embeddings and distortions ####
        model.eval()

        with torch.no_grad():
            outputs = model(torch.from_numpy(embed_lst).float())

        df_outputs, df_embeddings = compute_embeddings(outputs, backbone_names)

        df_outputs.to_csv(os.path.join(model_filepath, 'last_distortions_subtree_{}.csv'.format(c)), index=False, sep='\t')
        df_embeddings.to_csv(os.path.join(model_filepath, 'last_embeddings_subtree_{}.csv'.format(c)), index=False, sep='\t',header=False)


        #######################################################################
        ##### Load consecuitive model #####
        # NEED TO CHECK IF FILE EXISTS
        state = torch.load(os.path.join(model_filepath, "consec_model_subtree_{}.ckpt".format(c)))

        model.load_state_dict(state['state_dict'])
        model.to("cpu")

        #### Output embeddings and distortions ####
        model.eval()

        with torch.no_grad():
            outputs = model(torch.from_numpy(embed_lst).float())

        df_outputs, df_embeddings = compute_embeddings(outputs, backbone_names)

        df_outputs.to_csv(os.path.join(model_filepath, 'consec_distortions_subtree_{}.csv'.format(c)), index=False, sep='\t')
        df_embeddings.to_csv(os.path.join(model_filepath, 'consec_embeddings_subtree_{}.csv'.format(c)), index=False, sep='\t', header=False)
        """


        logging.info('\n==> Training for subtree {} completed!\n'.format(c))

        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)
        logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))



    logging.info('\n==> Training Completed!\n'.format(c))

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))




