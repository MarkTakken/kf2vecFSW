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




# Hyper-parameters
#input_size = 32896    # Canonical kmer count for k=8
#input_size = 8192      # Canonical kmer count for k=7
#N = 10570             # Number of samples in dataset
#hidden_size_fc1 = 4000
#hidden_size_fc1 = 2048
#hidden_size_fc2 = 2000
#embedding_size = 2 ** math.floor(math.log2(10 * N ** (1 / 2)))
#embedding_size = 1024
start_epoch = 0
#num_epochs = 8000
#batch_size = 16

# learning_rate = 0.00001           # 1e-4
# learning_rate_decay = 2000
learning_rate_base = 0.1
learning_rate_update_freq = 100


features_scaler = 1e4

#train_test_split = 0.95
#weight_decay = 1e-5     # L2 regularization
#resume = False



def train_classifier_model_chunks_func(features_folder, input_dir_fullgenomes, features_csv, clades_info, num_epochs, hidden_size_fc1, in_batch_sz, in_lr, in_lr_min, in_lr_decay, seed, custom_mask, cap_data, model_filepath):

    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)

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
    handlers = [logging.FileHandler(os.path.join(model_filepath, 'train_classifier_{}.log'.format(time.strftime("%Y%m%d_%H%M%S")) ), 'w+'), logging.StreamHandler()]

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
    # logging.info('Embedding Size: {}'.format(embedding_size))
    logging.info('Starting Epoch: {}'.format(start_epoch))
    logging.info('Total Epochs: {}'.format(num_epochs))
    logging.info('Batch Size: {}'.format(in_batch_sz))
    logging.info('Learning Rate: %g', in_lr)
    logging.info('Learning Rate Min: %g', in_lr_min)
    logging.info('Learning Rate Decay: %g', in_lr_decay)
    logging.info('Random Seed: {}'.format(seed))
    #logging.info('Resuming Training:{}'.format('Yes' if resume else 'No'))
    logging.info('Masking: {}'.format(custom_mask))
    logging.info('Cap kmer frequencies: {}'.format(cap_data))


    #######################################################################
    # Prepare dataset
    logging.info('\n==> Preparing Data...\n')

    # Making a dataframe of sample names
    feat_basename = [os.path.basename(i) for i in features_csv]
    feat_samples_names = [f.rsplit('.kf', 1)[0] for f in feat_basename]
    df_feat_samples_names = pd.DataFrame(feat_samples_names)
    df_feat_samples_names.set_index(0, inplace=True)

    # Extract pathway and file extension
    path_name = os.path.split(features_csv[0])[0]
    ext_name = os.path.splitext(os.path.basename(features_csv[0]))[1]  # Redundant


    # Subset feature input for all  clades
    feature_input = df_feat_samples_names


    # Get input dimensions
    tmp_feature_input = pd.read_csv(features_csv[0], index_col=0, header=None, sep=',')
    input_size = np.shape(tmp_feature_input)[1]
    logging.info("Dimensions of feature matrix rows: {}, cols: {}".format(np.shape(feature_input)[0], input_size))

    # Create mask
    if custom_mask:

        if input_size == 8192:  # k = 7
            my_alphabet_kmers = pd.read_csv("test_kmers_7_sorted", sep=" ", header=None, names=["kmer"])
        elif input_size == 2080:  # k = 6
            my_alphabet_kmers = pd.read_csv("test_kmers_6_sorted", sep=" ", header=None, names=["kmer"])
        elif input_size == 32:  # k = 3
            my_alphabet_kmers = pd.read_csv("vocab_generator_k3C_fin.fa", sep=" ", header=None, names=["kmer"])
        elif input_size == 136:  # k = 4
            my_alphabet_kmers = pd.read_csv("vocab_generator_k4C_fin.fa", sep=" ", header=None, names=["kmer"])
        elif input_size == 512:  # k = 5
            my_alphabet_kmers = pd.read_csv("vocab_generator_k5C_fin.fa", sep=" ", header=None, names=["kmer"])
        elif input_size == 32896:  # k = 8
            my_alphabet_kmers = pd.read_csv("vocab_generator_k8C_fin.fa", sep=" ", header=None, names=["kmer"])
        elif input_size == 131072:  # k = 9
            my_alphabet_kmers = pd.read_csv("vocab_generator_k9C_fin.fa", sep=" ", header=None, names=["kmer"])

        my_mask = list((my_alphabet_kmers["kmer"].apply(set).apply(len) > 2))

        tmp_feature_input = tmp_feature_input.iloc[:, [z for z in range(0, input_size) if my_mask[z] == True]]

        # Resize input
        input_size = np.shape(tmp_feature_input)[1]
        logging.info("Dimensions of feature matrix after masking rows: {}, cols: {}".format(np.shape(feature_input)[0], input_size))


    # feature_input = feature_input.iloc[:,:]*features_scaler
    # input_size = np.shape(feature_input)[1]
    # logging.info("Dimensions of feature matrix rows: {}, cols: {}".format(np.shape(feature_input)[0], np.shape(feature_input)[1]))

    # #######################################################################
    # Get names
    backbone_names = feature_input.index.tolist()

    #######################################################################


    #feature_input = read_feat_mtrx(dataset_features)  # read from dataframe (1000 columns, normalized)
    label_idx_dict, idx_label_dict = get_label_idx_maps(feature_input)    # convert first column into dict

    clade_input = pd.read_csv(clades_info, sep=' ', header=0, index_col=0)  # read from csv

    clade_input_sorted = sort_df(clade_input, label_idx_dict)
    clade_input_sorted = np.concatenate(clade_input_sorted)



    # Prepare train/test dataset split
    partition = {}
    partition['train'] = backbone_names
    partition['test'] = []


    # Read samples into list
    feature_input_list = [os.path.join(*[path_name, i + ext_name]) for i in backbone_names]

    feature_input_list_mp = [[f, input_size] for f in feature_input_list]

    if cap_data:
        with mp.Pool() as pool:
            clade_features_df = pool.map(my_read_csv_chunk_capped, feature_input_list_mp)
    else:
        with mp.Pool() as pool:
            clade_features_df = pool.map(my_read_csv_chunk, feature_input_list_mp)



    # for d in range(0, len(clade_features_df)):
    #     clade_features_df[d] = pd.read_csv(feature_input_list[d], sep=',', index_col=0, header=None,
    #                                        dtype={**{0: str}, **{i: np.uint16 for i in range(1, input_size + 1)}},
    #                                        low_memory=True, usecols = [z for z in tmp_feature_input.reset_index().columns.tolist()])
    #

    # Apply mask to inputs
    # if custom_mask:
    #
    #     for d in range(0, len(clade_features_df)):
    #         clade_features_df[d] = clade_features_df[d].iloc[:, [z for z in range(0, input_size) if my_mask[z] == True]]
    #         print(clade_features_df[d])




    # Custom dataset
    # training_set = datasets.Dataset_chunks(clade_features_df, partition['train'], label_idx_dict,
    #                                              features_scaler)
    # test_set = datasets.Dataset_chunks(clade_features_df, partition['test'], label_idx_dict, features_scaler)
    # val_set = datasets.Dataset(dataset_fname, partition['val'])
    training_set = datasets.Dataset_chunks_1row(clade_features_df, partition['train'], label_idx_dict,
                                                 features_scaler)
    test_set = datasets.Dataset_chunks_1row(clade_features_df, partition['test'], label_idx_dict, features_scaler)


    train_size = training_set.__len__()
    test_size = test_set.__len__()
    # val_size = testset.__len__()


    # Data loader
    train_loader = torch.utils.data.DataLoader(training_set, **params)


    if test_size !=0:
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


    class_count = np.unique(clade_input_sorted).size
    logging.info('Number of Classes: {}'.format(class_count))


    #model = models.NeuralNetClassifierOnly(input_size, hidden_size_fc1, class_count).to(device)
    model = models.NeuralNetClassifierOnly(input_size, hidden_size_fc1, class_count)
    model = DataParallel(model).to(device)

    # Custom weight initialization
    #model.apply(weight_init)
    #model.to(device)

    logging.info('\n==> Model parameters----------')
    # for parameter in model.parameters():
    #     logging.info(parameter)

    # list(model.parameters())[0].grad

    # Total number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info("Total parameters: {}".format(pytorch_total_params))

    # Total number of trainable parameters
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable parameters: {}".format(pytorch_trainable_params))


    #######################################################################
    # Custom parameter initialization
    #logging.info('\n==> Custom parameter initialization...\n')


    #######################################################################
    # Loss and optimizer
    criterion_b = nn.NLLLoss()
    criterion_b.to(device)


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
    highest_acc = -1
    best_epoch = -1


    for epoch in range(num_epochs):

        #######################################################################
        # Train the model
        model.train()
        train_loss2 = 0.0
        running_acc = 0.0
        items_count = 0

        num_batches = len(train_loader)


        for i, (images, labels) in enumerate(train_loader):

            images = images.reshape(-1, input_size).to(device)

            true_class = torch.from_numpy(clade_input_sorted[np.ix_(list(labels))]).to(device)

            # Forward pass
            model_class = model(images.float())

            loss_b = criterion_b(model_class, true_class)

            train_loss2 += loss_b.item() * images.shape[0]
            items_count += images.shape[0]
            loss  = loss_b


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            ps = torch.exp(model_class)
            top_p, top_class = ps.topk(1, dim=1)


            running_acc += accuracy_score(true_class.to('cpu'), top_class.to('cpu')) * images.shape[0]


        train_loss2 /=  items_count
        running_acc /=  items_count

        # Save model if train loss lower than lowest loss
        if train_loss2 < lowest_loss:
            lowest_loss = train_loss2
            highest_acc = running_acc
            best_epoch = epoch

            # Save the model
            # Access the underlying model
            actual_model = model.module
            actual_model.to('cpu')

            state = {
                'model_name': "NeuralNetClassifierOnly",
                'model_input_size': input_size,
                'model_hidden_size_fc1': hidden_size_fc1,
                'model_class_count': class_count,
                'state_dict': actual_model.state_dict(),
                # 'optimizer': optimizer.state_dict()
            }

            torch.save(state, (os.path.join(model_filepath, "classifier_model.ckpt")))

            actual_model.to(device)

        # elif epoch - best_epoch > early_stop_thresh:
        #     print("Early stopped training at epoch %d" % epoch)
        #     break  # terminate the training loop


        time_elapsed = time.time() - since
        hrs, _min, sec = hms(time_elapsed)

        if (i+1) % 1 == 0:
            logging.info('Epoch [{}/{}], Step [{}/{}], Train loss: {:.20f}, {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, train_loss2, running_acc, hrs, _min, sec))


        ###########################################################################
        # Test the model

        if test_size != 0:

            model.eval()

            test_loss2 = 0.0
            test_running_acc = 0.0
            num_batches = len(test_loader)
            items_count = 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    images = images.reshape(-1, input_size).to(device)


                    true_class = torch.from_numpy(clade_input_sorted[np.ix_(list(labels))]).to(device)


                    model_class = model(images.float())


                    loss_b = criterion_b(model_class, true_class)
                    test_loss2 += loss_b.item() * images.shape[0]
                    items_count += images.shape[0]
                    loss = loss_b


                    ps = torch.exp(model_class)
                    top_p, top_class = ps.topk(1, dim=1)

                    test_running_acc += accuracy_score(true_class.to('cpu'), top_class.to('cpu')) * images.shape[0]


            test_loss2 /= items_count
            test_running_acc /= items_count

            time_elapsed = time.time() - since
            hrs, _min, sec = hms(time_elapsed)

            if (i+1) % 1 == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Test loss: {:.20f}, {:.20f}, Time: {:02d}:{:02d}:{:02d}'
                      .format(epoch + 1, num_epochs, i + 1, num_batches, test_loss2, test_running_acc, hrs, _min, sec))


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
    logging.info('Best Epoch [{}/{}], Lowest loss: {:.20f}, Highest accuracy: {:.20f}'
                 .format(best_epoch + 1, num_epochs, lowest_loss, highest_acc))

    #######################################################################
    ##### Load best model #####
    # Not sure if I need to redefine to get rid of DataParallel
    model = models.NeuralNetClassifierOnly(input_size, hidden_size_fc1, class_count)

    state = torch.load(os.path.join(model_filepath, "classifier_model.ckpt"))

    model.load_state_dict(state['state_dict'])
    model.to("cpu")

    #torch.save(model.state_dict(), 'model.ckpt')
    #torch.save(optimizer.state_dict(), 'optimizer.ckpt')

    #######################################################################
    # Prepare data to compute embeddings for backbone species

    # Read sample names
    embed_lst_full_pathname = [os.path.join(*[input_dir_fullgenomes, i + ".kf"]) for i in backbone_names]

    # Read sample csv files into list
    with mp.Pool() as pool:
        embed_lst_tmp = pool.map(my_read_csv, embed_lst_full_pathname)


    # Concatenate list of dataframes and turn into numpy array
    embed_lst_df = pd.concat(embed_lst_tmp)
    embed_lst_df = embed_lst_df.iloc[:, :] * features_scaler
    embed_lst = embed_lst_df.to_numpy()

    #######################################################################
    # Compute model output for backbone

    model.eval()

    with torch.no_grad():
        model_class = model(torch.from_numpy(embed_lst).float())

    ps = torch.exp(model_class)
    top_p, top_class = ps.topk(1, dim=1)

    # Get names
    # backbone_names = feature_input.index.tolist()
    # print(backbone_names)

    # Detach gradient and convert to numpy
    df_classes = pd.DataFrame(np.hstack((top_class.detach().numpy(), top_p.detach().numpy(), ps.detach().numpy())))

    # Attach species names
    df_classes.columns = ["top_class", "top_p"] + [str(x) for x in list(range(class_count))]
    df_classes.insert(loc=0, column='true_class', value=clade_input_sorted.tolist())
    df_classes.insert(loc=0, column='genome', value=backbone_names)


    # Write to file
    logging.info("Dimensions of class output rows:{} cols:{}".format(len(df_classes), len(df_classes.columns)))
    df_classes.to_csv(os.path.join(model_filepath, "backbone_classes.out"), index=False, sep='\t')

    #######################################################################

    logging.info('\n==> Training Completed!\n')

    time_elapsed = time.time() - since
    hrs, _min, sec = hms(time_elapsed)
    logging.info('Time: {:02d}:{:02d}:{:02d}'.format(hrs, _min, sec))







