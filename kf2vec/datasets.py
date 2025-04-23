import torch
import pandas as pd
import numpy as np
import random


class Dataset_chunks_2rows(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, df, list_IDs, label_idx, features_scaler):
        'Initialization'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.kmer_frame = df
        self.list_IDs = list_IDs
        self.label_idx = label_idx
        self.features_scaler = features_scaler


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Select sample
        ID = self.list_IDs[idx]
        row_num = self.label_idx[ID]

        #print(row_num)

        # Load data and get label
        #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
        #X = self.kmer_frame.iloc[list(row_num), :].values

        # First round of subsetting
        #if len(self.kmer_frame[row_num]) >= 20:
        #    nrows = np.random.randint(low = 1, high = len(self.kmer_frame[row_num])/10)
        #else:
        #    nrows = np.random.randint(low=1, high=len(self.kmer_frame[row_num]))
        c = len(self.kmer_frame[row_num])
        nrows = int(np.floor(np.random.exponential(c / 5)) + 1)
        if nrows > c:
            nrows = np.random.randint(low=1, high=c+1)
        #print("nrows:", nrows,c)
        ix = np.random.randint(low = 0, high = len(self.kmer_frame[row_num])-nrows+1)
        subset = self.kmer_frame[row_num].iloc[ix:ix+nrows, :]


        #print(nrows,ix)
        # Normalize
        tmp = subset.sum(axis=0, numeric_only=True).to_frame()
        tmp[0] = (tmp[0] / tmp[0].sum()).replace(np.inf, np.nan).fillna(0) * self.features_scaler


        X = tmp[0].values

        # print("nrows 1:", c, nrows, ix)
        # print(X)


        # Second round of subsetting
        # if len(self.kmer_frame[row_num]) >= 20:
        #     nrows = np.random.randint(low=1, high=len(self.kmer_frame[row_num]) / 10)
        # else:
        #     nrows = np.random.randint(low=1, high=len(self.kmer_frame[row_num]))

        c = len(self.kmer_frame[row_num])
        nrows = int(np.floor(np.random.exponential(c / 5)) + 1)
        if nrows > c:
            nrows = np.random.randint(low=1, high=c+1)
        #print("nrows:",nrows,c)
        ix = np.random.randint(low=0, high=len(self.kmer_frame[row_num]) - nrows+1)
        subset = self.kmer_frame[row_num].iloc[ix:ix + nrows, :]


        # Normalize
        tmp = subset.sum(axis=0, numeric_only=True).to_frame()
        tmp[0] = (tmp[0] / tmp[0].sum()).replace(np.inf, np.nan).fillna(0) * self.features_scaler
        X2 = tmp[0].values

        # print("nrows 2:", c, nrows, ix)
        # print(X2)

        Z = np.concatenate((X, X2))

        y = row_num

        # X = self.label_idx[ID]
        # y = self.list_IDs[idx]

        #print(Z)
        #print(y)

        return Z, y

class Dataset_chunks_2rows_numpy(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, df, list_IDs, label_idx, features_scaler):
        'Initialization'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.kmer_frame = df
        self.list_IDs = list_IDs
        self.label_idx = label_idx
        self.features_scaler = features_scaler


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Select sample
        ID = self.list_IDs[idx]
        row_num = self.label_idx[ID]


        # Load data and get label
        #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
        #X = self.kmer_frame.iloc[list(row_num), :].values

        # First round of subsetting
        #if len(self.kmer_frame[row_num]) >= 20:
        #    nrows = np.random.randint(low = 1, high = len(self.kmer_frame[row_num])/10)
        #else:
        #    nrows = np.random.randint(low=1, high=len(self.kmer_frame[row_num]))

        c = len(self.kmer_frame[row_num])

        nrows = int(np.floor(np.random.exponential(c / 5)) + 1)
        if nrows > c:
            nrows = np.random.randint(low=1, high=c+1)
        #print("nrows:", nrows,c)
        ix = np.random.randint(low = 0, high = len(self.kmer_frame[row_num])-nrows+1)
        subset = self.kmer_frame[row_num][ix:ix+nrows]


        # Normalize
        tmp = subset.sum(axis=0)
        X = tmp / np.sum(tmp)
        X =  np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = X * self.features_scaler


        # print("nrows 1:", c, nrows, ix)
        # print(X)


        # Second round of subsetting
        # if len(self.kmer_frame[row_num]) >= 20:
        #     nrows = np.random.randint(low=1, high=len(self.kmer_frame[row_num]) / 10)
        # else:
        #     nrows = np.random.randint(low=1, high=len(self.kmer_frame[row_num]))


        #c = len(self.kmer_frame[row_num])
        nrows = int(np.floor(np.random.exponential(c / 5)) + 1)
        if nrows > c:
            nrows = np.random.randint(low=1, high=c+1)
        #print("nrows:",nrows,c)
        ix = np.random.randint(low=0, high=len(self.kmer_frame[row_num]) - nrows+1)
        subset = self.kmer_frame[row_num][ix:ix + nrows]


        # Normalize
        tmp = subset.sum(axis=0)
        X2 = tmp / np.sum(tmp)
        X2 = np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0)
        X2 = X2 * self.features_scaler


        # print("nrows 2:", c, nrows, ix)
        # print(X2)

        Z = np.concatenate([X, X2])
        y = row_num

        # X = self.label_idx[ID]
        # y = self.list_IDs[idx]

        #print(Z)
        #print(y)

        return Z, y




class Dataset_chunks_1row(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, df, list_IDs, label_idx, features_scaler):
        'Initialization'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.kmer_frame = df
        self.list_IDs = list_IDs
        self.label_idx = label_idx
        self.features_scaler = features_scaler


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Select sample
        ID = self.list_IDs[idx]
        row_num = self.label_idx[ID]

        #print(row_num)

        # Load data and get label
        #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
        #X = self.kmer_frame.iloc[list(row_num), :].values

        # First round of subsetting
        #if len(self.kmer_frame[row_num]) >= 20:
        #    nrows = np.random.randint(low = 1, high = len(self.kmer_frame[row_num])/10)
        #else:
        #    nrows = np.random.randint(low=1, high=len(self.kmer_frame[row_num]))
        c = len(self.kmer_frame[row_num])
        nrows = int(np.floor(np.random.exponential(c / 5)) + 1)
        if nrows > c:
            nrows = np.random.randint(low=1, high=c+1)
        #print("nrows:", nrows,c)
        ix = np.random.randint(low = 0, high = len(self.kmer_frame[row_num])-nrows+1)
        subset = self.kmer_frame[row_num].iloc[ix:ix+nrows, :]


        #print(nrows,ix)
        # Normalize
        tmp = subset.sum(axis=0, numeric_only=True).to_frame()
        tmp[0] = (tmp[0] / tmp[0].sum()).replace(np.inf, np.nan).fillna(0) * self.features_scaler


        X = tmp[0].values
        y = row_num

        # X = self.label_idx[ID]
        # y = self.list_IDs[idx]

        # print(X)
        # print(y)

        return X, y


class Dataset_chunks(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, df, list_IDs, label_idx, features_scaler):
        'Initialization'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.kmer_frame = df
        self.list_IDs = list_IDs
        self.label_idx = label_idx
        self.features_scaler = features_scaler


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Select sample
        ID = self.list_IDs[idx]
        row_num = self.label_idx[ID]

        #print(row_num)

        # Load data and get label
        #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
        #X = self.kmer_frame.iloc[list(row_num), :].values

        nrows = np.random.randint(low = 1, high = len(self.kmer_frame[row_num]))
        ix = np.random.randint(low = 0, high = len(self.kmer_frame[row_num])-nrows)
        subset = self.kmer_frame[row_num].iloc[ix:ix+nrows, :]


        # Normalize
        tmp = subset.sum(axis=0, numeric_only=True).to_frame()
        tmp[0] = (tmp[0] / tmp[0].sum()) * self.features_scaler


        X = tmp[0].values
        y = row_num

        # X = self.label_idx[ID]
        # y = self.list_IDs[idx]

        # print(X)
        # print(y)

        return X, y



class Dataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, df, list_IDs, label_idx):
        'Initialization'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.kmer_frame = df
        self.list_IDs = list_IDs
        self.label_idx = label_idx


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Select sample
        ID = self.list_IDs[idx]
        row_num = self.label_idx[ID]

        #print(row_num)

        # Load data and get label
        #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
        #X = self.kmer_frame.iloc[list(row_num), :].values


        X = self.kmer_frame.iloc[row_num, :].values
        y = row_num

        # X = self.label_idx[ID]
        # y = self.list_IDs[idx]

        # print(X)
        # print(y)

        return X, y


class Dataset_numpy(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(self, df, list_IDs, label_idx):
        'Initialization'
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.kmer_frame = df
        self.list_IDs = list_IDs
        self.label_idx = label_idx


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # Select sample
        ID = self.list_IDs[idx]
        row_num = self.label_idx[ID]

        #print(row_num)

        # Load data and get label
        #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
        #X = self.kmer_frame.iloc[list(row_num), :].values


        X = self.kmer_frame[row_num]
        y = row_num

        # X = self.label_idx[ID]
        # y = self.list_IDs[idx]

        # print(X)
        # print(y)

        return X, y


# class Dataset(torch.utils.data.Dataset):
#     """
#     Characterizes a dataset for PyTorch
#     """
#     def __init__(self, df, list_IDs, label_idx):
#         'Initialization'
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#         """
#         self.kmer_frame = df
#         self.list_IDs = list_IDs
#         self.label_idx = label_idx
#
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)
#
#     def __getitem__(self, idx):
#         'Generates one sample of data'
#         # if torch.is_tensor(idx):
#         #     idx = idx.tolist()
#
#         # Select sample
#         ID = self.list_IDs[idx]
#         row_num = self.label_idx[ID]
#
#         # Load data and get label
#         #X = torch.tensor(self.kmer_frame.iloc[row_num, :].values, dtype=torch.float64)
#         X = self.kmer_frame.iloc[row_num, :].values
#         y = row_num
#
#         # print(X)
#         # print(y)
#
#         return X, y


# class Lambda_Dataset(torch.utils.data.Dataset):
#   """
#   This is a custom dataset class.
#   """
#   def __init__(self, X):
#     self.X = X
# #    self.Y = Y
# #     if len(self.X) != len(self.Y):
# #       raise Exception("The length of X does not match the length of Y")
#
#   def __len__(self):
#     return len(self.X)
#
#   def __getitem__(self, index):
#     # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
#     _x = self.X[index]
#     _y = index
#
#     return _x, _y
#

