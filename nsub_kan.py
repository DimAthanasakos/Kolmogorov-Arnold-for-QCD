'''
DNN based on the nsubjettiness features. This script is loading the already calculated nsubs from the file nsubs.h5 and training a DNN on them.

'''

import kan
print(kan.__file__)
time.sleep(100)

import os
import time
import numpy as np
import math 
import sys
import glob
from collections import defaultdict

import functools

import socket 

import matplotlib.pyplot as plt
import sklearn
import scipy
from scipy.sparse import csr_matrix

# Data analysis and plotting
import pandas as pd
import seaborn as sns
import uproot
import h5py


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


#import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import networkx
import energyflow


# Define the path to your local repository
kan_repo_path = '/global/homes/d/dimathan/Kolmogorov-Arnold-for-QCD'

# Insert the local path at the beginning of sys.path
if kan_repo_path not in sys.path:
    sys.path.insert(0, kan_repo_path)
    from kan.KAN import KAN

# Now you can import the KAN class from your local repository
#from kan.KAN import KAN

#from kan import *

import random

from dataloader import read_file


class nsubKAN():
    
    #---------------------------------------------------------------
    def __init__(self, model_info):
        '''
        :param model_info: Dictionary of model info, containing the following keys:
                                'model_settings': dictionary of model settings
                                'n_total': total number of training+val+test examples
                                'n_train': total number of training examples
                                'n_test': total number of test examples
                                'torch_device': torch device
                                'output_dir':   output directory
                                'body_dim':     n-body phase space dimension
        '''
        
        #print('initializing ParT...')

        self.model_info = model_info
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'torch_device: {self.torch_device}')

        self.output_dir = model_info['output_dir']

        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val'] 
        self.test_frac = self.n_test/self.n_total
        self.val_frac = self.n_val/self.n_total

        self.batch_size = self.model_info['model_settings']['batch_size']
        self.learning_rate = self.model_info['model_settings']['learning_rate']
        self.epochs = self.model_info['model_settings']['epochs']
        self.K = self.model_info['model_settings']['K']
        self.N_cluster = self.model_info['model_settings']['N_cluster']
        self.load_model = self.model_info['model_settings']['load_model']

        self.output = defaultdict(list)

        # only qvs g for now
        if self.N_cluster in [2, 3, 5, 7, 10, 15]:
            path = '/pscratch/sd/d/dimathan/GNN/exclusive_subjets_200k/subjets_unshuffled.h5'
        elif self.N_cluster in [4, 6, 8]:
            path = '/pscratch/sd/d/dimathan/GNN/exclusive_subjets_qvsg_200k_N468/subjets_unshuffled.h5'
        else: 
            path = '/pscratch/sd/d/dimathan/GNN/exclusive_subjets_qvsg_200k_N203040506080100/subjets_unshuffled.h5'

        with h5py.File(path, 'r') as hf:
            self.X_nsub = np.array(hf[f'nsub_subjet_N{self.N_cluster}'])[:self.n_total, :3*(self.K-1) - 1]
            self.Y = hf[f'y'][:self.n_total]

      
        print('loaded from file')
        print()

        print(f'X_nsub.shape: {self.X_nsub.shape}')
        print(f'Y.shape: {self.Y.shape}')

        # print the first 5 rows of X_nsub and Y
        #print(f'X_nsub[:10]: {self.X_nsub[:10]}')
        #print(f'Y[:10]: {self.Y[:10]}')
        # how many of each class in Y: 
        print(f'Y class counts: {np.sum(self.Y == 0)}, {np.sum(self.Y == 1)}')
        self.model = self.init_model()



    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''

        # Define the model 
        model = KAN(width=[self.X_nsub.shape[1], 5, 1], grid = 5, k = 3, device = self.torch_device)
        model = model.to(self.torch_device)

        # Print the model architecture
        print()
        print(model)
        print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
        print()

        if self.load_model:
            path = '/global/homes/d/dimathan/GNNs-and-Jets/Kan_weights/1dim_model.pth'
            print(f"Loading model from {path}")
            model.load_state_dict(torch.load(path))

        return model
    
    #---------------------------------------------------------------
    def shuffle_and_split(self, X, Y, test_ratio=0.1, val_ratio=0.1):
        """
        Shuffles and splits the data into training, validation, and test sets.

        Parameters:
        - X: Features.
        - Y: Targets/Labels.
        - train_ratio: Proportion of the dataset to include in the train split.
        - test_ratio: Proportion of the dataset to include in the test split.
        - val_ratio: Proportion of the dataset to include in the validation split.
        - random_seed: The seed used by the random number generator.

        Returns:
        - X_train, Y_train: Training set.
        - X_val, Y_val: Validation set.
        - X_test, Y_test: Test set.
        """

        # First, split into training and temp (test + validation) sets
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=(test_ratio + val_ratio))

        # Now split the temp set into actual test and validation sets
        test_size_proportion = test_ratio / (test_ratio + val_ratio)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=test_size_proportion)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    #---------------------------------------------------------------
    # Train DNN, using hyperparameter optimization with keras tuner
    #---------------------------------------------------------------
    def train(self):
        print()
        print(f'Training nsub KAN...') 

        # shuffle X_nsub and self.Y and split into training/test/validation sets
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.shuffle_and_split(self.X_nsub, self.Y, test_ratio=self.test_frac, val_ratio=self.val_frac)     

        # split the training data into batches of size 64

        train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float().view(-1, 1))
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True) 
        
        test_data = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float().view(-1, 1))
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        # We now have to conform to the KAN input format, which is a dataset dictionary that contains: 'train_input', 'test_input', 'train_label', 'test_label'
        # Concatenate all data into a single tensor on the specified device

        train_inputs = torch.empty(0, self.X_nsub.shape[1], device=self.torch_device)
        train_labels = torch.empty(0, dtype=torch.long, device=self.torch_device)
        test_inputs = torch.empty(0, self.X_nsub.shape[1], device=self.torch_device)
        test_labels = torch.empty(0, dtype=torch.long, device=self.torch_device)

        for data, labels in train_loader:
            train_inputs = torch.cat((train_inputs, data.to(self.torch_device)), dim=0)
            train_labels = torch.cat((train_labels, labels.to(self.torch_device)), dim=0)

        for data, labels in test_loader:
            test_inputs = torch.cat((test_inputs, data.to(self.torch_device)), dim=0)
            test_labels = torch.cat((test_labels, labels.to(self.torch_device)), dim=0)


        dataset = {}
        dataset['train_input'] = train_inputs
        dataset['test_input'] = test_inputs
        dataset['train_label'] = train_labels.view(-1, 1)
        dataset['test_label'] = test_labels.view(-1, 1)

        print('dataset size:', len(dataset['train_input']) )
        # plot the model: Do a forward pass to generate the graph (no training occurs for this)
        self.model(dataset['train_input'])
        fig = self.model.plot(beta=100, scale=1)
        # Save the figure to a folder 
        folder = './kan_figures/3body_kan'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(f'{folder}/kan_pretraining.png')
        
        
        def train_acc():
            with torch.no_grad():
                predictions = self.model(dataset['train_input'])
                correct = (predictions > 0.).float() == dataset['train_label'].float()
                accuracy = torch.mean(correct.float())
            return accuracy

        def test_acc():
            with torch.no_grad():
                predictions = self.model(dataset['test_input'])
                correct = (predictions > 0.).float() == dataset['test_label'].float()
                accuracy = torch.mean(correct.float())
            return accuracy

        def train_auc_score():
            with torch.no_grad():
                predictions = self.model(dataset['train_input'])
                auc = roc_auc_score(dataset['train_label'].cpu(), predictions.cpu())
            return auc

        def test_auc_score():
            with torch.no_grad():
                predictions = self.model(dataset['test_input'])
                auc = roc_auc_score(dataset['test_label'].cpu(), predictions.cpu())
            return auc

        time_start = time.time()
        criterion = nn.BCEWithLogitsLoss()
        #train = True
        if not self.load_model:
            results = self.model.train(dataset, opt = 'LBFGS', lr = 0.005, lamb=2., lamb_l1=5,  loss_fn = criterion, device = self.torch_device, steps = self.epochs, metrics=(train_acc, test_acc, train_auc_score, test_auc_score), save_fig=True, img_folder='./video', batch = self.batch_size)
            # save it 
            best_model_params = self.model.state_dict()
            path = f'/global/homes/d/dimathan/GNNs-and-Jets/Kan_weights/1dim_model.pth'
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            print(f"Saving model to {path}")
            torch.save(best_model_params, path) 


            print(f'results: {results}')
            print()
            print(f"results['train_loss]: {results['train_loss']}")
            print(f"results['train_auc_score]: {results['train_auc_score']}")
            print(f"results['test_auc_score]: {results['test_auc_score']}")


            time_end = time.time()
            print(f'--------------------------------------------------------------------')
            print()
            print(f"Time to train model for 1 epoch = {(time_end - time_start)/self.epochs:.1f} seconds")
            print()

            print(f"results")
            print(f"train_acc: {results['train_acc'][-1]}")
            print(f"test_acc: {results['test_acc'][-1]}")
            print(f"train_auc: {results['train_auc_score'][-1]}")
            print(f"test_auc: {results['test_auc_score'][-1]}")
            # keep the max test auc
            print(f'====================================================================')
            print(f"max test auc: {max(results['test_auc_score'])}")
            print(f'====================================================================')
            print()

        # calculate the ROC curve and AUC
        with torch.no_grad():
            predictions = self.model(dataset['test_input'])
            fpr, tpr, thresholds = roc_curve(dataset['test_label'].cpu(), predictions.cpu())
            roc_auc = roc_auc_score(dataset['test_label'].cpu(), predictions.cpu())
            print(f'roc_auc: {roc_auc}')
            print(f'accuracy: {test_acc()}')


        fig_final = self.model.plot(scale=1) 
        fig_final.savefig(f'{folder}/post_training.png')


        # now prune the model: 

        pruned_model = self.model.prune(threshold = 0.1)

        print(f'results with pruned model')
        print()
        print(f'Total number of parameters: {sum(p.numel() for p in pruned_model.parameters())}')
        predictions_pruned = pruned_model(dataset['test_input'])
        #auc_pruned = roc_auc_score(dataset['test_label'].cpu(), predictions_pruned.cpu())
        auc_pruned = roc_auc_score(dataset['test_label'].cpu().detach().numpy(), predictions_pruned.cpu().detach().numpy())

        print(f"auc_pruned: {auc_pruned}")
        print(f'test accuracy pruned: {test_acc()}')
        print()

        fig_pruned = pruned_model.plot(scale=1)
        fig_pruned.savefig(f'{folder}/pruned.png')




#---------------------------------------------------------------
if __name__ == '__main__':
    model_info = {
        'output_dir': '/pscratch/sd/d/dimathan/OT/test_output',
        'n_total': 20000,
        'n_train': 16000, 
        'n_test': 2000,
        'n_val': 2000,
        'model_settings': {
            'epochs':2,
            'learning_rate':0.0003,
            'batch_size':512,
            'K': 2,
            'N_cluster': 100,
            'load_model': False
        }
    }
    classifier = nsubKAN(model_info)
    classifier.train()