import os
from torch import torch, nn
from torch.optim import Adam
import numpy as np
from random import randint

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import DEVICE, synthetic_dataset

from utils.pytorch_dl import MModalDataset

from experiments.pretraining.audio_rnn import audio_rnn_pretraining
from experiments.pretraining.text_rnn import text_rnn_pretraining


###############################################
# Load Task and synthetic dataset
###############################################
N = 1000 # instances of synthetic dataset
task = "Binary"
approach = 'sequential'
toy_data = synthetic_dataset(N)
###############################################
# PyTorch Dataloader
###############################################
mm_dset = MModalDataset(toy_data, task, approach)

kke = mm_dset[1]
pasok = len(mm_dset)
l = pasok

# split dataset to train-valid-test
test_size = int(0.2*l)
train_size = l - 2*test_size
mm_train, mm_valid, mm_test = random_split(mm_dset,
                                           [train_size, test_size, test_size])
# use dataloaders wrappers
train_loader = DataLoader(mm_train, batch_size=8, shuffle=True)
valid_loader = DataLoader(mm_valid)
test_loader = DataLoader(mm_test)

print("----------------")
print("ready to rock")


#######################################
#### text rnn hyperparams       ######
######################################
text_hyperparameters = []
input_size = 300 # glove size
hidden_size = 69 # hidden state size
num_layers = 1 # how many stacked rnn's
bidirectional = True
dropout = 0.0
architecture = 'GRU'
attention_size = hidden_size
batch_first = True
attn_layers = 1
attn_dropout = 0.05
attn_nonlinearity = 'tanh'

text_hyperparameters = [input_size, hidden_size,
                        num_layers, bidirectional,
                        dropout, architecture,
                        attention_size, batch_first,
                        attn_layers, attn_dropout,
                        attn_nonlinearity, task]


#######################################
#### audio rnn hyperparams       ######
######################################
audio_hyperparameters = []
input_size = 74 # covarep size
hidden_size = 19 # hidden state size
num_layers = 1 # how many stacked rnn's
bidirectional = True
dropout = 0.0
architecture = 'GRU'
attention_size = hidden_size
batch_first = True
attn_layers = 1
attn_dropout = 0.05
attn_nonlinearity = 'tanh'

audio_hyperparameters = [input_size, hidden_size,
                        num_layers, bidirectional,
                        dropout, architecture,
                        attention_size, batch_first,
                        attn_layers, attn_dropout,
                        attn_nonlinearity, task]

#########################################
# Training Audio/Text RNN Models
########################################
EPOCHS = 50
lr = 0.0001
data_loaders = (train_loader, valid_loader, test_loader)

text_rnn, text_accuracies = text_rnn_pretraining(data_loaders, text_hyperparameters,
                                                 EPOCHS, lr)





