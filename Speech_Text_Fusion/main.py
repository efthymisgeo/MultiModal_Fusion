import os
from torch import torch, nn
from torch.optim import Adam
import numpy as np
from random import randint

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import DEVICE, synthetic_dataset, learn_curves, pickle_save, pickle_load

from utils.pytorch_dl import MModalDataset
from utils.torch_dataloader import MultiModalDataset
from utils.model_dataloader import MOSI_Dataset, MOSI_Binary_Dataset

from experiments.pretraining.audio_rnn import audio_rnn_pretraining
from experiments.pretraining.text_rnn import text_rnn_pretraining

print(DEVICE)

###############################################
# Load Task and synthetic dataset
###############################################
N = 100 # instances of synthetic dataset
task = "Binary"
approach = 'sequential'
dataset = synthetic_dataset(N)
###############################################
# PyTorch Dataloader
###############################################

# load MOSI
# dataset = MOSI_Binary_Dataset()

# load mosi
mm_dset = MultiModalDataset(dataset, task, approach)

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

clip = 50.0

audio_hyperparameters = [input_size, hidden_size,
                        num_layers, bidirectional,
                        dropout, architecture,
                        attention_size, batch_first,
                        attn_layers, attn_dropout,
                        attn_nonlinearity, task]

#########################################
# Training Text RNN Models
########################################
'''
EPOCHS_t = 1
lr_t = 0.0001
data_loaders = (train_loader, valid_loader, test_loader)

text_rnn, text_accuracies, valid_losses, train_losses\
    = text_rnn_pretraining(data_loaders,
                           text_hyperparameters,
                           EPOCHS_t, lr_t)

text_rnn_metadata = {"model": text_rnn,
                 "accuracy": text_accuracies,
                 "valid_loss": valid_losses,
                 "train_loss": train_losses}
# Printing Learning Curves
#learn_curves(valid_losses, train_losses)
'''


# Training Audio RNN Model
EPOCHS_a = 10
lr_a = 0.001
data_loaders = (train_loader, valid_loader, test_loader)

audio_rnn, audio_accuracies, valid_losses, train_losses\
    = audio_rnn_pretraining(data_loaders,
                            audio_hyperparameters,
                            EPOCHS_a, lr_a, clip)
# Printing Learning Curves
# learn_curves(valid_losses, train_losses)

audio_rnn_metadata = {"model": audio_rnn,
                      "accuracy": audio_accuracies,
                      "valid_loss": valid_losses,
                      "train_loss": train_losses}

# save metadata dictionaries
#pickle_save("text_rnn", text_rnn_metadata)
pickle_save("audio_rnn", audio_rnn_metadata)

# load metadata dicts
#rnn_path = os.path.abspath("rnn_metadata")
#text_rnn_data = pickle_load(os.path.join(rnn_path,"text_rnn"))
#audio_rnn_data = pickle_load(os.path.join(rnn_path,"audio_rnn"))





