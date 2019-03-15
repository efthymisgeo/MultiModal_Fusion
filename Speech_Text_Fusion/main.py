import os
from torch import torch, nn
from torch.optim import Adam
import numpy as np
from random import randint

from models.Text_Rnn import Text_RNN

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import DEVICE, MAX_LEN, BATCH_SIZE, synthetic_dataset,\
    learn_curves, pickle_save, pickle_load

from utils.pytorch_dl import MModalDataset
from utils.torch_dataloader import MultiModalDataset
from utils.model_dataloader import MOSI_Dataset, MOSI_Binary_Dataset

from experiments.binary_classification import binary_model_training
from experiments.pretraining.audio_rnn import audio_rnn_pretraining
from experiments.pretraining.text_rnn import text_rnn_pretraining

print(DEVICE)


###############################################
# Load Task and synthetic dataset
###############################################
N = 200 # instances of synthetic dataset
task = "Binary"
approach = 'sequential'
dataset = synthetic_dataset(N)
###############################################
# PyTorch Dataloader
###############################################

# load MOSI
#dataset = MOSI_Binary_Dataset()

# load mosi
mm_dset = MultiModalDataset(dataset, task, approach)

# get dataset length
l = len(mm_dset)

# split dataset to train-valid-test
test_size = int(0.2*l)
train_size = l - 2*test_size
mm_train, mm_valid, mm_test = random_split(mm_dset,
                                           [train_size, test_size, test_size])
# use dataloaders wrappers
train_loader = DataLoader(mm_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(mm_valid)
test_loader = DataLoader(mm_test)


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

clip = 200.0

audio_hyperparameters = [input_size, hidden_size,
                        num_layers, bidirectional,
                        dropout, architecture,
                        attention_size, batch_first,
                        attn_layers, attn_dropout,
                        attn_nonlinearity, task]
'''
#########################################
# Training Text RNN Models
########################################

EPOCHS_t = 60
lr_t = 0.00001
data_loaders = (train_loader, valid_loader, test_loader)

text_rnn, text_accuracies, valid_losses, train_losses\
    = text_rnn_pretraining(data_loaders,
                           text_hyperparameters,
                           EPOCHS_t, lr_t)

# Printing Learning Curves
learn_curves(valid_losses, train_losses, "TextRNN_Loss")

# save model metadata
text_rnn_metadata = {"accuracy": text_accuracies,
                     "valid_loss": valid_losses,
                     "train_loss": train_losses}

# save metadata dict
pickle_save("text_rnn.p", text_rnn_metadata)


###########################################
# Training Audio RNN Model
###########################################

EPOCHS_a = 130
lr_a = 0.00001
data_loaders = (train_loader, valid_loader, test_loader)

audio_rnn, audio_accuracies, valid_losses, train_losses\
    = audio_rnn_pretraining(data_loaders,
                            audio_hyperparameters,
                            EPOCHS_a, lr_a, clip)
# Printing Learning Curves
learn_curves(valid_losses, train_losses, "AudioRNN_Loss")

#save model metadata
audio_rnn_metadata = {"accuracy": audio_accuracies,
                      "valid_loss": valid_losses,
                      "train_loss": train_losses}

# save metadata dictionaries
pickle_save("audio_rnn.p", audio_rnn_metadata)

##########################################
# Save/Load models
##########################################

# SAVING MODE
# save model dictionary to PATH
rnn_path = os.path.abspath("rnn_metadata")
TEXT_RNN_PATH = os.path.join(rnn_path, "text_rnn_model.py")
AUDIO_RNN_PATH = os.path.join(rnn_path, "audio_rnn_model.py")

# always tranfer to cpu for interuser compatibility
model = text_rnn.to("cpu")
torch.save(model.state_dict(), TEXT_RNN_PATH)

model = audio_rnn.to("cpu")
torch.save(model.state_dict(), AUDIO_RNN_PATH)
'''


###################################################################
###                     BINARY TASK
###################################################################
EPOCHS_bin = 100
lr_bin = 0.0001
#clip_bin_grad = 500

data_loaders = (train_loader, valid_loader, test_loader)

binary_model, binary_accuracies, bin_valid_losses, bin_train_losses \
    = binary_model_training(data_loaders,
                            text_hyperparameters,
                            audio_hyperparameters,
                            EPOCHS_bin, lr_bin, clip)
# Printing Learning Curves
learn_curves(bin_valid_losses, bin_train_losses, "Binary_Loss")


print("kapakipoooooooooooooo")

