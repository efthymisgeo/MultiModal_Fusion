""" This script allows to run the following to datasets CMU-MOSI, CMU-MOSEI: 
    - pretrain the Text classifier
    - pretrain the Audio classifier
    - train the Fusion model

    Commands to run the above respectively:
        CMU-MOSI:
            python3 main.py -mosi -pre_text
            python3 main.py -mosi -pre_audio
            python3 main.py -mosi

        CMU-MOSEI:
            python3 main.py -mosei -pre_text
            python3 main.py -mosei -pre_audio
            python3 main.py -mosei
"""
import os
from torch import torch, nn
from torch.optim import Adam
import numpy as np
from random import randint
import sys

from models.Text_Rnn import Text_RNN

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import DEVICE, MAX_LEN, BATCH_SIZE, synthetic_dataset,\
    learn_curves, pickle_save, pickle_load

from utils.pytorch_dl import MModalDataset
from utils.torch_dataloader import MultiModalDataset
from utils.model_dataloader import MOSI_Dataset, MOSI_Binary_Dataset
from utils.m_mosei_dataloader import MOSEI_Dataset, MOSEI_Binary_Dataset

from experiments.binary_attention import attention_model_training
from experiments.binary_classification import binary_model_training
from experiments.pretraining.audio_rnn import audio_rnn_pretraining
from experiments.pretraining.text_rnn import text_rnn_pretraining

print(DEVICE)

###############################################
# Load Task and synthetic dataset
###############################################
task = "Binary"
approach = 'sequential'
# N = 200 # instances of synthetic dataset
# dataset = synthetic_dataset(N)
###############################################
# PyTorch Dataloader
###############################################

if (len(sys.argv) > 1 and sys.argv[1] == '-mosei'):
    print(' ===== MOSEI dataset ===== ')
    dataset = MOSEI_Binary_Dataset()
    dataset_flag = 'mosei'
elif (len(sys.argv) > 1 and sys.argv[1] == '-mosi'):
    print(' ===== MOSI dataset ===== ')
    dataset = MOSI_Binary_Dataset()
    dataset_flag = 'mosi'
else:
    print('Specify dataset, e.g. using the command: "python3 main.py -mosi" to train the fusion model to MOSI dataset')
    print('Operation ended.')
    sys.exit()

# load dataset
mm_dset = MultiModalDataset(dataset, task, approach)

# get dataset length
l = len(mm_dset)

# split dataset to train-valid-test
test_size = int(0.2*l)
train_size = l - 2*test_size

# reproducability
if DEVICE == "cuda:1":
    torch.backends.cudnn.deterministic = True

torch.manual_seed(64)
mm_train, mm_valid, mm_test = random_split(mm_dset,
                                           [train_size, test_size, test_size])
# use dataloaders wrappers
train_loader = DataLoader(mm_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(mm_valid)
test_loader = DataLoader(mm_test)

############################################
#### text rnn hyperparams and pretraing ####
############################################
print(' ----- Pretrain Text classifier ----- ')
text_hyperparameters = []
input_size = 300 # glove size
hidden_size = 64*2 # hidden state size
num_layers = 1 # how many stacked rnn's
bidirectional = True
dropout = 0.25
architecture = 'LSTM'
attention_size = hidden_size
batch_first = True
attn_layers = 1
attn_dropout = 0.25
attn_nonlinearity = 'tanh'

text_hyperparameters = [input_size, hidden_size,
                        num_layers, bidirectional,
                        dropout, architecture,
                        attention_size, batch_first,
                        attn_layers, attn_dropout,
                        attn_nonlinearity, task]

#############################################
#### audio rnn hyperparams and pretraing ####
#############################################
print(' ----- Pretrain Audio classifier ----- ')
audio_hyperparameters = []
input_size = 74 # covarep size
hidden_size = 16*2 # hidden state size
num_layers = 1 # how many stacked rnn's
bidirectional = True
dropout = 0.25
architecture = 'LSTM'
attention_size = hidden_size
batch_first = True
attn_layers = 1
attn_dropout = 0.25
attn_nonlinearity = 'tanh'

clip = 200.0

audio_hyperparameters = [input_size, hidden_size,
                        num_layers, bidirectional,
                        dropout, architecture,
                        attention_size, batch_first,
                        attn_layers, attn_dropout,
                        attn_nonlinearity, task]


if (len(sys.argv) > 2 and sys.argv[2] == '-pre_text'):
   
    ################################################################
    # Training Text RNN Models
    ################################################################
    #drop = [0.0, 0.1, 0.5]
    #EPOCHS_ = [120, 120, 120]
    #lr_t = 0.00001

    print("###############################################")
    data_loaders = (train_loader, valid_loader, test_loader)
    EPOCHS_t = 110
    lr_t = 0.00001
    text_rnn, text_accuracies, valid_losses, train_losses = \
        text_rnn_pretraining(data_loaders, text_hyperparameters, EPOCHS_t, lr_t)

    # Saving Learning Curves
    learn_curves(valid_losses, train_losses, "TextRNN_Loss")

    # save model metadata
    text_rnn_metadata = {"accuracy": text_accuracies,
                        "valid_loss": valid_losses,
                        "train_loss": train_losses}

    # save metadata dict
    pickle_save(dataset_flag + "_text_rnn.p", text_rnn_metadata)
    ####################################################################
    # Save model
    ####################################################################

    # SAVING MODE
    # save model dictionary to PATH
    rnn_path = os.path.abspath("pretrained_models")
    TEXT_RNN_PATH = os.path.join(rnn_path, dataset_flag + "_text_rnn_model.pt")

    # always tranfer to cpu for interuser compatibility
    model = text_rnn.to("cpu")
    torch.save(model.state_dict(), TEXT_RNN_PATH)
elif (len(sys.argv) > 2 and sys.argv[2] == '-pre_audio'):
 
    ####################################################################
    # Training Audio RNN Model
    ####################################################################
    EPOCHS_a = 150
    lr_a = 0.0001
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
    pickle_save(dataset_flag + "audio_rnn.p", audio_rnn_metadata)
    
    ####################################################################
    # Save model
    ####################################################################
    # SAVING MODE
    # save model dictionary to PATH
    rnn_path = os.path.abspath("pretrained_models")
    AUDIO_RNN_PATH = os.path.join(rnn_path, dataset_flag + "_audio_rnn_model.pt")

    model = audio_rnn.to("cpu")
    torch.save(model.state_dict(), AUDIO_RNN_PATH)
else:
    ##########################################################
    #### fusion rnn hyperparams training and evaluation ######
    ##########################################################
    print(' ----- Train and Evaluate Fusion model ----- ')
    fusion_hyperparameters = []
    input_size =  2*(2*128+32)  # stacked fused size
    hidden_size = 2*128 #*2 # hidden fused state size
    num_layers = 1 # how many stacked rnn's
    bidirectional = True
    dropout = 0.25
    architecture = 'LSTM'
    attention_size = hidden_size
    batch_first = True
    attn_layers = 1
    attn_dropout = 0.25
    attn_nonlinearity = 'tanh'

    fusion_hyperparameters = [input_size, hidden_size,
                            num_layers, bidirectional,
                            dropout, architecture,
                            attention_size, batch_first,
                            attn_layers, attn_dropout,
                            attn_nonlinearity, task]
    
    # TODO change model names to prepend 'mosi_' or 'mosei_' 
    rnn_path = os.path.abspath("pretrained_models")
    TEXT_RNN_PATH = os.path.join(rnn_path, "text_rnn_model.pt")
    AUDIO_RNN_PATH = os.path.join(rnn_path, "audio_rnn_model.pt")

    model_paths = {"audio":AUDIO_RNN_PATH,
               "text":TEXT_RNN_PATH}

    counter = 0
    # golden tuple: (1e-3, 10, 0.15, 1e-5)5
    training_tuple = [(5e-4, 10, 0.35, 1e-5),
                    (1e-3, 10, 0.15, 1e-5),
                    (1e-3, 10, 0.15, 1e-5)]
    clip = 5
    for lr_bin, EPOCHS_bin, p_drop, L2_reg in training_tuple:
        # fusion, audio, text
        loss_weights = [1., 1., 1.]
        torch.manual_seed(64)

        data_loaders = (train_loader, valid_loader, test_loader)

        binary_model, binary_accuracies, bin_valid_losses, bin_train_losses \
            = attention_model_training(data_loaders,
                                    text_hyperparameters,
                                    audio_hyperparameters,
                                    fusion_hyperparameters,
                                    EPOCHS_bin,
                                    loss_weights,
                                    model_paths,
                                    lr_bin, clip,
                                    p_drop, L2_reg)
        # Printing Learning Curves
        counter +=1
        learn_curves(bin_valid_losses, bin_train_losses,
                    "Attention_Loss"+str(counter))

print("Operation completed.")

'''

####################################################################
## timestep attention
####################################################################

###################################################################
###                     BINARY TASK
###################################################################
#EPOCHS_bin = 80
#lr_bin = 0.00001
# 10
#p_drop = 0.15
#L2_reg = 0.0



                  #(5e-4, 10, 0.25, 1e-5),
                  #(1e-3, 10, 0.15, 1e-5), (1e-3, 10, 0.35, 1e-5)]

                     (1e-3, 10, 0.15, 0.0),
                  (1e-3, 10, 0.25, 0.0), (1e-3, 10, 0.25, 0.0),
                  (1e-3, 10, 0.15, 1e-5), (1e-3, 10, 0.15, 1e-5),
                  (1e-3, 10, 0.25, 1e-5), (1e-3, 10, 0.25, 1e-5)]

                  (1e-3, 10, 0.25, 1e-5), (1e-3, 10, ),
                  (1e-3, 10), (1e-3, 10),(1e-3, 10), (1e-3, 10),
                  (1e-3, 10), (1e-3, 10),(1e-3, 10), (1e-3, 10),
                  (5e-4, 14), (5e-4, 14), (5e-4, 14), (5e-4, 14),
                  (5e-4, 14), (5e-4, 14), (5e-4, 14), (5e-4, 14),
                  (5e-4, 14), (5e-4, 14), (5e-4, 14), (5e-4, 14),
                  (5e-4, 14), (5e-4, 14), (5e-4, 14), (5e-4, 14)]

torch.manual_seed(99)
clip=5
data_loaders = (train_loader, valid_loader, test_loader)

EPOCHS_t = 3
lr_t = 1e-3
text_rnn, text_accuracies, valid_losses, train_losses = \
    text_rnn_pretraining(data_loaders, text_hyperparameters, EPOCHS_t, lr_t)

# always tranfer to cpu for interuser compatibility
model = text_rnn.to("cpu")
torch.save(model.state_dict(), TEXT_RNN_PATH)


EPOCHS_a = 145
lr_a = 5e-4
data_loaders = (train_loader, valid_loader, test_loader)
audio_rnn, audio_accuracies, valid_losses, train_losses\
    = audio_rnn_pretraining(data_loaders,
                            audio_hyperparameters,
                            EPOCHS_a, lr_a, clip)

model = audio_rnn.to("cpu")
torch.save(model.state_dict(), AUDIO_RNN_PATH)


'''
