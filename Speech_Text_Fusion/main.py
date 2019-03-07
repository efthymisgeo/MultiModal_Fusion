import os
from torch import torch, nn
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import DEVICE

from models.Text_Rnn import Text_RNN

from utils.pytorch_dl import MModalDataset
from utils.model_train import train_text_rnn, eval_text_rnn


##########################################
## Construction of a synthetic dataset  ##
## glove = (*,300)
## covarep should be (*, 74) but is given
# in (1,5001) zero padded format where
# [:,-1] idx is real length value
## opinions = (1,1)
##########################################


def rand_np(a, b, size_tuple):
    return((b-a)
           * np.random.random_sample(size_tuple)
           + a)

def synthetic_dataset():
    ''' synthetic 2 point random dataset generation '''
    size_list = [3, 2, 4, 3, 1]
    real_size = [[7*74, 3*74, 9*74], [12*74, 20*74],
                 [4*74, 1*74, 5*74, 6*74],
                 [2*74, 8*74, 74], [3*74]]
    glove = []
    covarep = [[],[],[],[],[]]
    opinions = []
    for sen_id, sentense in enumerate(size_list):
        glove.append(rand_np(-2,2, (sentense,5032+1)))
        opinions.append(rand_np(-3, 3, (1, 1)))
        for utt_id, utterance in enumerate(real_size[sen_id][:]):
            covarep[sen_id].append(rand_np(-5,5, (1,5032+1)))
            covarep[sen_id][utt_id][:,-1] = real_size[sen_id][utt_id]
    dataset = {"Audio Features": covarep,
               "Word Embeddings": glove,
               "Opinion Labels": opinions}
    return(dataset)

###############################################
# Load Task and synthetic dataset
###############################################
task = "Binary"
approach = 'sequential'
toy_data = synthetic_dataset()
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
train_loader = DataLoader(mm_train, batch_size=2, shuffle=True)
valid_loader = DataLoader(mm_valid)
test_loader = DataLoader(mm_test)

print("----------------")
print("ready to rock")


#######################################
#### text rnn hyperparams       ######
######################################
text_hyperparameters = []
input_size = 300 # glove size
hidden_size = 50 # hidden state size
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

#########################################
# Model Definition
########################################
model = Text_RNN(*text_hyperparameters)
model.to(DEVICE)
print(model)
optimizer = Adam(model.parameters(),
                 lr = 0.001)

# choose loss function
if task == "5 Class":
    criterion = nn.CrossEntropyLoss()
elif task == "Binary":
    criterion = nn.BCEWithLogitsLoss()
else:
    raise ValueError("no such task")
#########################################
# Training Pipeline
#########################################
EPOCHS = 3

for epoch in range(1, EPOCHS+1):
    # train model
    train_text_rnn(epoch, train_loader, model, criterion, optimizer)
    # evaluate performanve on valid set
    train_loss, (y_train_gold, y_train_pred) = eval_text_rnn(train_loader,
                                                             model,criterion)

    # evaluate performance on test set
    test_loss, (y_test_gold, y_test_pred) = eval_text_rnn(test_loader,
                                                          model,criterion)




for batch_idx, (_, glove, opinions) in enumerate(train_loader):
    embds = glove[0].float()
    lens = glove[1]
    print(embds.size())
    print(lens.size())
    model.zero_grad()
    logits, att_weights = model(embds, lens)
    print(logits)



