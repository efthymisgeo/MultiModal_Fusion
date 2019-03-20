import os
import torch
import pickle
import numpy as np
from random import randint

import matplotlib.pyplot as plt

#############################
## DEVICE: CPU / GPU
# use GPU if available
##############################
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# MAX_LEN of GloVe/ averaged COVAREP
MAX_LEN = 50

BATCH_SIZE = 8


# save model dictionary to PATH
rnn_path = os.path.abspath("rnn_metadata")
TEXT_RNN_PATH = os.path.join(rnn_path, "text_rnn_model.py")
AUDIO_RNN_PATH = os.path.join(rnn_path, "audio_rnn_model.py")

#############################
###TOY DATASET GENERATION ###
#############################
def rand_np(a, b, size_tuple):
    return ((b - a)
            * np.random.random_sample(size_tuple)
            + a)

def synthetic_dataset(N):
    # list of each sentence length
    size_list = []

    covarep = []
    glove = []
    opinions = []

    for i in range(N):
        size = randint(1,10)
        size_list.append(size)

        _covarep = rand_np(-2,2,(size, 5032))
        _glove = rand_np(-5,5, (size, 300))
        _opinions = rand_np(-1,1, (1,1))

        zeros = np.zeros((size, 5033-300))
        _glove = np.concatenate((_glove, zeros), axis=1)

        length = [[randint(1,100)*74] for _ in range(size)]
        length = np.array(length)
        _covarep = np.concatenate((_covarep, length), axis=1)

        covarep.append(_covarep)
        glove.append(_glove)
        opinions.append(_opinions)

    dataset = {"Audio Features": covarep,
               "Word Embeddings": glove,
               "Opinion Labels": opinions}

    return (dataset)
##############################

#################################
## learning curves plot
#################################
def learn_curves(valid, train, name):
    fig = plt.figure()
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(valid, 'r', label='validation curve')
    plt.plot(train, 'g', label='training curve')
    plt.legend()
    plt.show()
    fig.savefig(name + '.png')


#####################################
# save dictionaries
#####################################

def pickle_save(fname, data):
    '''function that saves metadata dictionary'''
    filename = os.path.join('rnn_metadata', fname)
    save_dir = os.path.abspath(filename)
    filehandler = open(save_dir, "wb")
    pickle.dump(data, filehandler)
    filehandler.close()
    print('saved', fname, os.getcwd(), os.listdir())

def pickle_load(path, fname):
    filepath = os.path.join(path, fname)
    filehandler = open(filepath, 'rb')
    data = pickle.load(filehandler)
    filehandler.close()
    print('Loaded Succesfully ', fname)
    return(data)
