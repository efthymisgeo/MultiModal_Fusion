import torch

import numpy as np
from random import randint

import matplotlib.pyplot as plt

#############################
## DEVICE: CPU / GPU
# use GPU if available
##############################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    real_size = []
    covarep = []
    for i in range(N):
        size = randint(1,5)
        size_list.append(size)

        empty_list = []
        covarep.append(empty_list)

        acoustic_list = []
        for k in range(size):
            acoustic_size = 74*randint(1,6)
            acoustic_list.append(acoustic_size)

        real_size.append(acoustic_list)

    glove = []
    opinions = []
    for sen_id, sentense in enumerate(size_list):
        glove.append(rand_np(-2, 2, (sentense, 5032 + 1)))
        opinions.append(rand_np(-3, 3, (1, 1)))
        for utt_id, utterance in enumerate(real_size[sen_id][:]):
            covarep[sen_id].append(rand_np(-5, 5, (1, 5033)))
            covarep[sen_id][utt_id][:,-1] = utterance

    dataset = {"Audio Features": covarep,
               "Word Embeddings": glove,
               "Opinion Labels": opinions}

    return (dataset)
##############################




#################################
## learning curves plot
#################################


def learn_curves(test, valid):
    plt.figure()
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(test, 'r', label='test curve')
    plt.plot(valid, 'g', label='learning curve')
    plt.legend()
    plt.show()

###############################