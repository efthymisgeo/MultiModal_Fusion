import torch

import numpy as np
from random import randint

import matplotlib.pyplot as plt

#############################
## DEVICE: CPU / GPU
# use GPU if available
##############################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MAX_LEN of GloVe/ averaged COVAREP
MAX_LEN = 50

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