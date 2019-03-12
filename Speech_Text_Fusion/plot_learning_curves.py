import matplotlib.pyplot as plt
import os
from os.path import dirname, abspath

from config import pickle_load


#################################
## learning curves plot
#################################
def learning_curves(valid, train):
    plt.figure()
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(valid, 'r', label='validation curve')
    plt.plot(train, 'g', label='training curve')
    plt.legend()
    plt.show()




# load metadata dicts
rnn_path = os.path.abspath("rnn_metadata")


for file in os.listdir(rnn_path):
    if file == 'text_rnn.p':
        continue
        #text_rnn = pickle_load(rnn_path, file)
    else:
        audio_rnn = pickle_load(rnn_path, file)




# metadata format
# audio_rnn_metadata = {"model": audio_rnn,
#                       "accuracy": audio_accuracies,
#                       "valid_loss": valid_losses,
#                       "train_loss": train_losses}

valid_losses = audio_rnn["valid_loss"]
train_losses = audio_rnn["train_loss"]

learning_curves(valid_losses, train_losses)