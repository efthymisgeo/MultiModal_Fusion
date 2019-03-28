from torch import torch, nn
from torch.optim import Adam

from config import DEVICE

from models.Text_Rnn import Text_Encoder

from utils.model_train import train_audio_rnn, eval_audio_rnn

from sklearn.metrics import accuracy_score, f1_score

######################################
#### audio rnn hyperparams       ######
######################################
# audio_hyperparameters = []
# 0 - input_size = 74  # covarep size
# 1 - hidden_size = 38 # hidden state size
# 2 - num_layers = 1 # how many stacked rnn's
# 3 - bidirectional = True
# 4 - dropout = 0.0
# 5 - architecture = 'GRU'
# 6 - attention_size = hidden_size
# 7 - batch_first = True
# 8 - attn_layers = 1
# 9 - attn_dropout = 0.05
# 10 - attn_nonlinearity = 'tanh'
########################################


########################################
# we train our audio classifier and get
# the final weights after that
########################################

def audio_rnn_pretraining(data_loaders, rnn_params, EPOCHS,
                          learning_rate=0.001, clip =5.0):
    '''
    INPUTS:
    data_loaders : 3-len tuple that contains
        train_loader: PyTorch iterator
        valid_loader: same as above
        test_loader: same as above
    rnn_params: list of parameters
    EPOCHS (int)
    learning_rate

    OUTPUTS:
        model : trained model
        batch_accuracies: accuracy at every batch

    FUNCTION:
        trains model for given number of EPOCHS
    '''
    # model = audio_rnn
    audio_rnn = Text_Encoder(*rnn_params)

    audio_rnn.to(DEVICE)
    print(audio_rnn)

    optimizer = Adam(audio_rnn.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    valid_losses = []

    batch_accuracies = []

    print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
    print('         Training Audio RNN              ')
    print('-----------------------------------------')

    #########################################
    # Training Pipeline
    #########################################
    train_loader, valid_loader, test_loader = data_loaders

    for epoch in range(1, EPOCHS + 1):
        # train model
        train_audio_rnn(epoch, clip, train_loader, audio_rnn, criterion, optimizer)

        # evaluate performance on  test
        train_loss, (y_train_pred, y_train_gold) = eval_audio_rnn(train_loader,
                                                                  audio_rnn,
                                                                  criterion)
        # evaluate performanve on valid set
        valid_loss, (y_valid_pred, y_valid_gold) = eval_audio_rnn(valid_loader,
                                                                  audio_rnn,
                                                                  criterion)

        print("Valid Loss is: ", valid_loss)

        batch_accuracy = accuracy_score(y_train_gold, y_train_pred)
        print('Train accuracy at epoch ', epoch,
              'is ', batch_accuracy)

        valid_accuracy = accuracy_score(y_valid_gold, y_valid_pred)
        print('Valid accuracy at epoch ', epoch,
              'is ', valid_accuracy)

        valid_losses.append(valid_loss)
        train_losses.append(train_loss)

        batch_accuracies.append(batch_accuracy)

        # evaluate performance on test set
        test_loss, (y_test_pred, y_test_gold) = eval_audio_rnn(test_loader,
                                                               audio_rnn,
                                                               criterion)


        print("Test Set Accuracy is ", accuracy_score(y_test_gold,
                                                      y_test_pred),
              " F1 Score is: ", f1_score(y_test_gold, y_test_pred))

    return (audio_rnn, batch_accuracies, valid_losses, train_losses)


