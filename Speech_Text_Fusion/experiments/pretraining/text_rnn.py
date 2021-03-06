from torch import torch, nn
from torch.optim import Adam

from config import DEVICE, learn_curves

from models.Text_Rnn import Text_Encoder

from utils.model_train import train_text_rnn, eval_text_rnn

from sklearn.metrics import accuracy_score, f1_score

######################################
#### text rnn hyperparams       ######
######################################
# text_hyperparameters = []
# 0 - input_size = 300 # glove size
# 1 - hidden_size = 50 # hidden state size
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
# we train our text classifier and get
# the final weights after that
########################################

def text_rnn_pretraining(data_loaders, rnn_params, EPOCHS,
                         learning_rate=0.001):
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

    model = Text_Encoder(*rnn_params)
    model.to(DEVICE)
    print(model)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    valid_losses = []

    batch_accuracies = []

    print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
    print('         Training Text RNN               ')
    print('-----------------------------------------')

    #########################################
    # Training Pipeline
    #########################################
    train_loader, valid_loader, test_loader = data_loaders

    for epoch in range(1, EPOCHS + 1):
        # train model
        train_text_rnn(epoch, train_loader, model, criterion, optimizer)

        # evaluate performance on  test
        train_loss, (y_train_pred, y_train_gold) = eval_text_rnn(train_loader,
                                                                 model, criterion)
        # evaluate performanve on valid set
        valid_loss, (y_valid_pred, y_valid_gold) = eval_text_rnn(valid_loader,
                                                                 model, criterion)
        batch_accuracy = accuracy_score(y_train_gold, y_train_pred)

        print("Valid Loss is: ", valid_loss)

        print('Train Accuracy at epoch ', epoch,
              'is ', batch_accuracy)
        valid_accuracy = accuracy_score(y_valid_gold, y_valid_pred)
        print('Valid Accuracy at epoch ', epoch, 'is ', valid_accuracy)

        valid_losses.append(valid_loss)
        train_losses.append(train_loss)

        batch_accuracies.append(batch_accuracy)

        # evaluate performance on test set
        test_loss, (y_test_pred, y_test_gold) = eval_text_rnn(test_loader,
                                                              model, criterion)

        print("Test Set Accuracy is ", accuracy_score(y_test_gold,
                                                      y_test_pred),
              " F1 Score is: ", f1_score(y_test_gold, y_test_pred))

    return (model, batch_accuracies, valid_losses, train_losses)

