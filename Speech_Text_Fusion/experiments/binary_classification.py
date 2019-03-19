from torch import torch, nn
from torch.optim import Adam

from config import DEVICE, TEXT_RNN_PATH, AUDIO_RNN_PATH

from models.binary import Bin_Model

from utils.model_train import train_2_class_model, eval_2_class_model

from sklearn.metrics import accuracy_score


##############################################################
#           we train our binary calssifier                  ##
##############################################################

def binary_model_training(data_loaders, text_params, audio_params,
                          EPOCHS, learning_rate=0.001, clip =50.0):
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
    bin_model = Bin_Model(text_params, audio_params,
                          TEXT_RNN_PATH, AUDIO_RNN_PATH).to(DEVICE)
    print(bin_model)

    # get only trainable parameters
    parameters = [params for params in bin_model.parameters() if
                  params.requires_grad==True]
    optimizer = Adam(parameters, lr=learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    valid_losses = []

    batch_accuracies = []

    print('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
    print('               Training 2 Class Fusion Model         ')
    print('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')

    #########################################
    # Training Pipeline
    #########################################
    train_loader, valid_loader, test_loader = data_loaders

    for epoch in range(1, EPOCHS + 1):
        # train model
        train_2_class_model(epoch, clip, train_loader, bin_model, criterion,
                            optimizer)

        # evaluate performance on  test set
        train_loss, (y_train_pred, y_train_gold) = \
            eval_2_class_model(train_loader, bin_model, criterion)
        # evaluate performanve on valid set

        valid_loss, (y_valid_pred, y_valid_gold) = \
            eval_2_class_model(valid_loader, bin_model, criterion)
        print("Validation Loss at epoch", epoch, " is ", valid_loss)
        # get batch accuracies
        batch_accuracy = accuracy_score(y_train_gold, y_train_pred)
        print('Train accuracy at epoch ', epoch,
              'is ', batch_accuracy)
        valid_accuracy = accuracy_score(y_valid_gold, y_valid_pred)
        print('Valid accuracy at epoch ', epoch,
              'is ', valid_accuracy)
        # store train/valid losses
        valid_losses.append(valid_loss)
        train_losses.append(train_loss)
        # store accuracies
        batch_accuracies.append(batch_accuracy)

    # evaluate performance on test set
    test_loss, (y_test_pred, y_test_gold) = \
        eval_2_class_model(test_loader, bin_model, criterion)

    print("Test Set Accuracy is ", accuracy_score(y_test_gold,
                                                  y_test_pred))

    return (bin_model, batch_accuracies, valid_losses, train_losses)











