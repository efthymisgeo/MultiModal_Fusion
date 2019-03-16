from torch import torch, nn
from torch.optim import Adam

from config import DEVICE, TEXT_RNN_PATH, AUDIO_RNN_PATH

from models.attention_attention import Hierarchy_Attn

from utils.model_train import train_attention_model, eval_attention_model

from sklearn.metrics import accuracy_score


##############################################################
#           we train our binary calssifier                  ##
##############################################################

def attention_model_training(data_loaders, text_params, audio_params,
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
    attn_model = Hierarchy_Attn(text_params, audio_params).to(DEVICE)
    print(attn_model)

    parameters = attn_model.parameters()
    # get only trainable parameters
    # parameters = [params for params in atnn_model.parameters() if
    #              params.requires_grad==True]
    optimizer = Adam(parameters, lr=learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    valid_losses = []

    batch_accuracies = []

    print('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
    print('               Training Attention Fusion Model         ')
    print('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')

    #########################################
    # Training Pipeline
    #########################################
    train_loader, valid_loader, test_loader = data_loaders

    for epoch in range(1, EPOCHS + 1):
        # train model
        train_attention_model(epoch, clip, train_loader, attn_model, criterion,
                              optimizer)

        # evaluate performance on  test set
        train_loss, (y_train_pred, y_train_gold) = \
            eval_attention_model(train_loader, attn_model, criterion)
        # evaluate performanve on valid set

        valid_loss, (y_valid_pred, y_valid_gold) = \
            eval_attention_model(valid_loader, attn_model, criterion)
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
        eval_attention_model(test_loader, attn_model, criterion)

    print("Test Set Accuracy is ", accuracy_score(y_test_gold,
                                                  y_test_pred))

    return (attn_model, batch_accuracies, valid_losses, train_losses)




