from torch import torch, nn
from torch.optim import Adam

from config import DEVICE, TEXT_RNN_PATH, AUDIO_RNN_PATH

from models.attention_attention import Hierarchy_Attn

from utils.model_train import train_attention_model, eval_attention_model

from sklearn.metrics import accuracy_score, f1_score


##############################################################
#           we train our binary calssifier                  ##
##############################################################

def attention_model_training(data_loaders, text_params, audio_params,
                             fusion_params, EPOCHS, loss_weights,
                             learning_rate=0.001,
                             clip =5.0, p_drop = 0.15, L2_reg = 0.0):
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
    attn_model = Hierarchy_Attn(text_params, audio_params,
                                fusion_params, p_drop).to(DEVICE)
    print(attn_model)

    parameters = attn_model.parameters()
    # get only trainable parameters
    # parameters = [params for params in atnn_model.parameters() if
    #              params.requires_grad==True]

    optimizer = Adam(parameters, lr=learning_rate,
                     weight_decay= L2_reg)

    optimizer_list = [optimizer]

    loss_fusion = nn.BCEWithLogitsLoss()
    loss_audio = nn.BCEWithLogitsLoss()
    loss_text = nn.BCEWithLogitsLoss()

    criteria_list = [loss_fusion, loss_audio, loss_text]

    train_losses = []
    valid_losses = []
    batch_accuracies = []

    audio_train_losses = []
    audio_valid_losses = []
    audio_batch_accuracies = []

    text_train_losses = []
    text_valid_losses = []
    text_batch_accuracies = []

    print('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
    print('               Training Attention Fusion Model         ')
    print('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')

    #########################################
    # Training Pipeline
    #########################################
    train_loader, valid_loader, test_loader = data_loaders

    for epoch in range(1, EPOCHS + 1):
        # train model
        train_attention_model(epoch, clip, train_loader, attn_model,
                              criteria_list, loss_weights, optimizer_list)

        # evaluate performance on  test set
        train_loss, (y_train_pred, y_train_gold), audio_tr_pred, text_tr_pred = \
            eval_attention_model(train_loader, attn_model, criteria_list, loss_weights)

        # evaluate performanve on valid set
        valid_loss, (y_valid_pred, y_valid_gold), audio_val_pred, text_val_pred = \
            eval_attention_model(valid_loader, attn_model, criteria_list, loss_weights)

        print("Valid Loss at epoch: ", epoch, " is ", valid_loss)

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
        test_loss, (y_test_pred, y_test_gold), audio_test_pred, text_test_pred = \
            eval_attention_model(test_loader, attn_model, criteria_list, loss_weights)

        f1_fused = f1_score(y_test_gold, y_test_pred, average='weighted')

        print("Test Set Accuracy is ", accuracy_score(y_test_gold,
                                                  y_test_pred),
              " F1 score is: ", f1_fused)

        # audio-text modules
        audio_accuracy = accuracy_score(y_test_gold, audio_test_pred)
        text_accuracy = accuracy_score(y_test_gold, text_test_pred)

        f1_audio = f1_score(y_test_gold, audio_test_pred)
        f1_text = f1_score(y_test_gold, text_test_pred)

        # print submodal accuracies
        print("Audio accuracy is: ", audio_accuracy, " F1 score is: ", f1_audio)
        print("Text accuracy is: ", text_accuracy, " F1 score is: ", f1_text)


        loss_weights[0] = loss_weights[0]
        loss_weights[1] = loss_weights[1]*0.5
        loss_weights[2] = loss_weights[2]*0.7

        if loss_weights[1] < 0.3:
            loss_weights[1] = 0.3

        if loss_weights[2] < 0.5:
            loss_weights[2] = 0.5

        print(loss_weights)

    return (attn_model, batch_accuracies, valid_losses, train_losses)




