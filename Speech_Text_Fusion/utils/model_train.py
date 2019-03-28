import math
import sys
import torch
import torch.nn.functional as F
from config import DEVICE

def to_numpy(tensor, task='binary'):
    if task == 'binary':
        pred = tensor.view(-1).detach().cpu().numpy()
    elif task == '5 class':
        # fix me
        pass
    else:
        # regression task
        # fix me
        pass
    return(pred)

def flatten_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return(flat_list)


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()

########################################################################
#                   TEXT RNN TRAINING FUNCTIONS
########################################################################

def train_text_rnn(_epoch, dataloader, model, loss_function, optimizer):
    # enable train mode
    model.train()
    # initialize epoch's loss
    running_loss = 0.0

    for index, batch in enumerate(dataloader,1):
        _, (glove, lengths), labels = batch
        # we assume batch already in correct device through dataloader

        # sort fetaures per length for packing
        lengths, perm = torch.sort(lengths, descending=True)
        glove = glove[perm].float()
        labels = labels[perm].view(-1,1).float()

        # feedforward pass
        model.zero_grad()
        # get model prediction
        y_pred, _, _, _ = model(glove, lengths)
        # compute loss
        loss = loss_function(y_pred, labels)
        # backward pass: compute gradient wrt model parameters
        loss.backward()
        # update weights
        optimizer.step()
        running_loss += loss.item()
        # print statistics
        progress(loss=running_loss/index,
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_text_rnn(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    model.eval()
    running_loss = 0.0


    y_predicted = []  # the predicted labels
    y = []  # the gold labels
    # we don't want to keep gradients so everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            _, (glove, lengths), labels = batch
            lengths, perm = torch.sort(lengths, descending=True)
            glove = glove[perm].float()
            labels = labels[perm].view(-1,1)
            y_hat, _, _, _ = model(glove, lengths)
            # We compute the loss to compare train/test we dont backpropagate in test time
            loss = loss_function(y_hat, labels.float())
            # make predictions (class = argmax of posteriors)
            probs = torch.sigmoid(y_hat)
            #sntmnt_class = torch.argmax(y_hat, dim=1)
            sntmnt_class = torch.ge(probs,0.5).int()
            # collect the predictions, gold labels and batch loss
            pred = to_numpy(sntmnt_class)
            labels = to_numpy(labels.int())

            y_predicted.append(pred)
            y.append(labels)

            running_loss += loss.item()

    y_predicted = flatten_list(y_predicted)
    y = flatten_list(y)

    return running_loss / index, (y_predicted, y)

#############################################################################
#        AUDIO RNN TRAINING/EVALUATION HELPER FUNCTIONS
#############################################################################
def train_audio_rnn(_epoch, clip, dataloader,
                    model, loss_function, optimizer):
    # enable train mode
    model.train()
    # initialize epoch's loss
    running_loss = 0.0

    for index, batch in enumerate(dataloader,1):
        (covarep, lengths), (_, _), labels = batch
        # we assume batch already in correct device through dataloader

        # sort fetaures per length for packing
        lengths, perm = torch.sort(lengths, descending=True)
        covarep = covarep[perm].float()
        labels = labels[perm].view(-1,1).float()

        # feedforward pass
        model.zero_grad()
        # get model prediction
        y_pred, _, _, _ = model(covarep, lengths)
        # compute loss
        loss = loss_function(y_pred, labels)
        # backward pass: compute gradient wrt model parameters
        loss.backward()
        # clip gradients
        _ = torch.nn.utils.clip_grad_value_(model.parameters(), clip)
        # update weights
        optimizer.step()
        running_loss += loss.item()
        # print statistics
        progress(loss=running_loss/index,
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_audio_rnn(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    model.eval()
    running_loss = 0.0


    y_predicted = []  # the predicted labels
    y = []  # the gold labels
    # we don't want to keep gradients so everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            (covarep, lengths), _, labels = batch
            lengths, perm = torch.sort(lengths, descending=True)
            covarep =covarep[perm].float()
            labels = labels[perm].view(-1,1)
            y_hat, _, _, _ = model(covarep, lengths)
            # We compute the loss to compare train/test we dont backpropagate in test time
            loss = loss_function(y_hat, labels.float())
            # make predictions (class = argmax of posteriors)
            probs = torch.sigmoid(y_hat)
            #sntmnt_class = torch.argmax(y_hat, dim=1)
            sntmnt_class = torch.ge(probs,0.5).int()
            # collect the predictions, gold labels and batch loss
            pred = to_numpy(sntmnt_class)
            labels = to_numpy(labels.int())

            y_predicted.append(pred)
            y.append(labels)

            running_loss += loss.item()

    y_predicted = flatten_list(y_predicted)
    y = flatten_list(y)

    return running_loss / index, (y_predicted, y)

##################################################################################
#           2 CLASS CLASSIFICATION MODEL TRAINING FUNCTIONS
##################################################################################

def train_2_class_model(_epoch, clip, dataloader,
                        model, loss_function, optimizer):
    # enable train mode
    model.train()
    # initialize epoch's loss
    running_loss = 0.0

    for index, batch in enumerate(dataloader,1):
        (covarep, _), (glove, lengths), labels = batch
        # we assume batch already in correct device through dataloader

        # sort fetaures per length for packing
        lengths, perm = torch.sort(lengths, descending=True)
        glove = glove[perm].float()
        covarep = covarep[perm].float()
        labels = labels[perm].view(-1,1).float()

        # feedforward pass
        model.zero_grad()
        # get model prediction
        y_pred = model(covarep, lengths, glove, lengths)
        # compute loss
        loss = loss_function(y_pred, labels)
        # backward pass: compute gradient wrt model parameters
        loss.backward()
        # clip gradients
        #_ = torch.nn.utils.clip_grad_value_(model.parameters(), clip)
        # update weights
        optimizer.step()
        running_loss += loss.item()
        # print statistics
        progress(loss=loss.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index

###########################
###
###########################
def eval_2_class_model(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    model.eval()
    running_loss = 0.0


    y_predicted = []  # the predicted labels
    y = []  # the gold labels
    # we don't want to keep gradients so everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):

            (covarep, _), (glove, lengths), labels = batch

            # sort fetaures per length for packing
            lengths, perm = torch.sort(lengths, descending=True)
            glove = glove[perm].float()
            covarep = covarep[perm].float()
            labels = labels[perm].view(-1, 1).float()

            y_hat = model(covarep, lengths, glove, lengths)
            # We compute the loss to compare train/test we dont backpropagate in test time
            loss = loss_function(y_hat, labels)
            # make predictions (class = argmax of posteriors)
            probs = torch.sigmoid(y_hat)
            #sntmnt_class = torch.argmax(y_hat, dim=1)
            sntmnt_class = torch.ge(probs,0.5).int()
            # collect the predictions, gold labels and batch loss
            pred = to_numpy(sntmnt_class)
            labels = to_numpy(labels.int())

            y_predicted.append(pred)
            y.append(labels)

            running_loss += loss.item()

    y_predicted = flatten_list(y_predicted)
    y = flatten_list(y)

    return running_loss / index, (y_predicted, y)


##############################################################
###### GEOPAR'S ATTENTION TRAIN EVAL IMPLEMENTATION
##############################################################

def train_attention_model(_epoch, clip,
                          dataloader, model,
                          loss_list, loss_weights,
                          optimizer_list):
    # enable train mode
    model.train()
    # initialize epoch's loss
    running_total_loss = 0.0
    optimizer = optimizer_list[0]

    for index, batch in enumerate(dataloader,1):
        (covarep, _), (glove, lengths), labels = batch
        # we assume batch already in correct device through dataloader

        # sort fetaures per length for packing
        lengths, perm = torch.sort(lengths, descending=True)
        glove = glove[perm].float()
        covarep = covarep[perm].float()
        labels = labels[perm].view(-1,1).float()

        # feedforward pass
        model.zero_grad()
        # get model prediction
        fusion_pred, audio_pred, text_pred = model(covarep, glove, lengths)

        # compute loss
        f_loss = loss_list[0]
        a_loss = loss_list[1]
        t_loss = loss_list[2]

        fusion_loss = f_loss(fusion_pred, labels)*loss_weights[0]
        audio_loss = a_loss(audio_pred, labels)*loss_weights[1]
        t_loss = t_loss(text_pred, labels)*loss_weights[2]

        total_loss =  fusion_loss + audio_loss + t_loss

        # backward pass: compute gradient wrt model parameters
        total_loss.backward()
        # clip gradients
        _ = torch.nn.utils.clip_grad_value_(model.parameters(), clip)
        # update weights

        optimizer.step()

        running_total_loss += total_loss.item()
        # print statistics
        progress(loss=running_total_loss/index,
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))


    return running_total_loss / index


def eval_attention_model(dataloader, model, loss_list, loss_weights):
    # IMPORTANT: switch to eval mode
    model.eval()
    running_total_loss = 0.0

    fusion_predicted = []
    audio_predicted = []
    text_predicted = []

    y_predicted = []  # the predicted labels
    y = []  # the gold labels
    # we don't want to keep gradients so everything under
    # torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):

            (covarep, _), (glove, lengths), labels = batch

            # sort fetaures per length for packing
            lengths, perm = torch.sort(lengths, descending=True)
            glove = glove[perm].float()
            covarep = covarep[perm].float()
            labels = labels[perm].view(-1, 1).float()

            f_pred, a_pred, t_pred = model(covarep, glove, lengths)

            # compute loss
            f_loss = loss_list[0]
            a_loss = loss_list[1]
            t_loss = loss_list[2]

            fusion_loss = f_loss(f_pred, labels) * loss_weights[0]
            audio_loss = a_loss(a_pred, labels) * loss_weights[1]
            t_loss = t_loss(t_pred, labels) * loss_weights[2]

            total_loss = fusion_loss + audio_loss + t_loss

            # make predictions (class = argmax of posteriors)
            fusion_probs = torch.sigmoid(f_pred)
            audio_probs = torch.sigmoid(a_pred)
            text_probs = torch.sigmoid(t_pred)

            #sntmnt_class = torch.argmax(y_hat, dim=1)
            fusion_class = torch.ge(fusion_probs,0.5).int()
            audio_class = torch.ge(audio_probs,0.5).int()
            text_class = torch.ge(text_probs,0.5).int()

            # collect the predictions, gold labels and batch loss
            fusion_pred = to_numpy(fusion_class)
            audio_pred = to_numpy(audio_class)
            text_pred = to_numpy(text_class)

            labels = to_numpy(labels.int())

            fusion_predicted.append(fusion_pred)
            audio_predicted.append(audio_pred)
            text_predicted.append(text_pred)

            y.append(labels)
            y_predicted.append(fusion_pred)
            running_total_loss += total_loss.item()

    y_predicted = flatten_list(y_predicted)
    y = flatten_list(y)

    return running_total_loss / index, (y_predicted, y), audio_predicted, text_predicted
