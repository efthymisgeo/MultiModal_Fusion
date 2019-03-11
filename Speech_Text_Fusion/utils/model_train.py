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
        y_pred, _, _ = model(glove, lengths)
        # compute loss
        loss = loss_function(y_pred, labels)
        # backward pass: compute gradient wrt model parameters
        loss.backward()
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
            y_hat, _, _ = model(glove, lengths)
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
#### AUDIO RNN TRAINING/EVALUATION HELPER FUNCTIONS
#############################################################################
def train_audio_rnn(_epoch, dataloader, model, loss_function, optimizer):
    # enable train mode
    model.train()
    # initialize epoch's loss
    running_loss = 0.0

    for index, batch in enumerate(dataloader,1):
        (covarep, lengths), _, labels = batch
        # we assume batch already in correct device through dataloader

        # sort fetaures per length for packing
        lengths, perm = torch.sort(lengths, descending=True)
        covarep = covarep[perm].float()
        labels = labels[perm].view(-1,1).float()

        # feedforward pass
        model.zero_grad()
        # get model prediction
        y_pred, _, _ = model(covarep, lengths)
        # compute loss
        loss = loss_function(y_pred, labels)
        # backward pass: compute gradient wrt model parameters
        loss.backward()
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
            covarep = covarep[perm].float()
            labels = labels[perm].view(-1,1)
            y_hat, _, _ = model(covarep, lengths)
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
