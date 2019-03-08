import math
import sys
import torch
from config import DEVICE

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
            labels = labels[perm].view(-1,1).float()
            y_hat, _, _ = model(glove, lengths)
            # We compute the loss to compare train/test we dont backpropagate in test time
            loss = loss_function(y_hat, labels)
            # make predictions (class = argmax of posteriors)
            #sntmnt_class = torch.argmax(y_hat, dim=1)
            sntmnt_class = torch.ge(y_hat,0.5)
            # collect the predictions, gold labels and batch loss
            y_predicted.append(sntmnt_class)
            y.append(labels)

            running_loss += loss.item()

    return running_loss / index, (y_predicted, y)
