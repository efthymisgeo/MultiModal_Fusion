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
        y_pred,_ = model(glove, lengths)
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
        for index, batch in enumerate(dataloader, 1):
            _, (glove, lengths), labels = batch
            lengths, perm = torch.sort(lengths, descending=True)
            glove = glove[perm].float()
            labels = labels[perm].view(-1,1).float()
            y_hat, _ = model(glove, lengths)
            # We compute the loss to compare train/test we dont backpropagate in test time
            loss = loss_function(y_hat, labels)
            # make predictions (class = argmax of posteriors)
            #sntmnt_class = torch.argmax(y_hat, dim=1)
            sntmnt_class = torch.ge(y_hat,0.5)
            # collect the predictions, gold labels and batch loss
            y_predicted.append(sntmnt_class)

            running_loss += loss.item()

    return running_loss / index, (y_predicted, y)


def train_dataset(_epoch, dataloader, model, loss_function, optimizer, sort):
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0
    # obtain the model's device ID
    device = next(model.parameters()).device

    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths = batch

        # move the batch tensors to the right device
        inputs.to(device)  # EX9
        labels.to(device)
        lengths.to(device)
        # Step 0 - sort the batch if needed
        if sort == True:
            lengths, perm = torch.sort(lengths,
                                       descending=True)
            inputs = inputs[perm]
            labels = labels[perm]
        #lengths = [i for i in lengths]
        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        model.zero_grad()  # EX9
        #model.init_hidden()
        # Step 2 - forward pass: y' = model(x)
        y_pred = model(inputs, lengths)
        ##### for MR Dataset #####
        #y_pred = y_pred.view(-1)# EX9
        # loss needs torch.double inputs
        # Step 3 - compute loss: L = loss_function(y', y)
        loss = loss_function(y_pred.type(torch.FloatTensor),
                             labels.type(torch.LongTensor)) # EX9
        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()  # EX9

        # Step 5 - update weights
        optimizer.step()  # EX9

        running_loss += loss.item()

        # print statistics
        progress(loss=loss.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_dataset(dataloader, model, loss_function, sort):
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_predicted = []  # the predicted labels
    y = []  # the gold labels

    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # get the inputs (batch)
            inputs, labels, lengths = batch
            # Step 0 - sort the batch if needed
            if sort == True:
                lengths, perm = torch.sort(lengths,
                                           descending=True)
                inputs = inputs[perm]
                labels = labels[perm]

                # Step 1 - move the batch tensors to the right device
            inputs.to(device)  # EX9
            labels.to(device)
            lengths.to(device)

            # Step 2 - forward pass: y' = model(x)
            #y_hat = model(inputs, lengths).view(-1)  # EX9
            y_hat = model(inputs, lengths)
            #y_pred, _ = torch.max(y_hat, dim=1)
            # Step 3 - compute loss.
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            loss = loss_function(y_hat.type(torch.FloatTensor),
                                 labels.type(torch.LongTensor))  # EX9

            # Step 4 - make predictions (class = argmax of posteriors)
            sntmnt_class = torch.argmax(y_hat, dim=1)  # EX9
            #sntmnt_class = torch.ge(y_hat,0.5)
            # Step 5 - collect the predictions, gold labels and batch loss
            y_predicted.append(sntmnt_class)  # EX9

            running_loss += loss.item()

    return running_loss / index, (y_predicted, y)