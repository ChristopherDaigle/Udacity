__author__ = "Chris"

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def build_classifier(model, input_units, hidden_units, classes, dropout):
    """
    Function to build a new classifier

    :param model: type of model
    :param input_units: number of input units to the NN
    :param hidden_units: number of hidden units of the NN
    :param classes: number of classes to categorize
    :param dropout: probability of dropout

    :return: classified but untrained model
    """
    # Weights of the pretrained model should be frozen so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define the NN structure
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        'fc1', nn.Linear(input_units, hidden_units),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replace the pretrained classifier
    model.classifier = classifier

    return model


def validation(model, valid_loader, criterion, gpu):
    """
    Function to validate the trained model

    :param model: type of model
    :param valid_loader: transformed validation data
    :param criterion: loss function
    :param gpu: gpu mode (T/F)
    :return: loss value and accuracy
    """
    valid_loss = 0
    accuracy = 0

    if gpu == True:
        images, labels = images.to('cuda'), labels.to('cuda')
    else:
        pass

    for ii, (images, labels) in enumerate(valid_loader):
        if gpu == True:
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass

        outputs = model.forward(images)
        valid_loss += criterion(outputs, labels).item()
        probs = torch.exp(outputs)
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


def train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, gpu):
    """
    Function to train neural network

    :param model: type of model
    :param epochs: number of epochs
    :param train_loader: transformed training data
    :param valid_loader: transformed validation data
    :param criterion: loss function
    :param optimizer: optimization method
    :param gpu: gpu mode (T/F)
    :return:
    """
    steps = 0
    print_every = 10

    if gpu == True:
        model.to('cuda')
    else:
        pass

    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            if gpu == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass

            # zero out gradients
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validate
            if steps % print_every == 0:
                # set model to evaluation mode
                model.eval()
                # Turn off gradients (not training)
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, gpu)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                running_loss = 0
                # Turn training mode back on
                model.train()
    return model, optimizer
