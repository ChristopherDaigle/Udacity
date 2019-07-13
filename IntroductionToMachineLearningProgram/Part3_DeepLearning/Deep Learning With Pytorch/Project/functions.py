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
    :return:
    """
    valid_loss = 0
    accuracy = 0
