__author__ = "Chris"

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models


# Data pipeline
def pipeline(data_dir):
    """
    Function to import, transform, and load data

    :param data_dir: directory for the data to be transformed
    :return: transformed data
    """
    train_transforms = transforms.Compose(
        [transforms.RandomRotation(75),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    valid_and_test_transforms = transforms.Compose(
        [transforms.Resize(225),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    # Transform data
    train_data = datasets.ImageFolder(data_dir + '/train',
                                      transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid',
                                      transform=valid_and_test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test',
                                     transform=valid_and_test_transforms)
    # Load transformed data
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=64,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=32)

    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data


def process_image(image):
    """
    Function to process an image

    :param image:
    :return: NumPy array
    """
    # Convert image to PIL image from file path
    pil_im = Image.open(f'{image}' + '.jpg')
    # Image transform framework
    transform = transforms.Compose(
        [transforms.Resize(225),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    # Transform the image
    pil_tfd = transform(pil_im)
    # Convert to NumPy Array
    array_im_tfd = np.array(pil_tfd)
    # Conver to torch from NumPy array
    img_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)
    # Add dimensions for (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    return img_add_dim
