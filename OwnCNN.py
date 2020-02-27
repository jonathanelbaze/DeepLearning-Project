import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
from torchvision import datasets, transforms, models

import skorch
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt

from ImportExplo import *


# Load and Transform
path = os.getcwd()
print(path)

dataset_dir = "cars_train/"

print(dataset_dir)
train_tranform = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_tranform = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder(dataset_dir, transform=train_tranform)
print(train_maker_labels)