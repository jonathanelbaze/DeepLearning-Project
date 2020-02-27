import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import skorch
import sklearn
import numpy as np

import matplotlib

# Load and Transform
dataset_dir = "../car_train/"

train_tranform = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# test = transforms.Compose([transforms.Resize((400, 400)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = train_tfms)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True, num_workers = 2)
#
# dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir+"test", transform = test_tfms)
# testloader = torch.utils.data.DataLoader(dataset2, batch_size = 32, shuffle=False, num_workers = 2)
