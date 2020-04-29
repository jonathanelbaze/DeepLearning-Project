import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import pickle
import skorch
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt

from ImportExplo import *

# Load and Transform
path = os.getcwd()
print(path)

train_dir = "cars_train/"
test_dir = "cars_test/"
print(train_dir)

train_tranform = transforms.Compose([transforms.Resize((250, 250)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_tranform = transforms.Compose([transforms.Resize((250, 250)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.ImageFolder(root=train_dir,
                                              transform=train_tranform)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=32,
                                           shuffle=True,
                                           )

test_data = torchvision.datasets.ImageFolder(root=test_dir,
                                              transform=test_tranform)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=32,
                                          shuffle=True,
                                          )


device = "cpu"
print(device)

def train_model(model, criterion, optimizer, scheduler, n_epochs=5):
    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs and assign them to cuda
            inputs, labels = data
            # inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
            print(1)

        epoch_duration = time.time() - since
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 / 32 * running_correct / len(train_loader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch + 1, epoch_duration, epoch_loss, epoch_acc))

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # switch the model to eval mode to evaluate on test data
        model.eval()
        test_acc = eval_model(model)
        test_accuracies.append(test_acc)

        # re-set the model to train mode after validating
        model.train()
        scheduler.step(test_acc)
        since = time.time()
    print('Finished Training')
    return model, losses, accuracies, test_accuracies


def eval_model(model):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (test_acc))
    return test_acc

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
models.
#model_ft = models.resnet34(pretrained=True)
#num_ftrs = model_ft.fc.in_features

# replace the last fc layer with an untrained one (requires grad by default)
model_ft.fc = nn.Linear(num_ftrs, 49)
model_ft = model_ft.to(device)

# uncomment this block for half precision model
"""
model_ft = model_ft.half()


for layer in model_ft.modules():
    if isinstance(layer, nn.BatchNorm2d):
        layer.float()
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

"""
probably not the best metric to track, but we are tracking the training accuracy and measuring whether
it increases by atleast 0.9 per epoch and if it hasn't increased by 0.9 reduce the lr by 0.1x.
However in this model it did not benefit me.
"""
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

model_ft, training_losses, training_accs, test_accs = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=10)

torch.save(model_ft.state_dict(), "C:/Users/Georges/PycharmProjects/DeepLearning-Project/bestmodel_dic.pth")
torch.save(model_ft, "C:/Users/Georges/PycharmProjects/DeepLearning-Project/bestmodel.pth")

with open('training_acc.pkl', 'wb') as f:
    pickle.dump(training_accs, f)

with open('test_acc.pkl', 'wb') as f:
    pickle.dump(test_accs, f)

# plot the stats
with open('training_acc.pkl', 'rb') as f:
    training_acc = pickle.load(f)

with open('test_acc.pkl', 'rb') as f:
    test_acc = pickle.load(f)

f, axarr = plt.subplots(2,2, figsize = (12, 8))
axarr[0, 1].plot(training_acc)
axarr[0, 1].set_title("Training acc")
axarr[1, 0].plot(test_acc)

axarr[1, 0].set_title("Test acc")