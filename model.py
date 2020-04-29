import pickle
import torch
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms, models

device = "cpu"

test_dir = "cars_test/"

test_tranform = transforms.Compose([transforms.Resize((250, 250)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_data = torchvision.datasets.ImageFolder(root=test_dir,
                                              transform=test_tranform)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=32,
                                          shuffle=True,
                                          )

a = torch.load("bestmodel.pth")

def eval_model(model):
    correct = 0.0
    total = 0.0
    pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = a(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred.append(predicted)

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (test_acc))
    return test_acc, pred

accuracy, preds = eval_model(a)

with open('acc_test.pkl', 'wb') as f:
    pickle.dump(accuracy, f)

with open('preds.pkl', 'wb') as f:
    pickle.dump(preds, f)

# plot the stats
with open('preds.pkl', 'rb') as f:
    predictions = pickle.load(f)

print(predictions)
