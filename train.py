import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import vgg19, densenet121, vgg16
from torchvision import datasets, models, transforms
import torchvision
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from collections import OrderedDict
import time
import json
import copy
import seaborn as sns
import numpy as np
from PIL import Image
from torch.autograd import Variable
import argparse
import json

parser = argparse.ArgumentParser(description='Train a model to recognize flowers')
parser.add_argument('data_directory', type=str, help='Directory containing data' , default='data_dir')
parser.add_argument('--gpu', type=bool, default=True, help='Whether to use GPU during training or not')
parser.add_argument('--arch', type=str, default='vgg19', help='Choose the architecture between VGG and Densenet')
parser.add_argument('--epochs', type=str, default=10, help='Choose number of epochs to train for')
parser.add_argument('--lr', type=str, default=0.0001, help='Choose learning rate to use for training')
args = parser.parse_args()

#ARGS HERE
arch = 'densenet121'
#arch = 'vgg19' alternatively
epochs = args.epochs
device = 'cuda' if args.gpu == True else 'cpu'

train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

# : Define your transforms for the training, validation, and testing sets
data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),                                                 
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                                 ]),
                   
                   'valid' : transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])
                                                ]),
                   
                   'test' : transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                               ])
                  }
                                                

# : Load the datasets with ImageFolder
image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                  'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
                 }

# : Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train' : DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
               'valid' : DataLoader(image_datasets['valid'], batch_size=32),
               'test' : DataLoader(image_datasets['test'], batch_size=32)
              }

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss/len(testloader), accuracy/len(testloader)

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear((25088 if arch == 'vgg19' else 1024),4096)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(4096,1024)),
    ('relu2', nn. ReLU()),
    ('dropout2', nn.Dropout(0.2)),
    ('fc3', nn.Linear(1024,102)),
    ('output', nn.LogSoftmax(dim=1))]))

model = getattr(models, arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = classifier
if args.gpu:
    model.cuda()
else:
    model.cpu()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)

model.train()

print_every = 40
steps = 0

for epoch in range(epochs):
    accuracy = 0
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:

        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(inputs)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        if steps % print_every == 0:
            test_loss, test_accuracy = validation(model, dataloaders['valid'], criterion, device)
            print("Epoch: {}/{}".format(epoch+1, epochs),
                  "Train Loss: {:.4f}".format(running_loss/print_every),
                  "Train Accuracy : {:.4f}".format(accuracy/print_every),
                  "Validation Loss : {:.4f}".format(test_loss),
                  "Validation Accuracy : {:.4f}".format(test_accuracy))
            model.train()
            accuracy = 0
            running_loss = 0

# Do validation on the test set
test_loss, test_accuracy = validation(model, dataloaders['test'], criterion, device)
print("Test Loss : {:.4f}".format(test_loss),
    "Test Accuracy : {:.4f}".format(test_accuracy))

# Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'arch' : arch,
              'classifier' : model.classifier,
              'state_dict' : model.state_dict(),
              'optimizer' : optimizer,
              'optimizer_dict' : optimizer.state_dict(),
              'epochs' : epochs,
              'class_to_idx' : model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')