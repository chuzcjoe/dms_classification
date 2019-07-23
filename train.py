import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import os
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

 
model_path = 'vgg16.pth'
BATCH_SIZE = 128
LR = 0.01
EPOCH = 10
data = '../DMS/data/'
 
class VGG(nn.Module):
    def __init__(self, features, num_classes=3):
        super(VGG, self).__init__()
        
        self.features = features
        self.classifier = nn.Sequential( 
            #fc6
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
 
            #fc7
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
 
            #fc8
            nn.Linear(4096, num_classes))
        
        self._initialize_weights()
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
 
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
 

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers) 
 
def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)
    #model.load_state_dict(torch.load(model_path))
    return model
 
def getData(): 
    traindir = os.path.join(data, 'train')
    valdir = os.path.join(data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Grayscale(3),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Grayscale(3),
            transforms.RandomResizedCrop(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
    classes = ('bg','drink','phone')
    return train_loader, val_loader, classes
 
def train():
    trainset_loader, testset_loader, _ = getData()
    net = vgg16()
    #net.train()
    print(net)
 
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
 
    # Train the model
    for epoch in range(10):
        for step, (inputs,labels) in enumerate(trainset_loader):
            optimizer.zero_grad() 
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if step % 10 ==9:
                acc = test(net, testset_loader)
                print('Epoch', epoch, '|step ', step, 'loss: %.4f' %loss.item(), 'test accuracy:%.4f' %acc)
    print('Finished Training')
    return net
 
def test(net, testdata):
    correct, total = .0, .0
    for inputs, labels in testdata:
        net.eval()
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    #net.train()
    return float(correct) / total
 
if __name__ == '__main__':
    net = train()

