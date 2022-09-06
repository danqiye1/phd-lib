import torch
import torch.nn as nn

from pdb import set_trace as bp

class LeNet(nn.Module):
    """ Vanilla LeNet5 model """
    
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.base = LeNetBase()
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)
        
    def forward(self, img):
        X = self.base(img)
        X = self.fc2(X)
        return X

class LeNetBase(nn.Module):
    """ Base of Multi-head LeNet5 Model """
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.convnet = LeNetConv()
        self.fc1 = nn.Linear(in_features=120, out_features=84)

    def forward(self, img):
        X = self.convnet(img)
        X = torch.flatten(X, start_dim=1)
        X = self.fc1(X)
        X = nn.functional.relu(X)
        return X

class LeNetHead(nn.Module):
    """ Multi-head LeNet5 model """

    def __init__(self, base: LeNetBase, num_classes=10):
        super(LeNetHead, self).__init__()
        self.base = base
        self.head = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, img):
        """ Forward method for training 
        
        Note only the last head gets trained as the rest are expected to be frozen.
        For evaluation, use forward_eval.
        """
        X = self.base(img)
        X = self.head(X)
        return X

    def freeze(self):
        """ Freeze all layers """
        for params in self.parameters():
            params.requires_grad = False

class LeNetConv(nn.Module):
    """ LeNet5 Convolution Layers """

    def __init__(self):
        super(LeNetConv, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU(inplace=True),
        )

    def forward(self, img):
        return self.convnet(img)