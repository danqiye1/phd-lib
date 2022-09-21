import torch
import torch.nn as nn

class LeNet(nn.Module):
    """ Vanilla LeNet5 model """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.base = LeNetBase()
        self.head = LeNetHead(num_classes)
        
    def forward(self, img):
        X = self.base(img)
        X = self.head(X)
        return X

class MultiHeadLeNet(nn.Module):
    """ MultiHeadLeNet for Continual Learning """
    def __init__(self, num_classes=10, priors=[0], attention=False):
        super(MultiHeadLeNet, self).__init__()
        self.base = LeNetBase()
        self.heads = nn.ModuleList([LeNetHead(num_classes)])
        # Prior distribution of each task
        self.priors = priors
        # Keep track of the number of tasks
        self.task_embeddings = torch.nn.Embedding(1, 84)
        self.gate = torch.nn.Sigmoid()
        self.use_attention = attention

    def forward(self, img):
        """ Forward pass
        Note that there are 2 pathways; one for training; one for inference.
        Depending on whether use_attention is set, the hard-attention-to-task
        mechanism is used. Make sure to call self.consolidate() before inference
        to consolidate the task attention embeddings.
        """
        X = self.base(img)
        if self.training:
            if self.use_attention:
                task_id = len(self.heads) - 1
                X = X * self.mask(task_id).expand_as(X)
            X = self.heads[-1](X)
            self.priors[-1] += img.size(0)
        else:
            outputs = []
            for idx, head in enumerate(self.heads):
                if self.use_attention:
                    X = X * self.mask(idx).expand_as(X)
                outputs.append(torch.softmax(head(X), dim=1) * self.priors[idx] / sum(self.priors))
            X = torch.cat(outputs, dim=1)
        return X

    def mask(self, task_id, s=1):
        """ Hard attention to task embedding """
        task_tensor = torch.Tensor([task_id]).type(torch.LongTensor).cuda()
        return self.gate(s * self.task_embeddings(task_tensor))

    def add_head(self, num_classes):
        """ Add head for a new class """
        self.heads.append(LeNetHead(num_classes))
        old_embedding_weights = self.task_embeddings.weight.clone()
        new_embedding_weights = torch.nn.Embedding(1, 84).weight.clone().cuda()
        self.task_embeddings = torch.nn.Embedding(len(self.heads), 84)
        self.task_embeddings.weight = torch.nn.Parameter(
                                        torch.cat((old_embedding_weights, new_embedding_weights)))
        self.priors.append(0)

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

    def __init__(self, num_classes=10):
        super(LeNetHead, self).__init__()
        self.head = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, X):
        """ Forward method for training 
        
        Note only the last head gets trained as the rest are expected to be frozen.
        For evaluation, use forward_eval.
        """
        return self.head(X)

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