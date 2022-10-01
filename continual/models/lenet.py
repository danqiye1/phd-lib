import torch
import torch.nn as nn
from functools import reduce
from patterns.models import LeNetBase, LeNetHead

class MultiHeadLeNet(nn.Module):
    """ MultiHeadLeNet for Continual Learning """
    def __init__(self, device=torch.device("cpu"), benchmark='SplitMNIST'):
        super(MultiHeadLeNet, self).__init__()
        self.base = LeNetBase()
        self.heads = nn.ModuleList([])
        assert benchmark in ['SplitMNIST', 'PermutedMNIST'], \
            "Benchmark must be SplitMNIST or PermutedMNIST!"
        self.benchmark = benchmark

        # Keep track of which device model is on
        self.device = device

    def forward(self, img):
        """ Forward pass
        Note that there are 2 pathways; one for training; one for inference.
        Depending on whether use_attention is set, the hard-attention-to-task
        mechanism is used.
        """
        X = self.base(img)
        if self.training:
            X = self.heads[-1](X)
        else:
            outputs = []
            for _, head in enumerate(self.heads):
                outputs.append(torch.softmax(head(X), dim=1))

            if self.benchmark == "SplitMNIST":
                X = torch.cat(outputs, dim=1)
            else:
                X = reduce(lambda a, b: a + b, outputs)
                X = X / len(outputs)
        return X

    def add_head(self, num_classes):
        """ Add head for a new class """
        # This fix part of the issue
        for head in self.heads:
            head.requires_grad_(False)
        self.heads.append(LeNetHead(num_classes))