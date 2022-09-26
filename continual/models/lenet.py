import torch
import torch.nn as nn
from patterns.models import LeNetBase, LeNetHead

class MultiHeadLeNet(nn.Module):
    """ MultiHeadLeNet for Continual Learning """
    def __init__(self, priors=[], device=torch.device("cpu")):
        super(MultiHeadLeNet, self).__init__()
        self.base = LeNetBase()
        self.heads = nn.ModuleList([])
        # Prior distribution of each task
        self.priors = priors

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
            self.priors[-1] += img.size(0)
        else:
            outputs = []
            for idx, head in enumerate(self.heads):
                outputs.append(torch.softmax(head(X), dim=1) * self.priors[idx] / sum(self.priors))

            X = torch.cat(outputs, dim=1)
        return X

    def add_head(self, num_classes):
        """ Add head for a new class """
        self.heads.append(LeNetHead(num_classes))
        self.priors.append(0)