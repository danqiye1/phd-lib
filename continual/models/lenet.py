import torch
import torch.nn as nn
from patterns.models import LeNetBase, LeNetHead

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