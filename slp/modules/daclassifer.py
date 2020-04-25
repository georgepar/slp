import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data.sampler import Sampler

from slp.modules.feedforward import FF

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class DALoss(nn.Module):
    def __init__(self, loss_fn_cl, loss_fn_d):
        super(DALoss, self).__init__()
        self.loss_fn_cl = loss_fn_cl
        self.loss_fn_d = loss_fn_d

    def forward(self, pred, tar, domain_pred, domain_targets, epoch):
        predictions = torch.stack([p for p,t in zip(pred, tar) if t>=0])
        targets = torch.stack([t for t in tar if t>=0])
        loss_cl = self.loss_fn_cl(predictions, targets)
        loss_d = self.loss_fn_d(domain_pred, domain_targets)
        if epoch > 2:
            return loss_cl + 0.01 * loss_d #NOTSURE
        else:
            return loss_cl

class DAClassifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes, num_domains):
        super(DAClassifier, self).__init__()
        self.encoder = encoder
        self.clf = FF(encoded_features, num_classes,
                      activation='none', layer_norm=False,
                      dropout=0.)
        self.da = FF(encoded_features, num_domains,
                      activation='none', layer_norm=False,
                      dropout=0.)

    def forward(self, x, lengths):
        x = self.encoder(x, lengths)
        y = grad_reverse(x)
        return self.clf(x), self.da(y)

class DASubsetRandomSampler(Sampler):
    def __init__(self, indices_source, indices_target, s_dataset_size, num_source, batch_size):  
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.s_dataset_size = s_dataset_size
        self.num_source = num_source
        self.num_target = batch_size - num_source
        self.batch_size = batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.indices_source))
        tarperm = torch.randperm(len(self.indices_target))
        T = 0
        for i, s in enumerate(perm, 1): 
            yield self.indices_source[s]
            if i % self.num_source == 0:
                for j in range(self.num_target):
                    t = T + j
                    yield self.s_dataset_size + self.indices_target[tarperm[t]]
                T = t + 1 

    def __len__(self):
        n_full = int(np.floor(len(self.indices_source) / self.num_source))
        last = len(self.indices_source) % self.num_source
        return int(n_full * self.batch_size + last)
