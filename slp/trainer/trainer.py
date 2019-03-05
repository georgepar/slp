import os

import torch
from slp.util import system as sysutil
import torch.nn as nn
from slp.util import mktensor


class _BaseTrial(object):
    def __init__(self,
                 model,
                 optimizer=None,
                 checkpoint_dir='../../checkpoints',
                 checkpoint=None,
                 dtype=torch.float,
                 device='cpu'):
        self.dtype = dtype
        self.device = device
        if checkpoint:
            model, optimizer = self._load_checkpoint(
                checkpoint_dir, checkpoint, model, optimizer)
        self.model = model.type(dtype).to(device)
        self.optimizer = optimizer

    @staticmethod
    def _load_checkpoint(checkpoint_dir, checkpoint,
                         model, optimizer=None):
        if sysutil.is_subpath(checkpoint, checkpoint_dir):
            checkpoint = os.path.join(checkpoint_dir, checkpoint)
        state_dict = torch.load(checkpoint)
        model_state_dict = state_dict
        if 'optimizer' in state_dict and optimizer is not None:
            optimizer.load_state_dict(state_dict['optimizer'])
        if 'model' in state_dict:
            model_state_dict = state_dict['model']
        model.load_state_dict(model_state_dict)
        return model, optimizer


class Trial(_BaseTrial):
    def __init__(self,
                 model,
                 optimizer=None,
                 checkpoint_dir='../../checkpoints',
                 checkpoint=None,
                 dtype=torch.float,
                 device='cpu'):
        super(Trial, self).__init__(model,
                                    optimizer=optimizer,
                                    checkpoint_dir=checkpoint_dir,
                                    checkpoint=checkpoint,
                                    dtype=dtype,
                                    device=device)


    def attach(self):
        pass

    def train(self, train_loader, val_loader):
        pass

    def eval(self, test_loader):
        pass
