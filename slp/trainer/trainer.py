import os
import warnings

import torch
import torch.nn as nn

from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar

from slp.trainer.hanlders import CheckpointHandler
from slp.util import mktensor, from_checkpoint
from slp.util import system as sysutil


class Experiment(object):
    def __init__(self,
                 model,
                 optimizer,
                 checkpoint_dir='../../checkpoints',
                 experiment_name='experiment',
                 model_checkpoint=None,
                 optimizer_checkpoint=None,
                 patience=10,
                 dtype=torch.float,
                 device='cpu'):
        self.dtype = dtype
        self.device = device
        self.model = (from_checkpoint(model_checkpoint,
                                      model,
                                      map_location=torch.device('cpu'))
            .type(dtype)
            .to(device))
        self.optimizer = from_checkpoint(optimizer_checkpoint,
                                         optimizer)

        self.pbar = ProgressBar()

        self.checkpoint = CheckpointHandler(
            checkpoint_dir, experiment_name, score_name='validation_loss',
            score_function=self._score_fn, n_saved=2,
            require_empty=False, save_as_state_dict=True)

        self.early_stop = None

        self.patience = patience
        self.early_stop = EarlyStopping(
            patience, self._score_fn, self.trainer)



    @staticmethod
    def _score_fn(engine):
        """Returns the scoring metric for checkpointing and early stopping

        Args:
            engine (ignite.engine.Engine): The engine that calculates the val loss

        Returns:
            (float): The validation loss
        """
        return -engine.state.metrics['loss']


class Trial(Experiment):
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
