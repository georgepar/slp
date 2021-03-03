# python examples/mnist.py --bsz 128 --bsz-eval 256

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

import pytorch_lightning as pl
from argparse import ArgumentParser
from torchvision.transforms import Compose, ToTensor, Normalize  # type: ignore
from torchvision.datasets import MNIST  # type: ignore

from loguru import logger

from slp.config.config_parser import parse_config, make_cli_parser
from slp.util.log import configure_logging
from slp.plbind import (
    PLDataModuleFromDatasets,
    PLModule,
    make_trainer,
    watch_model,
    FromLogits,
)

# Could be read from yaml with OmegaConf.from_yaml
CONFIG = {
    "model": {"intermediate_hidden": 100},
    "optimizer": "Adam",
    "optim": {"lr": 1e-3},
}


class Net(nn.Module):
    def __init__(self, intermediate_hidden=50):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, intermediate_hidden)
        self.fc2 = nn.Linear(intermediate_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def get_data():
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)

    val = MNIST(download=False, root=".", transform=data_transform, train=False)
    return train, val


def get_parser():
    parser = ArgumentParser("MNIST classification example")
    parser.add_argument(
        "--hidden",
        dest="model.intermediate_hidden",
        type=int,
        default=12,
        help="Intermediate hidden layers for linear module",
    )
    return parser


if __name__ == "__main__":
    # SETUP ##################################################
    parser = get_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)

    config = parse_config(parser, parser.parse_args().config)

    if config.trainer.experiment_name == "experiment":
        config.trainer.experiment_name = "mnist-classification"

    configure_logging(f"logs/{config.trainer.experiment_name}")

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    # Get data and make datamodule ##########################
    train, test = get_data()

    ldm = PLDataModuleFromDatasets(train, test=test, seed=config.seed, **config.data)

    # Create model, optimizer, criterion, scheduler ###########
    model = Net(**config.model)

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None
    if config.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    # Wrap in PLModule, & configure metrics ####################
    lm = PLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
        hparams=config,
    )

    # Run debugging session or fit & test the model ############
    if config.debug:
        logger.info("Running in debug mode: Fast run on 5 batches")
        trainer = make_trainer(fast_dev_run=5)
        trainer.fit(lm, datamodule=ldm)

        logger.info("Running in debug mode: Overfitting 5 batches")
        trainer = make_trainer(overfit_batches=5)
        trainer.fit(lm, datamodule=ldm)

    else:
        trainer = make_trainer(**config.trainer)
        watch_model(trainer, model)

        trainer.fit(lm, datamodule=ldm)

        trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())

        logger.info("Run finished. Uploading files to wandb...")
