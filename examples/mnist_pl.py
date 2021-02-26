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

from slp import configure_logging
from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.plbind import (
    PLDataModuleFromDatasets,
    PLModule,
    make_trainer,
    watch_model,
    FromLogits,
    add_optimizer_args,
    add_trainer_args,
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


def get_data():
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)

    val = MNIST(download=False, root=".", transform=data_transform, train=False)
    return train, val


if __name__ == "__main__":
    EXPERIMENT_NAME = "mnist-classification"
    configure_logging(f"logs/{EXPERIMENT_NAME}")

    parser = get_parser()
    parser = add_optimizer_args(parser)
    parser = add_trainer_args(parser)
    parser = PLDataModuleFromDatasets.add_argparse_args(parser)

    dict_config = OmegaConf.create(CONFIG)
    user_cli, default_cli = OmegaConf.from_argparse(parser)

    config = OmegaConf.merge(default_cli, dict_config, user_cli)

    logger.info("Running with the following configuration")
    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    train, test = get_data()

    ldm = PLDataModuleFromDatasets(train, test=test, seed=config.seed, **config.data)

    model = Net(**config.model)

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None
    if config.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    lm = PLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
        hparams=config,
    )

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
