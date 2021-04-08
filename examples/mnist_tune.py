# python examples/mnist.py --bsz 128 --bsz-eval 256
import math
import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from loguru import logger
from ray import tune
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import Compose, Normalize, ToTensor  # type: ignore

from slp.config.config_parser import make_cli_parser, parse_config
from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.plbind import (
    FromLogits,
    PLDataModuleFromDatasets,
    PLModule,
    make_trainer,
    make_trainer_for_ray_tune,
)
from slp.plbind.dm import split_data
from slp.util.log import configure_logging
from slp.util.tuning import run_tuning


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


def get_data(seed=None, train=True, val=True, test=False):
    # Fix: https://stackoverflow.com/a/66820249
    MNIST.resources = [
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    with FileLock(os.path.expanduser("~/data.lock")):
        train_data = MNIST(
            download=True, root=".", transform=data_transform, train=True
        )

    train_data, val_data = split_data(train_data, 0.2, seed)
    test_data = None

    if test:
        test_data = MNIST(
            download=False, root=".", transform=data_transform, train=False
        )

    return train_data, val_data, test_data


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


def train_mnist(config, train=None, val=None):
    # Convert dictionary to omegaconf dictconfig object
    config = OmegaConf.create(config)

    # Create data module
    ldm = PLDataModuleFromDatasets(
        train, val=val, seed=config.seed, no_test_set=True, **config.data
    )

    # Create model, optimizer, criterion, scheduler
    model = Net(**config.model)

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None

    if config.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    # Wrap in PLModule, & configure metrics
    lm = PLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        metrics={
            "acc": FromLogits(pl.metrics.classification.Accuracy())
        },  # Will log train_acc and val_acc
        hparams=config,
    )

    # Map Lightning metrics to ray tune metris.
    metrics_map = {"accuracy": "val_acc", "validation_loss": "val_loss"}
    assert (
        config["tune"]["metric"] in metrics_map.keys()
    ), "Metrics mapping should contain the metric you are trying to optimize"
    # Train model
    trainer = make_trainer_for_ray_tune(metrics_map=metrics_map, **config.trainer)

    trainer.fit(lm, datamodule=ldm)


def configure_search_space(config):
    config["model"] = {
        "intermediate_hidden": tune.choice([16, 32, 64, 100, 128, 256, 300, 512])
    }
    config["optimizer"] = tune.choice(["SGD", "Adam", "AdamW"])
    config["optim"]["lr"] = tune.loguniform(1e-4, 1e-1)
    config["optim"]["weight_decay"] = tune.loguniform(1e-4, 1e-1)
    config["data"]["batch_size"] = tune.choice([16, 32, 64, 128])

    return config


if __name__ == "__main__":
    # SETUP ##################################################
    parser = get_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)  # type: ignore

    config = parse_config(parser, parser.parse_args().config)

    if config.trainer.experiment_name == "experiment":
        config.trainer.experiment_name = "mnist-classification"

    configure_logging()

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    # These arguments may be provided from the command line or a config file
    # config = OmegaConf.to_container(config)
    # config["tune"] = {
    #     "num_trials": 10,
    #     "cpus_per_trial": 1,
    #     "gpus_per_trial": 0.12,
    #     "metric": "accuracy",
    #     "mode": "max",
    # }
    # config["wandb"]["project"] = "tuning-mnist-classification"
    # config["trainer"]["max_epochs"] = 15
    # config = OmegaConf.create(config)

    # Handle train / val splitting.
    # All trials should run on the same validation set
    train, val, _ = get_data()
    best_config = run_tuning(
        config,  # type: ignore
        "configs/best.mnist.tune.yml",
        train_mnist,
        configure_search_space,
        train,
        val,
    )
