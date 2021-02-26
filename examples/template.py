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
    PLDataModuleFromDatasets,  # or PLDataModuleFromCorpus see slp/plbind/dm.py
    PLModule,  # or any other PLModule. See slp/plbind/module.py
    make_trainer,
    watch_model,
    FromLogits,
    add_optimizer_args,
    add_trainer_args,
)

# Could be read from yaml with OmegaConf.from_yaml
# Configuration format looks like this:
CONFIG = {
    "seed": None,
    "debug": False,  # Perform a debugging run
    "data": {
        # Args for PLDataModuleFromDatasets or PLDataModuleFromCorpus
    }
    "model": {
        # Model parameters to be passed when model is instantiated.
    },
    "optimizer": "Adam",  # Or any other optimizer from torch.optim
    "optim": {
        "lr": 1e-3,
        "weight_decay": 1e-2,
    },
    "trainer": {
        "experiment_name": "my-cool-experiment",
        # Args for make_trainer
    },
    "lr_scheduler": True,  # Default support for ReduceLROnPlateau
    "lr_schedule": {
        # Args for ReduceLROnPlateau
    },
}


def get_parser():
    parser = ArgumentParser("My cool experiment")
    # Add model args here. dest should be model.arg_name
    # For example
    # parser.add_argument(
    #     "--hidden",
    #     dest="model.intermediate_hidden",
    #     type=int,
    #     default=12,
    #     help="Intermediate hidden layers for linear module",
    # )
    return parser


def get_data():
    # Download the original train, dev and test splits.
    # If a dev or test split is not provided, then it will be split randomly from the training set
    # Return torch.utils.data.Dataset to use with PLDataModuleFromDatasets or raw corpora to use with PLDataModuleFromCorpus
    train, val, test = None, None, None
    return train, val, test


if __name__ == "__main__":
    # Make default argument parser
    parser = get_parser()
    parser = add_optimizer_args(parser)
    parser = add_trainer_args(parser)
    parser = PLDataModuleFromDatasets.add_argparse_args(parser)  # or PLDataModuleFromCorpus.add_argparse_args(parser)

    config_file = parser.parse_args().config  # Path to config file

    # Merge Configurations Precedence: default kwarg values < default argparse values < config file values < user provided CLI args values
    if config_file is not None:
        dict_config = OmegaConf.from_yaml(config_file)
    else:
        # dict_config = OmegaConf.create(CONFIG)
        dict_config = OmegaConf.create({})
    user_cli, default_cli = OmegaConf.from_argparse(parser)

    config = OmegaConf.merge(default_cli, dict_config, user_cli)

    # Setup logging module.
    EXPERIMENT_NAME = config.trainer.experiment_name
    configure_logging(f"logs/{EXPERIMENT_NAME}")

    logger.info("Running with the following configuration")
    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    train, val, test = get_data()

    ldm = PLDataModuleFromDatasets(train, val=val, test=test, seed=config.seed, **config.data)

    model = MyCoolNet(**config.model)

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
    criterion = nn.CrossEntropyLoss()  # for classification or nn.MSELoss() for regression

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
