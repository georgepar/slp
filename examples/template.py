import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from argparse import ArgumentParser

from loguru import logger

from slp.util.log import configure_logging
from slp.config.config_parser import parse_config, make_cli_parser
from slp.plbind import (
    PLDataModuleFromDatasets,  # or PLDataModuleFromCorpus see slp/plbind/dm.py
    PLModule,  # or any other PLModule. See slp/plbind/module.py
    make_trainer,
    watch_model,
    FromLogits,
)


class MyCoolNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# Configuration format looks like this:
# CONFIG_FORMAT = {
#     "seed": None,
#     "debug": False,  # Perform a debugging run
#     "data": {
#         # Args for PLDataModuleFromDatasets or PLDataModuleFromCorpus
#     }
#     "model": {
#         # Model parameters to be passed when model is instantiated.
#     },
#     "optimizer": "Adam",  # Or any other optimizer from torch.optim
#     "optim": {
#         "lr": 1e-3,
#         "weight_decay": 1e-2,
#     },
#     "trainer": {
#         "experiment_name": "my-cool-experiment",
#         # Args for make_trainer
#     },
#     "lr_scheduler": True,  # Default support for ReduceLROnPlateau
#     "lr_schedule": {
#         # Args for ReduceLROnPlateau
#     },
# }


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


def setup():
    # Your config parsing goes here.
    parser = get_parser()
    parser = make_cli_parser(parser, parser.parse_args().config)
    configure_logging(f"logs/{config.trainer.experiment_name}")
    # Add all default command line parsers and merge with yaml config file.
    config = parse_config(parser, PLDataModuleFromDatasets)
    return config


def get_data():
    # Download the original train, dev and test splits.
    # If a dev or test split is not provided, then it will be split randomly from the training set
    # Return torch.utils.data.Dataset to use with PLDataModuleFromDatasets or raw corpora to use with PLDataModuleFromCorpus
    train, val, test = None, None, None
    return train, val, test


def get_lightning_module(config):
    # Create your model, optimizer, criterion and lr_scheduler.
    model = MyCoolNet(**config.model)

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)

    lr_scheduler = None
    if config.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    criterion = (
        nn.CrossEntropyLoss()
    )  # for classification or nn.MSELoss() for regression

    # Make your Lightning module. Note you can pass the metrics you need to monitor
    lm = PLModule(  # or RnnPLModule etc...
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
        hparams=config,  # This will automatically log configuration in wandb etc..
    )

    return lm


def get_lightning_data_module(config):
    # Get your data, preprocess, and create the LightningDataModule
    train, val, test = get_data()
    ldm = PLDataModuleFromDatasets(
        train, val=val, test=test, seed=config.seed, **config.data
    )
    return ldm


if __name__ == "__main__":
    # Boilerplate: Implement setup, get_data, get_lightning_data_module, get_lightning_module
    config = setup()

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    ldm = get_lightning_data_module(config)
    lm = get_lightning_module(config)

    if config.debug:
        # Debug run on a small dataset

        logger.info("Running in debug mode: Fast run on 5 batches")
        trainer = make_trainer(fast_dev_run=5)
        trainer.fit(lm, datamodule=ldm)

        logger.info("Running in debug mode: Overfitting 5 batches")
        trainer = make_trainer(overfit_batches=5)
        trainer.fit(lm, datamodule=ldm)

    else:
        # Train and evaluate the model
        trainer = make_trainer(**config.trainer)
        watch_model(
            trainer, model
        )  # make wandb track gradients, parameters and model graph

        trainer.fit(lm, datamodule=ldm)

        trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
