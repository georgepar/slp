# python examples/mnist_rnn.py --bsz 128 --bsz-eval 256

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
from slp.data.collators import SequenceClassificationCollator
from slp.modules.rnn import RNN

from slp.util.log import configure_logging
from slp.plbind import (
    PLDataModuleFromDatasets,
    RnnPLModule,
    make_trainer,
    watch_model,
    FromLogits,
)


collate_fn = SequenceClassificationCollator()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size=40, num_classes=10, bidirectional=False):
        super().__init__()
        self.encoder = RNN(input_size, hidden_size, bidirectional=bidirectional)
        out_size = hidden_size if not bidirectional else 2 * hidden_size
        self.clf = nn.Linear(out_size, num_classes)

    def forward(self, x, lengths):
        _, x, _ = self.encoder(x, lengths)
        out = self.clf(x)
        return out


def get_parser():
    parser = ArgumentParser("MNIST classification example")
    parser.add_argument(
        "--hidden",
        dest="model.hidden_size",
        type=int,
        help="Intermediate hidden layers for linear module",
    )
    parser.add_argument(
        "--bi",
        dest="model.bidirectional",
        action="store_true",
        help="Use BiLSTM",
    )
    return parser


def get_data():
    def squeeze(x):
        return x.squeeze()

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), squeeze])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)

    val = MNIST(download=False, root=".", transform=data_transform, train=False)
    return train, val


if __name__ == "__main__":
    # SETUP ##################################################
    parser = get_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)

    config = parse_config(parser, parser.parse_args().config)

    if config.trainer.experiment_name == "experiment":
        config.trainer.experiment_name = "mnist-rnn-classification"

    configure_logging(f"logs/{config.trainer.experiment_name}")

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    train, test = get_data()

    # Get data and make datamodule ##########################
    ldm = PLDataModuleFromDatasets(
        train, test=test, seed=config.seed, collate_fn=collate_fn, **config.data
    )

    # Create model, optimizer, criterion, scheduler ###########
    model = Net(28, **config.model)

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None
    if config.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    # Wrap in PLModule, & configure metrics ####################
    lm = RnnPLModule(
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
