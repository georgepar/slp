import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from argparse import ArgumentParser
from torchvision.transforms import Compose, ToTensor, Normalize  # type: ignore
from torchvision.datasets import MNIST  # type: ignore

from loguru import logger

from slp import configure_logging
from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.module import PLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.plbind.helpers import FromLogits


pl.utilities.seed.seed_everything(seed=42)


CONFIG = {
    "model": {"intermediate_hidden": 50},
    "optimizer": {"name": "Adam", "lr": 1e-3},
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--hidden", dest="model.intermediate_hidden", type=int, default=12
        )
        return parser


def get_parser():
    parser = ArgumentParser("MNIST classification example")
    return parser


def optimizer_parser(parent_parser):
    optimizer_parser = ArgumentParser(parents=[parent_parser], add_help=False)
    optimizer_parser.add_argument(
        "--name", dest="optimizer.name", type=str, default="SGD"
    )
    optimizer_parser.add_argument("--lr", dest="optimizer.lr", type=float, default=1e-2)
    return optimizer_parser


def get_data():
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)

    val = MNIST(download=False, root=".", transform=data_transform, train=False)
    return train, val


if __name__ == "__main__":
    EXPERIMENT_NAME = "mnist-classification"
    configure_logging(f"logs/{EXPERIMENT_NAME}")

    parser = get_parser()
    parser = optimizer_parser(parser)
    parser = Net.add_model_specific_args(parser)

    dict_config = OmegaConf.create(CONFIG)
    user_cli, default_cli = OmegaConf.from_argparse(parser)

    config = OmegaConf.merge(default_cli, dict_config, user_cli)

    train, test = get_data()

    ldm = PLDataModuleFromDatasets(
        train, test=test, batch_size=128, batch_size_eval=256
    )

    model = Net(intermediate_hidden=config.model.intermediate_hidden)
    optimizer = optim.__getattribute__(config.optimizer.name)(
        model.parameters(), lr=config.optimizer.lr
    )
    criterion = nn.CrossEntropyLoss()

    lm = PLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
        hparams=config,
    )

    trainer = make_trainer(EXPERIMENT_NAME, max_epochs=100, gpus=1)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
