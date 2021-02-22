import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torchvision.transforms import Compose, ToTensor, Normalize  # type: ignore
from torchvision.datasets import MNIST  # type: ignore

from loguru import logger

from slp import configure_logger
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.module import AutoEncoderPLModule

import pytorch_lightning as pl


EXPERIMENT_NAME = "mnist-autoencoder"

configure_logger(f"logs/{EXPERIMENT_NAME}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        # encode
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        # decode
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x


def get_data():
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)

    val = MNIST(download=False, root=".", transform=data_transform, train=False)
    return train, val


if __name__ == "__main__":
    train, test = get_data()

    ldm = PLDataModuleFromDatasets(
        train, test=test, batch_size=128, batch_size_eval=256
    )

    model = Net()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    lm = AutoEncoderPLModule(model, optimizer, criterion)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(lm, datamodule=ldm)
