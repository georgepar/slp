import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
import pytorch_lightning as pl

from torchvision.transforms import Compose, ToTensor, Normalize  # type: ignore
from torchvision.datasets import MNIST  # type: ignore

from loguru import logger

from slp import configure_logging
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.module import PLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.plbind.helpers import FromLogits


pl.utilities.seed.seed_everything(seed=42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

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


if __name__ == "__main__":
    EXPERIMENT_NAME = "mnist-classification"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

    train, test = get_data()

    ldm = PLDataModuleFromDatasets(
        train, test=test, batch_size=128, batch_size_eval=256
    )

    model = Net()
    optimizer = Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    lm = PLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(EXPERIMENT_NAME, max_epochs=100, gpus=1)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path='best', test_dataloaders=ldm.test_dataloader())
