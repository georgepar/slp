import copy

import torch
import torch.nn.functional as F

from loguru import logger

from torchvision.transforms import Compose, ToTensor, Normalize  # type: ignore
from torchvision.datasets import MNIST  # type: ignore
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from ignite.metrics import Loss, Accuracy

from slp.trainer import Trainer


DEBUG = False


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
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True),
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False),
        batch_size=val_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_data_loaders(32, 32)
    model = Net()
    optimizer = Adam(model.parameters(), lr=1e-2)
    criterion = nn.NLLLoss()
    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    trainer = Trainer(
        model,
        optimizer,
        checkpoint_dir="../checkpoints/" if not DEBUG else None,
        metrics=metrics,
        non_blocking=False,
        patience=1,
        loss_fn=criterion,
    )
    if DEBUG:
        logger.info("Starting end to end test")
        print("--------------------------------------------------------------")
        trainer.fit_debug(train_loader, val_loader)
        logger.info("Overfitting single batch")
        print("--------------------------------------------------------------")
        trainer.overfit_single_batch(train_loader)
    else:
        trainer.fit(train_loader, val_loader, epochs=50)
