import copy

import torch
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import Compose, Normalize, ToTensor  # type: ignore

from slp.modules.rnn import RNN
from slp.util import log

from slp.data.collators import SequenceClassificationCollator
from slp.trainer import SequentialTrainer as Trainer

DEBUG = False

collator = SequenceClassificationCollator()

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bidirectional=True):
        super().__init__()
        self.encoder = RNN(input_size, hidden_size, bidirectional=bidirectional)
        out_size = hidden_size if not bidirectional else 2 * hidden_size
        self.clf = nn.Linear(out_size, num_classes)

    def forward(self, x, lengths):
        _, x, _ = self.encoder(x, lengths)
        out = self.clf(x)
        return out


def squeeze(x):
    return x.squeeze()


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), squeeze])
    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collator
    )
    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False),
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collator
    )

    return train_loader, val_loader


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_data_loaders(32, 32)
    model = Net(28, 20, 10)
    optimizer = Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
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
        log.info("Starting end to end test")
        print("--------------------------------------------------------------")
        trainer.fit_debug(train_loader, val_loader)
        log.info("Overfitting single batch")
        print("--------------------------------------------------------------")
        trainer.overfit_single_batch(train_loader)
    else:
        trainer.fit(train_loader, val_loader, epochs=50)
