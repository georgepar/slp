import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from torchnlp.datasets import smt_dataset  # type: ignore

from slp import configure_logger
from slp.data.corpus import WordpieceCorpus
from slp.data.datasets import ClassificationCorpus
from slp.data.collators import SequenceClassificationCollator
from slp.data.transforms import ToTensor
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.trainer import SequentialTrainer

import pytorch_lightning as pl

EXPERIMENT_NAME = "smt-sentiment-classification"

configure_logger(f"logs/{EXPERIMENT_NAME}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

collate_fn = SequenceClassificationCollator(device="cpu")


class WordpieceCorpusDataModule(pl.LightningDataModule):
    def __init__(self):
        super().init__()


if __name__ == "__main__":
    train, dev = smt_dataset(directory="../data/", train=True, dev=True)

    raw_train = [d["text"] for d in train]
    labels_train = [d["label"] for d in train]

    raw_dev = [d["text"] for d in dev]
    labels_dev = [d["label"] for d in dev]

    corpus_train = WordpieceCorpus(
        raw_train,
        bert_model="bert-base-uncased",
        lower=True,
    )

    corpus_dev = WordpieceCorpus(
        raw_dev,
        bert_model="bert-base-uncased",
        lower=True,
    )

    to_tensor = ToTensor(device="cpu")

    dataset_train = ClassificationCorpus(corpus_train, labels_train).map(to_tensor)
    dataset_dev = ClassificationCorpus(corpus_dev, labels_dev).map(to_tensor)

    train_loader = DataLoader(
        dataset_train,
        batch_size=8,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        dataset_dev,
        batch_size=8,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    encoder = WordRNN(
        256,
        vocab_size=corpus_train.vocab_size,
        embeddings_dim=300,
        bidirectional=True,
        merge_bi="sum",
        packed_sequence=True,
        attention=True,
    )

    model = Classifier(encoder, encoder.out_size, 3)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    class SmtDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return dev_loader

    class Model(pl.LightningModule):
        def __init__(self, model, optimizer, criterion):
            super().__init__()
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

        def configure_optimizers(self):
            return self.optimizer

        def training_step(self, batch, batch_idx):
            inputs, targets, lengths = batch
            y_hat = self(inputs, lengths)
            loss = self.criterion(y_hat, targets)
            return loss

        def validation_step(self, batch, batch_idx):
            inputs, targets, lengths = batch
            y_hat = self(inputs, lengths)
            loss = self.criterion(y_hat, targets)
            return loss

    trainer = pl.Trainer(gpus=1)
    trainer.fit(Model(model, optimizer, criterion), datamodule=SmtDataModule())
