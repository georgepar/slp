import torch
import torch.nn as nn

from torch.optim import Adam
from loguru import logger
from torchnlp.datasets import smt_dataset  # type: ignore

from slp.plbind.dm import PLDataModuleFromCorpus
from slp.plbind.module import RnnPLModule
from slp import configure_logging
from slp.data.collators import SequenceClassificationCollator
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN

import pytorch_lightning as pl

EXPERIMENT_NAME = "smt-sentiment-classification"

configure_logging(f"logs/{EXPERIMENT_NAME}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

collate_fn = SequenceClassificationCollator(device="cpu")


if __name__ == "__main__":
    train, dev = smt_dataset(directory="../data/", train=True, dev=True)

    raw_train = [d["text"] for d in train]
    labels_train = [d["label"] for d in train]

    raw_dev = [d["text"] for d in dev]
    labels_dev = [d["label"] for d in dev]

    ldm = PLDataModuleFromCorpus(
        raw_train,
        labels_train,
        val=raw_dev,
        val_labels=labels_dev,
        batch_size=8,
        batch_size_eval=32,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=1,
        tokens="wordpieces",
        bert_model="bert-base-uncased",
        lower=True,
    )

    encoder = WordRNN(
        256,
        vocab_size=ldm.vocab_size,
        embeddings_dim=300,
        bidirectional=True,
        merge_bi="sum",
        packed_sequence=True,
        finetune_embeddings=True,
        attention=True,
    )

    model = Classifier(encoder, encoder.out_size, 3)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    lm = RnnPLModule(model, optimizer, criterion)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(lm, datamodule=ldm)
