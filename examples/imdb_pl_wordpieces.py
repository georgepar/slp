import torch
import torch.nn as nn

from torch.optim import Adam
from loguru import logger
from torchnlp.datasets import imdb_dataset  # type: ignore

from slp.plbind.dm import PLDataModuleFromCorpus
from slp.plbind.module import RnnPLModule
from slp import configure_logger
from slp.data.collators import SequenceClassificationCollator
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN

import pytorch_lightning as pl

EXPERIMENT_NAME = "smt-sentiment-classification"

configure_logger(f"logs/{EXPERIMENT_NAME}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

collate_fn = SequenceClassificationCollator(device="cpu")


if __name__ == "__main__":
    train, test = imdb_dataset(directory="../data/", train=True, test=True)

    raw_train = [d["text"] for d in train]
    labels_train = [d["sentiment"] for d in train]

    raw_test = [d["text"] for d in test]
    labels_test = [d["sentiment"] for d in test]

    ldm = PLDataModuleFromCorpus(
        raw_train,
        labels_train,
        test=raw_test,
        test_labels=labels_test,
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
