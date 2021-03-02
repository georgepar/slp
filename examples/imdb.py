import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl

from loguru import logger
from torchnlp.datasets import imdb_dataset  # type: ignore

from slp.plbind.dm import PLDataModuleFromCorpus
from slp.plbind.module import RnnPLModule
from slp.util.log import configure_logging
from slp.data.collators import SequenceClassificationCollator
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.plbind.trainer import make_trainer, watch_model
from slp.plbind.helpers import FromLogits


collate_fn = SequenceClassificationCollator(device="cpu")


if __name__ == "__main__":
    EXPERIMENT_NAME = "imdb-words-sentiment-classification"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

    train, test = imdb_dataset(directory="./data/", train=True, test=True)

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
        tokens="words",
        embeddings_file="./cache/glove.6B.50d.txt",
        embeddings_dim=50,
        lower=True,
        limit_vocab_size=-1,
        lang="en_core_web_md",
    )

    encoder = WordRNN(
        256,
        embeddings=ldm.embeddings,
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

    lm = RnnPLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(
        EXPERIMENT_NAME,
        max_epochs=100,
        gpus=1,
        save_top_k=1,
    )
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
