import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

from loguru import logger
from torchnlp.datasets import smt_dataset  # type: ignore

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
    EXPERIMENT_NAME = "smt-words-sentiment-classification"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

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
        tokens="words",
        embeddings_file="./cache/glove.6B.50d.txt",
        embeddings_dim=50,
        lower=True,
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
