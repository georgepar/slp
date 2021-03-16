import pytorch_lightning as pl
import torch.nn as nn
from loguru import logger
from torch.optim import Adam, AdamW
from torchnlp.datasets import imdb_dataset  # type: ignore

from slp.data.collators import SequenceClassificationCollator
from slp.modules.classifier import TransformerTokenSequenceClassifier
from slp.plbind.dm import PLDataModuleFromCorpus
from slp.plbind.helpers import FromLogits
from slp.plbind.module import TransformerClassificationPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging

if __name__ == "__main__":

    pl.utilities.seed.seed_everything(seed=42)
    MAX_LENGTH = 1024
    collate_fn = SequenceClassificationCollator(device="cpu", max_length=MAX_LENGTH)

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
        batch_size=64,
        batch_size_eval=64,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=1,
        lower=True,
        tokenizer="bert-base-uncased",
    )
    ldm.setup()
    model = TransformerTokenSequenceClassifier(
        2,
        vocab_size=ldm.vocab_size,
        max_length=2 * MAX_LENGTH,
        nystrom=True,
        kernel_size=33,
        num_layers=2,
        num_heads=4,
        hidden_size=256,
        inner_size=512,
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    lm = TransformerClassificationPLModule(
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
