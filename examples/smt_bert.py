import torch
import torch.nn as nn
import pytorch_lightning as pl

from argparse import ArgumentParser

from torch.optim import Adam
from loguru import logger
from torchnlp.datasets import smt_dataset  # type: ignore

from slp.config.config_parser import make_cli_parser, parse_config
from slp.plbind.dm import PLDataModuleFromCorpus
from slp.plbind.module import BertPLModule
from slp.util.log import configure_logging
from slp.data.collators import SequenceClassificationCollator
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.plbind.trainer import make_trainer, watch_model
from slp.plbind.helpers import FromLogits

from transformers import BertForSequenceClassification, AdamW


collate_fn = SequenceClassificationCollator(device="cpu")


def get_parser():
    parser = ArgumentParser("Bert finetuning for SST-2")
    parser.add_argument(
        "--binary",
        dest="binary",
        action="store_true",
        help="Perform binary classification. If False performs fine-grained 5-class classification",
    )
    return parser


def get_data(config):
    train, dev, test = smt_dataset(
        directory="../data/",
        train=True,
        dev=True,
        test=True,
        fine_grained=True,
    )

    def filter_neutrals(data, labels):
        logger.info("Filtering neutral labels for binary task")

        new_data, new_labels = [], []

        for d, l in zip(data, labels):
            # l positive or very positive
            if "positive" in l:
                new_data.append(d)
                new_labels.append("positive")
            # l negative or very negative
            elif "negative" in l:
                new_data.append(d)
                new_labels.append("negative")
            else:
                continue

        return new_data, new_labels

    raw_train = [d["text"] for d in train]
    labels_train = [d["label"] for d in train]

    raw_dev = [d["text"] for d in dev]
    labels_dev = [d["label"] for d in dev]

    raw_test = [d["text"] for d in dev]
    labels_test = [d["label"] for d in dev]

    num_labels = 5
    if config.binary:
        raw_train, labels_train = filter_neutrals(raw_train, labels_train)
        raw_dev, labels_dev = filter_neutrals(raw_dev, labels_dev)
        raw_test, labels_test = filter_neutrals(raw_test, labels_test)
        num_labels = 2

    return (
        raw_train,
        labels_train,
        raw_dev,
        labels_dev,
        raw_test,
        labels_test,
        num_labels,
    )


if __name__ == "__main__":
    parser = get_parser()
    parser = make_cli_parser(parser, PLDataModuleFromCorpus)

    args = parser.parse_args()
    config_file = args.config

    config = parse_config(parser, config_file)
    # Set these by default.
    config.hugging_face_model = config.data.tokenizer
    config.data.add_special_tokens = True
    config.data.lower = "uncased" in config.hugging_face_model

    if config.trainer.experiment_name == "experiment":
        config.trainer.experiment_name = "finetune-bert-smt"

    configure_logging(f"logs/{config.trainer.experiment_name}")

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    (
        raw_train,
        labels_train,
        raw_dev,
        labels_dev,
        raw_test,
        labels_test,
        num_labels,
    ) = get_data(config)

    ldm = PLDataModuleFromCorpus(
        raw_train,
        labels_train,
        val=raw_dev,
        val_labels=labels_dev,
        test=raw_test,
        test_labels=labels_test,
        collate_fn=collate_fn,
        **config.data,
    )

    model = BertForSequenceClassification.from_pretrained(
        config.hugging_face_model, num_labels=num_labels
    )

    logger.info(model)

    # Leave this hardcoded for now.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    lm = BertPLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(**config.trainer)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
