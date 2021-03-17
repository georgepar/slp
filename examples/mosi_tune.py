import math
import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from loguru import logger
from ray import tune
from slp.config.config_parser import make_cli_parser, parse_config
from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.data.cmusdk import mosi
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSI
from slp.modules.classifier import TransformerLateFusionClassifier
from slp.plbind import (
    FromLogits,
    PLDataModuleFromDatasets,
    PLModule,
    make_trainer,
    make_trainer_for_ray_tune,
)
from slp.plbind.dm import split_data
from slp.plbind.module import MultimodalTransformerClassificationPLModule
from slp.util.log import configure_logging
from slp.util.tuning import run_tuning

modalities = {"text", "audio", "visual"}


def get_data(
    seed=None,
    remove_pauses=False,
    pad="front",
    max_length=-1,
    train=True,
    val=True,
    test=False,
):
    pad_front = False
    pad_back = False

    if pad == "front":
        pad_front = True
    elif pad == "back":
        pad_back = True

    # with FileLock(os.path.expanduser("~/data.lock")):
    train_data, dev_data, test_data, w2v = mosi(
        "/home/efthygeo/experimental/pytorch-slp/slp/data/mosi_final_aligned/",
        pad_front=pad_front,
        pad_back=pad_back,
        max_length=max_length,
        remove_pauses=remove_pauses,
        modalities=modalities,
        already_aligned=True,
        align_features=False,
        # cache="./cache/mosi.p",
    )

    train = MOSI(train_data, modalities=modalities, binary=False, text_is_tokens=False)
    dev = MOSI(dev_data, modalities=modalities, binary=False, text_is_tokens=False)
    test = MOSI(test_data, modalities=modalities, binary=False, text_is_tokens=False)

    if not test:
        test = None

    return train, dev, test


def get_parser():
    parser = ArgumentParser("MOSI Tuning")
    parser.add_argument(
        "--hidden",
        dest="model.hidden_size",
        type=int,
        default=100,
        help="Intermediate hidden layers for linear module",
    )

    parser.add_argument(
        "--inner",
        dest="model.inner_size",
        type=int,
        default=200,
        help="Inner size",
    )

    parser.add_argument(
        "--heads",
        dest="model.num_heads",
        type=int,
        default=2,
        help="Number of heads",
    )

    parser.add_argument(
        "--layers",
        dest="model.num_layers",
        type=int,
        default=2,
        help="Number of transformer layers",
    )

    parser.add_argument(
        "--prenorm", dest="model.prenorm", action="store_true", help="Use prenorm"
    )

    parser.add_argument(
        "--scalenorm", dest="model.scalenorm", action="store_true", help="Use scalenorm"
    )

    parser.add_argument("--dropout", dest="model.dropout", default=0.1, help="Dropout")

    parser.add_argument(
        "--kernel-size",
        dest="model.kernel_size",
        default=None,
        help="Residual convolution in attention",
    )

    return parser


def train_mosi(config, train=None, val=None):
    # Convert dictionary to omegaconf dictconfig object
    config = OmegaConf.create(config)

    train, val, _ = get_data(
        remove_pauses=config.preprocessing.remove_pauses,
        pad=config.preprocessing.pad,
        max_length=config.preprocessing.max_length,
    )
    collate_fn = MultimodalSequenceClassificationCollator(device="cpu")
    # Create data module
    config.data.batch_size_eval = config.data.batch_size
    ldm = PLDataModuleFromDatasets(
        train,
        val=val,
        seed=config.seed,
        no_test_set=True,
        collate_fn=collate_fn,
        **config.data
    )

    feature_sizes = {"audio": 74, "visual": 35, "text": 300}
    # Create model, optimizer, criterion, scheduler
    model = TransformerLateFusionClassifier(
        feature_sizes,
        1,
        max_length=512,
        nystrom=False,
        kernel_size=config.model.kernel_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        hidden_size=config.model.hidden_size,
        inner_size=config.model.inner_size_multiple * config.model.hidden_size,
        # inner_size=config.model.inner_size,
        prenorm=config.model.prenorm,
        scalenorm=config.model.scalenorm,
    )

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
    criterion = nn.MSELoss()

    lr_scheduler = None

    if config.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    lm = MultimodalTransformerClassificationPLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        hparams=config,
    )

    # Map Lightning metrics to ray tune metris.
    metrics_map = {"validation_loss": "best_score"}
    assert (
        config["tune"]["metric"] in metrics_map.keys()
    ), "Metrics mapping should contain the metric you are trying to optimize"
    # Train model
    trainer = make_trainer_for_ray_tune(metrics_map=metrics_map, **config.trainer)

    trainer.fit(lm, datamodule=ldm)


def configure_search_space(config):
    config["preprocessing"] = {
        "remove_pauses": tune.choice([True, False]),
        "pad": tune.choice(["front", "back"]),
        "max_length": tune.choice([-1, 20, 50, 75]),
    }
    config["model"] = {
        "hidden_size": tune.qrandint(64, 384, q=8),
        # "inner_size":  tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        # "inner_size_multiple": tune.qrandint(2, 4, q=2),
        "inner_size_multiple": 2,
        "num_heads": tune.choice([2, 4, 8]),
        # "num_layers": tune.randint(1, 4),
        "num_layers": 3,
        # "prenorm": tune.choice([False, True]),
        "prenorm": False,
        "scalenorm": tune.choice([False, True]),
        # "kernel_size": tune.choice([None, 11, 33]),
        "kernel_size": 33,
        "dropout": tune.uniform(0.2, 0.4),
    }
    # config["model"]["inner_size"] = tune.sample_from(lambda spec: spec.config.hidden_size * np.random.choice([2, 4]))

    # config["inner_size_multiple"] * config["model"]["hidden_size"]
    # config["lr_scheduler"] = tune.choice([False, True])
    config["lr_scheduler"] = True
    # config["lr_schedule"]["patience"] = tune.randint(2, 5)
    config["lr_schedule"]["patience"] = 5
    # config["lr_schedule"]["factor"] = tune.loguniform(0.1, 0.5)
    config["lr_schedule"]["factor"] = 0.2
    # config["optimizer"] = tune.choice(["SGD", "Adam", "AdamW"])
    config["optimizer"] = "AdamW"
    # config["optim"]["lr"] = tune.loguniform(1e-4, 1e-2)
    config["optim"]["lr"] = 1e-4
    # config["optim"]["weight_decay"] = tune.loguniform(1e-4, 1e-1)
    config["optim"]["weight_decay"] = 5e-4
    # config["data"]["batch_size"] = tune.choice([8, 16, 32, 64])
    config["data"]["batch_size"] = tune.choice([8, 16])
    config["trainer"]["gradient_clip_val"] = tune.choice([0, 0.1])
    # config["trainer"]["gradient_clip_val"] = 0.1

    return config


if __name__ == "__main__":
    # SETUP ##################################################
    parser = get_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)  # type: ignore

    config = parse_config(parser, parser.parse_args().config)

    if config.trainer.experiment_name == "experiment":
        config.trainer.experiment_name = "mosi-tuning"

    configure_logging()

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    # These arguments may be provided from the command line or a config file
    config = OmegaConf.to_container(config)
    # config["tune"] = {
    #     "num_trials": 10,
    #     "cpus_per_trial": 1,
    #     "gpus_per_trial": 0.12,
    #     "metric": "accuracy",
    #     "mode": "max",
    # }
    config["wandb"] = {}
    config["wandb"]["project"] = "tuning-mosi-search-space-2"
    # config["trainer"]["max_epochs"] = 15
    config = OmegaConf.create(config)

    # Handle train / val splitting.
    # All trials should run on the same validation set
    best_config = run_tuning(
        config,  # type: ignore
        "configs/best.mosi.tune.yml",
        train_mosi,
        configure_search_space,
    )
