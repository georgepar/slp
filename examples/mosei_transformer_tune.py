import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from loguru import logger
from ray import tune
from slp.config.config_parser import make_cli_parser, parse_config
from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.data.cmusdk import mosei
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSEI
from slp.modules.classifier import TransformerLateFusionClassifier
from slp.plbind import (FromLogits, PLDataModuleFromDatasets, PLModule,
                        make_trainer, make_trainer_for_ray_tune)
from slp.plbind.dm import PLDataModuleFromDatasets, split_data
from slp.plbind.helpers import FromLogits
from slp.plbind.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7, MoseiF1
from slp.plbind.module import MultimodalTransformerClassificationPLModule
from slp.util.log import configure_logging
from slp.util.mosei import get_mosei_parser
from slp.util.tuning import run_tuning
from torch.optim import AdamW


def get_data(
    seed=None,
    remove_pauses=False,
    pad="front",
    max_length=-1,
    train=True,
    val=True,
    test=False,
    modalities={"text", "audio", "visual"},
):
    pad_front = False
    pad_back = False

    if pad == "front":
        pad_front = True
    elif pad == "back":
        pad_back = True

    # with FileLock(os.path.expanduser("~/data.lock")):
    train_data, dev_data, test_data, w2v = mosei(
        "/data/scratch/efthygeo/mosei_final_aligned",
        pad_front=pad_front,
        pad_back=pad_back,
        max_length=max_length,
        remove_pauses=remove_pauses,
        modalities=modalities,
        already_aligned=True,
        align_features=False,
        # cache="/home/efthygeo/experimental/pytorch-slp/slp/cache/mosei.p",
    )

    for x in train_data:
        if "glove" in x:
            x["text"] = x["glove"]

    for x in dev_data:
        if "glove" in x:
            x["text"] = x["glove"]

    for x in test_data:
        if "glove" in x:
            x["text"] = x["glove"]

    train = MOSEI(train_data, modalities=modalities, text_is_tokens=False)
    dev = MOSEI(dev_data, modalities=modalities, text_is_tokens=False)
    test = MOSEI(test_data, modalities=modalities, text_is_tokens=False)

    if not test:
        test = None

    return train, dev, test


def train_mosei(config, train=None, val=None):
    # Convert dictionary to omegaconf dictconfig object
    config = OmegaConf.create(config)
    modalities = set(config.modalities)

    train, val, _ = get_data(
        remove_pauses=config.preprocessing.remove_pauses,
        pad=config.preprocessing.pad,
        max_length=config.preprocessing.max_length,
        modalities=modalities,
    )

    collate_fn = MultimodalSequenceClassificationCollator(
        device="cpu", modalities=modalities
    )

    # Create data module
    config.data.batch_size_eval = config.data.batch_size
    ldm = PLDataModuleFromDatasets(
        train,
        val=val,
        no_test_set=True,
        batch_size=config.data.batch_size,
        batch_size_eval=config.data.batch_size_eval,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )

    # Create model, optimizer, criterion, scheduler

    model = TransformerLateFusionClassifier(
        config.model.feature_sizes,
        1,
        max_length=1024,
        nystrom=False,
        kernel_size=config.model.kernel_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        hidden_size=config.model.hidden_size,
        inner_size=config.model.inner_size_multiple * config.model.hidden_size,
        prenorm=False,
        scalenorm=config.model.scalenorm,
        multi_modal_drop=config.model.multi_modal_drop,
        p_mmdrop=config.model.p_mmdrop,
        # p_drop_modalities=config.model.p_drop_modalities,
    )

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)

    criterion = nn.L1Loss()

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
        metrics={
            "acc2": MoseiAcc2(exclude_neutral=True),
            "acc2_zero": MoseiAcc2(exclude_neutral=False),
            "acc5": MoseiAcc5(),
            "acc7": MoseiAcc7(),
            "f1": MoseiF1(exclude_neutral=True),
            "f1_zero": MoseiF1(exclude_neutral=False),
            "mae": torchmetrics.MeanAbsoluteError(),
        },
    )

    # lm = MultimodalTransformerClassificationPLModule(
    #     model,
    #     optimizer,
    #     criterion,
    #     lr_scheduler=lr_scheduler,
    #     hparams=config,
    # )

    # Map Lightning metrics to ray tune metris.
    metrics_map = {"validation_loss": "best_score",
                   "val_accuracy": "acc2",
                   "val_f1": "f1",
                   "acc5": "acc5",
                   "acc7": "acc7"}
    assert (
        config["tune"]["metric"] in metrics_map.keys()
    ), "Metrics mapping should contain the metric you are trying to optimize"
    # Train model
    trainer = make_trainer_for_ray_tune(metrics_map=metrics_map, **config.trainer)

    trainer.fit(lm, datamodule=ldm)


def configure_search_space(config):
    config["preprocessing"] = {
        "remove_pauses": False,
        # "remove_pauses": tune.choice([True, False]),
        "pad": "front",
        # "pad": tune.choice(["front", "back"]),
        "max_length": 75,
        # "max_length": tune.choice([-1, 75]),
    }
    model_tune = {
        "hidden_size": 128,
        # "hidden_size": tune.qrandint(64, 384, q=8),
        # "inner_size":  tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        # "inner_size_multiple": tune.qrandint(2, 4, q=2),
        "inner_size_multiple": 2,
        "num_heads": 2,
        # "num_heads": tune.choice([2, 4, 8]),
        # "num_layers": tune.randint(2, 4),
        "num_layers": 4,
        # "num_layers": 3,
        # "prenorm": tune.choice([False, True]),
        "prenorm": False,
        "scalenorm": True,
        # "scalenorm": tune.choice([False, True]),
        # "kernel_size": tune.choice([None, 11, 33]),
        "kernel_size": 33,
        "dropout": 0.3,
        "p_mmdrop": tune.grid_search([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
        # "p_mmdrop": tune.uniform(0.1, 0.5),
        # "multi_modal_drop": "none",
        "multi_modal_drop": tune.grid_search(["mmdrop_hard", "mmdrop_soft"]),
        # "multi_modal_drop": tune.grid_search(["dropout", "mmdrop_hard", "mmdrop_soft", "none"]),
        # "mmdrop_mode": tune.grid_search(["hard", "soft"])
        # "mmdrop_mode": tune.choice(["hard", "soft"])
    }
    config["model"].update(model_tune)
    # config["model"]["inner_size"] = tune.sample_from(lambda spec: spec.config.hidden_size * np.random.choice([2, 4]))

    # config["inner_size_multiple"] * config["model"]["hidden_size"]
    # config["lr_scheduler"] = tune.choice([False, True])
    config["lr_scheduler"] = True
    # config["lr_schedule"]["patience"] = tune.randint(2, 5)
    config["lr_schedule"]["patience"] = 3
    # config["lr_schedule"]["factor"] = tune.loguniform(0.1, 0.5)
    config["lr_schedule"]["factor"] = 0.2
    # config["optimizer"] = tune.choice(["SGD", "Adam", "AdamW"])
    config["optimizer"] = "AdamW"
    config["optim"]["lr"] = 2e-4
    # config["optim"]["lr"] = tune.loguniform(1e-5, 1e-2)
    # config["optim"]["lr"] = 1e-4
    # config["optim"]["weight_decay"] = tune.loguniform(1e-4, 1e-1)
    config["optim"]["weight_decay"] = 1e-3
    # config["data"]["batch_size"] = tune.choice([8, 16, 32, 64])
    config["data"]["batch_size"] = 8
    # config["data"]["batch_size"] = tune.choice([8, 16, 32])
    # config["trainer"]["gradient_clip_val"] = tune.choice([0, 1.0])
    # config["trainer"]["gradient_clip_val"] = 0.1

    return config


if __name__ == "__main__":
    # parser
    parser = get_mosei_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)

    config = parse_config(parser, parser.parse_args().config)

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
    config["wandb"]["project"] = "tuning-mosei-transformer-architecture-tareted-bottleneck"
    # config["trainer"]["max_epochs"] = 15
    config = OmegaConf.create(config)

    # Handle train / val splitting.
    # All trials should run on the same validation set
    best_config = run_tuning(
        config,  # type: ignore
        "configs/best.mosei.tune.yml",
        train_mosei,
        configure_search_space,
    )
