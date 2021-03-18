import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from loguru import logger
from slp.data.cmusdk import mosei
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSEI
from slp.modules.classifier import TransformerLateFusionClassifier
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.config.config_parser import make_cli_parser, parse_config
from slp.plbind.helpers import FromLogits
from slp.plbind.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7, MoseiF1
from slp.plbind.module import MultimodalTransformerClassificationPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from slp.util.system import safe_mkdirs, is_file
from torch.optim import AdamW


def get_parser():
    parser = ArgumentParser("Transformer-MOSEI example")

    # model hyperparameters
    parser.add_argument(
        "--max_length",
        dest="model.max_length",
        type=int,
        default=1024,
        help="Sequence max length for transformer layers",
    )
    parser.add_argument(
        "--kernel_size",
        dest="model.kernel_size",
        type=int,
        default=33,
        help="Kernel Size used for residual trick in transformer layers",
    )
    parser.add_argument(
        "--use_nystrom",
        dest="model.nystrom",
        default=False,
        action="store_true",
        help="Use nystrom self-attention approximation",
    )
    parser.add_argument(
        "--num_landmarks",
        dest="model.num_landmarks",
        type=int,
        default=32,
        help="Number of landmarks used for nystrom approximation",
    )
    parser.add_argument(
        "--num_layers",
        dest="model.num_layers",
        type=int,
        default=3,
        help="Number of Transformer Layers",
    )
    parser.add_argument(
        "--num_heads",
        dest="model.num_heads",
        type=int,
        default=4,
        help="Number of Transformer Heads per Layer",
    )
    parser.add_argument(
        "--dropout",
        dest="model.dropout",
        type=float,
        default=0.3,
        help="Drop probability for each Transformer Layer",
    )
    parser.add_argument(
        "--hidden_size",
        dest="model.hidden_size",
        type=int,
        default=100,
        help="Hidden Size for each Transformer Layer",
    )
    parser.add_argument(
        "--inner_size",
        dest="model.inner_size",
        type=int,
        default=200,
        help="Inner Size for each Transformer Layer",
    )
    parser.add_argument(
        "--prenorm",
        dest="model.prenorm",
        default=False,
        action="store_true",
        help="Use the normalization before residual."
        "Default value is False as in Vaswani.",
    )
    parser.add_argument(
        "--scalenorm",
        dest="model.scalenorm",
        default=False,
        action="store_true",
        help="Use scalenorm (L2) instead of LayerNorm (Vaswani).",
    )
    parser.add_argument(
        "--mmdrop",
        dest="model.multi_modal_drop",
        type=str,
        default="dropout",
        choices=["dropout", "mmdrop", "both", "none"],
        help="Which dropout is applied to the late fusion stage",
    )
    parser.add_argument(
        "--mmdrop-mode",
        dest="model.mmdrop_mode",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="two versions of mmdrop",
    )
    parser.add_argument(
        "--p-mmdrop",
        dest="model.p_mmdrop",
        type=float,
        default=0.33,
        help="probability of droping 1/3 modlities",
    )
    parser.add_argument(
        "--p-drop-modalities",
        dest="model.p_drop_modalities",
        default=[0.33, 0.33, 0.33],
        help="Per modal drop rate",
    )

    # dataset specifications
    parser.add_argument(
        "--pad-front",
        dest="preprocessing.pad_front",
        action="store_true",
        help="Handles front padding. Default is True",
    )
    parser.add_argument(
        "--pad-back",
        dest="preprocessing.pad_back",
        action="store_true",
        help="Handles back padding. Default is False",
    )
    parser.add_argument(
        "--remove-pauses",
        dest="preprocessing.remove_pauses",
        default=False,
        action="store_true",
        help="When used removes pauses from dataset",
    )
    parser.add_argument(
        "--already-unaligned",
        dest="preprocessing.already_aligned",
        action="store_true",
        help="When used indicates unaligned data otherwise"
        " aligned scenario is used.",
    )
    parser.add_argument(
        "--do-align",
        dest="preprocessing.align_features",
        action="store_true",
        help="When used automatically aligns the features.",
    )

    return parser


if __name__ == "__main__":
    # parser
    parser = get_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)

    config = parse_config(parser, parser.parse_args().config)

    # if config.trainer.experiment_name != "mosei-transformer":
    #     config.trainer.experiment_name = "mosei-transformer"

    configure_logging(f"logs/{config.trainer.experiment_name}")
    modalities = set(config.modalities)
    max_length = config.model.max_length
    collate_fn = MultimodalSequenceClassificationCollator(
        device="cpu", modalities=modalities
    )

    train_data, dev_data, test_data, w2v = mosei(
        "data/mosei_final_aligned/",
        pad_back=config.preprocessing.pad_back,
        pad_front=config.preprocessing.pad_front,
        max_length=-1,
        modalities=modalities,
        remove_pauses=config.preprocessing.remove_pauses,
        already_aligned=config.preprocessing.already_aligned,
        align_features=config.preprocessing.align_features,
        cache="./cache/mosei.p",
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

    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,
        batch_size=config.data.batch_size,
        batch_size_eval=config.data.batch_size_eval,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )
    ldm.setup()

    feature_sizes = config.model.feature_sizes

    model = TransformerLateFusionClassifier(
        feature_sizes,
        1,
        max_length=2 * config.model.max_length,
        nystrom=config.model.nystrom,
        kernel_size=config.model.kernel_size,
        num_landmarks=config.model.num_landmarks,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        hidden_size=config.model.hidden_size,
        inner_size=config.model.inner_size,
        prenorm=config.model.prenorm,
        scalenorm=config.model.scalenorm,
        multi_modal_drop=config.model.multi_modal_drop,
        mmdrop_mode=config.model.mmdrop_mode,
        p_mmdrop=config.model.p_mmdrop,
        p_drop_modalities=config.model.p_drop_modalities,
    )

    print(model)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    criterion = nn.L1Loss()

    lm = MultimodalTransformerClassificationPLModule(
        model,
        optimizer,
        criterion,
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

    trainer = make_trainer(**config.trainer)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    results = trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())

    import os
    import csv

    csv_folder_path = os.path.join(
        config.trainer.experiments_folder, config.trainer.experiment_name, "results_csv"
    )

    csv_name = os.path.join(csv_folder_path, "results.csv")
    fieldnames = list(results[0].keys())
    if is_file(csv_name):
        # folder already exits and so does the .csv
        csv_exists = True
        print(f"csv already exists")
    else:
        csv_exists = False
        safe_mkdirs(csv_folder_path)

    with open(csv_name, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(results[0])
