import sys

import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from loguru import logger
from slp.config.config_parser import make_cli_parser, parse_config
from slp.data.cmusdk import mosei
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSEI
from slp.modules.classifier import RNNSymAttnFusionRNNClassifier
from slp.modules.baseline import AudioVisualTextClassifier
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.helpers import FromLogits
from slp.plbind.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7, MoseiF1
from slp.plbind.module import RnnPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from slp.util.mosei import get_mosei_parser
from slp.util.system import is_file, safe_mkdirs
from torch.optim import Adam
import torch.optim as optim


if __name__ == "__main__":
    # parser
    parser = get_mosei_parser()
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
        modalities=modalities,
        max_length=-1,
        pad_back=config.preprocessing.pad_back,
        pad_front=config.preprocessing.pad_front,
        remove_pauses=config.preprocessing.remove_pauses,
        already_aligned=config.preprocessing.already_aligned,
        align_features=config.preprocessing.align_features,
        cache="./cache/mosei_avt_unpadded.p",
        # cache="./cache/mosei_avt.p",
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

    # model = RNNSymAttnFusionRNNClassifier(
    #     feature_sizes,
    #     1,
    #     num_layers=config.model.num_layers,
    #     batch_first=config.model.batch_first,
    #     bidirectional=config.model.bidirectional,
    #     packed_sequence=config.model.packed_sequence,
    #     merge_bi=config.model.merge_bi,
    #     rnn_type=config.model.rnn_type,
    #     attention=config.model.attention,
    #     hidden_size=config.model.hidden_size,
    #     num_heads=config.model.num_heads,
    #     max_length=config.model.max_length,
    #     dropout=config.model.dropout,
    #     nystrom=False,
    #     multi_modal_drop=config.model.multi_modal_drop,
    #     p_mmdrop=config.model.p_mmdrop,
    #     mmdrop_before_fuse=config.model.mmdrop_before_fuse,
    #     mmdrop_after_fuse=config.model.mmdrop_after_fuse,
    #     p_drop_modalities=config.model.p_drop_modalities,

    model = AudioVisualTextClassifier(
        feature_sizes,
        1,
        num_layers=config.model.num_layers,
        batch_first=config.model.batch_first,
        bidirectional=config.model.bidirectional,
        packed_sequence=config.model.packed_sequence,
        merge_bi=config.model.merge_bi,
        rnn_type=config.model.rnn_type,
        attention=config.model.attention,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        max_length=config.model.max_length,
        dropout=config.model.dropout,
        nystrom=False,
        multi_modal_drop=config.model.multi_modal_drop,
        p_mmdrop=config.model.p_mmdrop,
        mmdrop_before_fuse=config.model.mmdrop_before_fuse,
        mmdrop_after_fuse=config.model.mmdrop_after_fuse,
        p_drop_modalities=config.model.p_drop_modalities,
    )
    # model = TransformerSymmetricAttnFusionClassifier(
    #     feature_sizes,
    #     1,
    #     max_length=1024,
    #     nystrom=False,
    #     kernel_size=config.model.kernel_size,
    #     num_landmarks=config.model.num_landmarks,
    #     num_layers=config.model.num_layers,
    #     num_heads=config.model.num_heads,
    #     dropout=config.model.dropout,
    #     hidden_size=config.model.hidden_size,
    #     inner_size=config.model.inner_size,
    #     prenorm=False,
    #     scalenorm=config.model.scalenorm,
    #     multi_modal_drop=config.model.multi_modal_drop,
    #     p_mmdrop=config.model.p_mmdrop,
    #     mmdrop_before_fuse=config.model.mmdrop_before_fuse,
    #     mmdrop_after_fuse=config.model.mmdrop_after_fuse,
    #     # p_drop_modalities=config.model.p_drop_modalities,
    # )

    print(model)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )

    lr_scheduler = None

    if config.lr_schedule:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    criterion = nn.L1Loss()

    lm = RnnPLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
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

    from slp.util.mosei import test_mosei

    results = test_mosei(lm, ldm, trainer, modalities)

    import csv
    import os

    csv_folder_path = os.path.join(
        config.trainer.experiments_folder, config.trainer.experiment_name, "results_csv"
    )

    csv_name = os.path.join(csv_folder_path, "results.csv")
    fieldnames = list(results.keys())

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
        writer.writerow(results)

    # results = trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())

    # import csv
    # import os

    # csv_folder_path = os.path.join(
    #     config.trainer.experiments_folder, config.trainer.experiment_name, "results_csv"
    # )

    # csv_name = os.path.join(csv_folder_path, "results.csv")
    # fieldnames = list(results[0].keys())

    # if is_file(csv_name):
    #     # folder already exits and so does the .csv
    #     csv_exists = True
    #     print(f"csv already exists")
    # else:
    #     csv_exists = False
    #     safe_mkdirs(csv_folder_path)

    # with open(csv_name, "a") as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    #     if not csv_exists:
    #         writer.writeheader()
    #     writer.writerow(results[0])
